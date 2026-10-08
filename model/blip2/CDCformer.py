"""
CDCformer.py
=================================
Decoder-Only BLIP-2 OPT with Model Parallelism
  - Q-Former + Context Modules  → cuda:0  (轻量)
  - OPT-2.7B                  → cuda:1  (重量)
  - 无 ViT, 直接接收 image_embeds
  - 设备 & dtype 安全
  - 完整可运行
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from transformers import AutoTokenizer, OPTForCausalLM
from model.blip2.blip2 import Blip2Base


class CDCformer(nn.Module):
    """
    CDCformer
    ========
    Q-Former (cuda:0)  →  OPT-2.7B (cuda:1)
    """

    def __init__(
        self,
        num_query_token: int = 32,
        qformer_hidden_size: int = 768,
        opt_model: str = "/pretrained/weights/opt-2.7b",
        prompt: str = "",
        max_txt_len: int = 32,
    ):
        super().__init__()

        # ============================================================
        # 设备定义（严格分离）
        # ============================================================
        self.qformer_device = torch.device("cuda:0")   # 轻量部分
        self.opt_device      = torch.device("cuda:1")   # 重量部分

        # ============================================================
        # OPT Decoder (frozen, float16, cuda:1)
        # ============================================================
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        ).to(self.opt_device).eval()

        for p in self.opt_model.parameters():
            p.requires_grad = False

        self.eos_token_id = self.opt_tokenizer("\n").input_ids[0]

        # ============================================================
        # Q-Former (float16, cuda:0)
        # ============================================================
        self.Qformer, self.query_tokens = Blip2Base.init_Qformer(
            num_query_token, qformer_hidden_size
        )
        # 删除不需要的模块以节省显存
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output      = None
            layer.intermediate = None

        self.Qformer = self.Qformer.to(self.qformer_device).half()

        # ============================================================
        # 投影层：Q-Former → OPT hidden size (cuda:0)
        # ============================================================
        self.opt_proj = nn.Linear(
            qformer_hidden_size,
            self.opt_model.config.hidden_size,   # 2560
        ).to(self.qformer_device).half()

        # ============================================================
        # Prompt
        # ============================================================
        self.prompt     = prompt
        self.max_txt_len = max_txt_len
        _pt = self.opt_tokenizer(prompt, return_tensors="pt")
        self.prompt_length = _pt.attention_mask.sum(1)   # 在 CPU 上，后面会处理

        # ============================================================
        # Cross-image context modeling (cuda:0)
        # ============================================================
        D = qformer_hidden_size

        self.input_proj = nn.Linear(128, D).to(self.qformer_device).half()

        self.context1 = nn.Linear(D, D, bias=False).to(self.qformer_device).half()
        self.context2 = nn.Linear(D, D).to(self.qformer_device).half()
        self.gate1    = nn.Linear(D, D, bias=False).to(self.qformer_device).half()
        self.gate2    = nn.Linear(D, D).to(self.qformer_device).half()
        self.context3 = nn.Linear(3 * D, D).to(self.qformer_device).half()
        self.dropout   = nn.Dropout(0.5)

    # ============================================================
    # 工具函数
    # ============================================================
    def _to_half(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.half()
        return x

    # ============================================================
    # Reshape: [B, C, H, W] → [B, H*W, C]
    # ============================================================
    def _reshape_feat(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.flatten(2)          # [B, C, H*W]
        x = x.transpose(1, 2)     # [B, H*W, C]
        return x

    # ============================================================
    # Forward
    # ============================================================
    def forward(
        self,
        image_embeds_A: torch.Tensor,
        image_embeds_B: torch.Tensor,
        diff_enhance:   torch.Tensor,
        text_input:      list,
    ):
        dtype = torch.float16

        # ---- 1. 输入 reshape & 转到 cuda:0 ----
        featA = self._reshape_feat(self._to_half(image_embeds_A)).to(self.qformer_device)
        featB = self._reshape_feat(self._to_half(image_embeds_B)).to(self.qformer_device)
        diff  = self._reshape_feat(self._to_half(diff_enhance)).to(self.qformer_device)

        B = featA.size(0)

        # ---- 2. Dual-image context gating (cuda:0) ----
        def apply_context(x, d):
            ctx  = torch.tanh(self.context1(d) + self.context2(x))
            gate = torch.sigmoid(self.gate1(d) + self.gate2(x))
            return gate * self.dropout(ctx)

        ctx_a = apply_context(featA, diff)
        ctx_b = apply_context(featB, diff)

        fused_a = torch.cat([featA, diff, ctx_a], dim=-1)
        fused_b = torch.cat([featB, diff, ctx_b], dim=-1)

        featA = self.context3(fused_a)
        featB = self.context3(fused_b)

        image_embeds = torch.cat([featA, featB], dim=1)          # [B, 2*N, D]
        image_masks  = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=self.qformer_device
        )

        # ---- 3. Q-Former (cuda:0) ----
        query_tokens = self.query_tokens.expand(B, -1, -1).to(self.qformer_device).half()
        query_outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_masks,
            return_dict=True,
        )

        # image_tokens = self.opt_proj(query_outputs.last_hidden_state)   # [B, 32, 2560]
        image_tokens = query_outputs.last_hidden_state
        image_tokens = image_tokens.to(self.opt_device)                 # ✅ 移到 cuda:1
        image_masks  = image_masks.to(self.opt_device)

        # ---- 4. Text labels (cuda:1) ----
        text = [t + "\n" for t in text_input]
        text_tokens = self.opt_tokenizer(
            text, return_tensors="pt", padding="longest",
            truncation=True, max_length=self.max_txt_len,
        )
        text_tokens = {k: v.to(self.opt_device) for k, v in text_tokens.items()}

        labels = text_tokens["input_ids"].masked_fill(
            text_tokens["input_ids"] == self.opt_tokenizer.pad_token_id, -100
        )
        # prompt 部分不计算 loss
        plen = self.prompt_length.to(self.opt_device)
        if self.prompt:
            labels[:, :plen] = -100

        empty_labels = torch.full(
            (image_tokens.size(0), image_tokens.size(1)),
            -100, dtype=torch.long, device=self.opt_device,
        )
        labels = torch.cat([empty_labels, labels], dim=1)

        # ---- 5. OPT forward (cuda:1) ----
        text_embeds = self.opt_model.get_input_embeddings()(text_tokens["input_ids"])
        text_embeds = text_embeds.half()

        inputs_embeds  = torch.cat([image_tokens, text_embeds], dim=1)
        # attention_mask = torch.cat([image_masks, text_tokens["attention_mask"]], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.size()[:-1], dtype=torch.long, device=self.opt_device
        )

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return {"loss": outputs.loss}

    # ============================================================
    # Generate
    # ============================================================
    @torch.no_grad()
    def generate(
        self,
        image_embeds_A: torch.Tensor,
        image_embeds_B: torch.Tensor,
        diff_enhance:   torch.Tensor,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> list:
        dtype = torch.float16

        # ---- 1. Reshape & device (cuda:0) ----
        featA = self._reshape_feat(self._to_half(image_embeds_A)).to(self.qformer_device)
        featB = self._reshape_feat(self._to_half(image_embeds_B)).to(self.qformer_device)
        diff  = self._reshape_feat(self._to_half(diff_enhance)).to(self.qformer_device)

        B = featA.size(0)

        # ---- 2. Context gating (cuda:0) ----
        def apply_context(x, d):
            ctx  = torch.tanh(self.context1(d) + self.context2(x))
            gate = torch.sigmoid(self.gate1(d) + self.gate2(x))
            return gate * self.dropout(ctx)

        ctx_a = apply_context(featA, diff)
        ctx_b = apply_context(featB, diff)

        fused_a = torch.cat([featA, diff, ctx_a], dim=-1)
        fused_b = torch.cat([featB, diff, ctx_b], dim=-1)

        featA = self.context3(fused_a)
        featB = self.context3(fused_b)

        image_embeds = torch.cat([featA, featB], dim=1)
        image_masks  = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=self.qformer_device
        )

        # ---- 3. Q-Former (cuda:0) ----
        query_tokens = self.query_tokens.expand(B, -1, -1).to(self.qformer_device).half()
        q_out = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_masks,
            return_dict=True,
        )

        # image_tokens = self.opt_proj(q_out.last_hidden_state).to(self.opt_device)
        image_tokens = q_out.last_hidden_state.to(self.opt_device)
        image_masks = image_masks.to(self.opt_device)

        # ---- 4. Prompt (cuda:1) ----
        use_prompt = prompt or self.prompt
        use_prompt = [use_prompt] * B

        prompt_tokens = self.opt_tokenizer(
            use_prompt, return_tensors="pt", padding="longest",
            truncation=True, max_length=self.max_txt_len,
        )
        prompt_tokens = {k: v.to(self.opt_device) for k, v in prompt_tokens.items()}

        text_embeds = self.opt_model.get_input_embeddings()(prompt_tokens["input_ids"])
        text_embeds = text_embeds.half()

        inputs_embeds  = torch.cat([image_tokens, text_embeds], dim=1)
        # attention_mask = torch.cat([image_masks, prompt_tokens["attention_mask"]], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.size()[:-1], dtype=torch.long, device=self.opt_device
        )

        # ---- 5. Generate (cuda:1) ----
        outputs = self.opt_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            eos_token_id=self.eos_token_id,
            **kwargs,
        )

        return self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
