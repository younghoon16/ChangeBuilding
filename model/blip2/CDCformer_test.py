import logging
from typing import Optional
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, OPTForCausalLM
import transformers


class CDCformer(nn.Module):
    """
    Decoder-Only BLIP-2 OPT
    -------------------------
    ✅ 无 ViT
    ✅ 无 samples dict
    ✅ forward 直接接收 image_embeds
    ✅ Q-Former 保留
    ✅ OPT decoder-only
    ✅ 设备 & dtype 安全
    """

    def __init__(
        self,
        num_query_token=32,
        qformer_hidden_size=768,
        opt_model="/pretrained/weights/opt-2.7b",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        super().__init__()

        assert version.parse(transformers.__version__) >= version.parse("4.27")

        # ============================================================
        # OPT Decoder (frozen, float16)
        # ============================================================
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        ).eval()

        for p in self.opt_model.parameters():
            p.requires_grad = False

        self.eos_token_id = self.opt_tokenizer("\n").input_ids[0]

        # ============================================================
        # Q-Former (float16)
        # ============================================================
        from model.blip2.blip2 import Blip2Base
        self.Qformer, self.query_tokens = Blip2Base.init_Qformer(
            num_query_token, qformer_hidden_size
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None

        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.Qformer = self.Qformer.half()

        # ============================================================
        # Vision → OPT projection (float16)
        # ============================================================
        # self.opt_proj = nn.Linear(
        #     qformer_hidden_size,
        #     self.opt_model.config.hidden_size,
        # ).half()

        # ============================================================
        # Prompt
        # ============================================================
        self.prompt = prompt
        self.max_txt_len = max_txt_len

        prompt_tokens = self.opt_tokenizer(prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        # ============================================================
        # Cross-image context modeling (float16)
        # ============================================================
        D = qformer_hidden_size

        self.context1 = nn.Linear(D, D, bias=False).half()
        self.context2 = nn.Linear(D, D).half()
        self.gate1 = nn.Linear(D, D, bias=False).half()
        self.gate2 = nn.Linear(D, D).half()
        self.context3 = nn.Linear(3 * D, D).half()
        self.dropout = nn.Dropout(0.5)

    # ============================================================
    # 工具函数：统一 dtype & device
    # ============================================================
    def _to_half(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(dtype=torch.float16)
        return x

    # ============================================================
    # Forward
    # ============================================================
    def forward(
        self,
        image_embeds_A: torch.Tensor,
        image_embeds_B: torch.Tensor,
        diff_enhance: torch.Tensor,
        text_input: list[str],
    ):
        device = image_embeds_A.device
        dtype = torch.float16

        # ✅ 统一 dtype
        image_embeds_A = self._to_half(image_embeds_A)
        image_embeds_B = self._to_half(image_embeds_B)
        diff_enhance = self._to_half(diff_enhance)

        B = image_embeds_A.size(0)

        # ========================================================
        # Dual-image context gating
        # ========================================================
        def apply_context(x, d):
            ctx = torch.tanh(self.context1(d) + self.context2(x))
            gate = torch.sigmoid(self.gate1(d) + self.gate2(x))
            return gate * self.dropout(ctx)

        ctx_a = apply_context(image_embeds_A, diff_enhance)
        ctx_b = apply_context(image_embeds_B, diff_enhance)

        fused_a = torch.cat([image_embeds_A, diff_enhance, ctx_a], dim=-1)
        fused_b = torch.cat([image_embeds_B, diff_enhance, ctx_b], dim=-1)

        image_embeds_A = self.context3(fused_a)
        image_embeds_B = self.context3(fused_b)

        image_embeds = torch.cat([image_embeds_A, image_embeds_B], dim=1)
        image_masks = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=device
        )

        # ========================================================
        # Q-Former
        # ========================================================
        query_tokens = self.query_tokens.expand(B, -1, -1).to(dtype=dtype)

        query_outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_masks,
            return_dict=True,
        )

        # image_tokens = self.opt_proj(query_outputs.last_hidden_state)
        image_tokens = query_outputs.last_hidden_state
        image_masks = torch.ones(
            image_tokens.size()[:-1], dtype=torch.long, device=device
        )

        # ========================================================
        # Text
        # ========================================================
        text = [t + "\n" for t in text_input]

        text_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        )

        labels = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )

        if self.prompt:
            labels[:, : self.prompt_length] = -100

        empty_labels = torch.full(
            image_tokens.size()[:-1], -100, dtype=torch.long, device=device
        )
        
        empty_labels = empty_labels.to(device)
        labels = labels.to(device)
        
        labels = torch.cat([empty_labels, labels], dim=1)

        # ========================================================
        # Decoder inputs
        # ========================================================
        text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
        
        input_ids = text_tokens['input_ids']
        attention_mask = text_tokens['attention_mask']
        
        text_embeds = self.opt_model.get_input_embeddings()(input_ids)
        text_embeds = text_embeds.to(dtype=dtype)

        inputs_embeds = torch.cat([image_tokens, text_embeds], dim=1)
        attention_mask = torch.cat([image_masks, attention_mask], dim=1)

        # ========================================================
        # Causal LM loss
        # ========================================================
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
        diff_enhance: torch.Tensor,
        prompt: Optional[str] = None,
        **kwargs,
    ):
        device = image_embeds_A.device
        dtype = torch.float16

        # ✅ 统一 dtype
        image_embeds_A = self._to_half(image_embeds_A)
        image_embeds_B = self._to_half(image_embeds_B)
        diff_enhance = self._to_half(diff_enhance)

        B = image_embeds_A.size(0)

        # ========================================================
        # Context gating
        # ========================================================
        def apply_context(x, d):
            ctx = torch.tanh(self.context1(d) + self.context2(x))
            gate = torch.sigmoid(self.gate1(d) + self.gate2(x))
            return gate * self.dropout(ctx)

        ctx_a = apply_context(image_embeds_A, diff_enhance)
        ctx_b = apply_context(image_embeds_B, diff_enhance)

        fused_a = torch.cat([image_embeds_A, diff_enhance, ctx_a], dim=-1)
        fused_b = torch.cat([image_embeds_B, diff_enhance, ctx_b], dim=-1)

        image_embeds_A = self.context3(fused_a)
        image_embeds_B = self.context3(fused_b)

        image_embeds = torch.cat([image_embeds_A, image_embeds_B], dim=1)
        image_masks = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=device
        )

        # ========================================================
        # Q-Former
        # ========================================================
        query_tokens = self.query_tokens.expand(B, -1, -1).to(dtype=dtype)

        query_outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_masks,
            return_dict=True,
        )

        # image_tokens = self.opt_proj(query_outputs.last_hidden_state)
        image_tokens = query_outputs.last_hidden_state
        image_masks = torch.ones(
            image_tokens.size()[:-1], dtype=torch.long, device=device
        )

        # ========================================================
        # Prompt
        # ========================================================
        prompt = prompt or self.prompt
        prompt = [prompt] * B

        prompt_tokens = self.opt_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        )
        
        prompt_tokens = {k: v.to(device) for k, v in prompt_tokens.items()}

        input_ids = prompt_tokens["input_ids"]
        attention_mask = prompt_tokens["attention_mask"]

        text_embeds = self.opt_model.get_input_embeddings()(input_ids)
        text_embeds = text_embeds.to(dtype=dtype)

        inputs_embeds = torch.cat([image_tokens, text_embeds], dim=1)
        attention_mask = torch.cat([image_masks, attention_mask], dim=1)

        # ========================================================
        # Generate
        # ========================================================
        outputs = self.opt_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            eos_token_id=self.eos_token_id,
            **kwargs,
        )

        return self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)