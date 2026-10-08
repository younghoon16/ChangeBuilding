import os
# ============================================================
# ⚠️ 必须在最开头设置可见设备，只能放在 import torch 之前
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 只暴露 7 号物理 GPU 给程序

import torch
import time
import numpy as np
from model.blip2.CDCformer_test import CDCformer

# ============================================================
# 确认设备
# ============================================================
DEVICE = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES=7 后，cuda:0 就是物理 7 号卡

print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)} (物理 GPU 7)")
print(f"✅ CUDA 版本: {torch.version.cuda}")
print(f"✅ PyTorch 版本: {torch.__version__}")

# ============================================================
# 1. 创建模型
# ============================================================
print("\n" + "=" * 70)
print("CDCformer 模型性能测试")
print("=" * 70)

print("\n📦 加载模型...")
model = CDCformer(
    num_query_token=32,
    qformer_hidden_size=768,
    opt_model="./pretrained/weights/opt-2.7b",
    prompt="Describe the change between two images.",
    max_txt_len=32,
)

model = model.to(DEVICE)
model.eval()

# ============================================================
# 2. 统计参数量
# ============================================================
print("\n" + "=" * 70)
print("📊 参数量统计")
print("=" * 70)

def count_parameters(model):
    """统计参数量"""
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    param_details = {}

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params
        else:
            frozen_params += num_params

        # 按模块分组统计
        module_name = name.split('.')[0]
        if module_name not in param_details:
            param_details[module_name] = 0
        param_details[module_name] += num_params

    return total_params, trainable_params, frozen_params, param_details

def format_params(num_params):
    """格式化参数量显示"""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return f"{num_params}"

total_params, trainable_params, frozen_params, param_details = count_parameters(model)

print(f"\n{'模块':<25} {'参数量':>15} {'占比':>10}")
print("-" * 55)
for module_name, count in sorted(param_details.items(), key=lambda x: -x[1]):
    percentage = count / total_params * 100
    print(f"{module_name:<25} {format_params(count):>15} {percentage:>9.1f}%")
print("-" * 55)
print(f"{'总计':<25} {format_params(total_params):>15} {'100.0%':>10}")
print(f"\n✅ 可训练参数: {format_params(trainable_params)} ({trainable_params/total_params*100:.1f}%)")
print(f"❄️  冻结参数:   {format_params(frozen_params)} ({frozen_params/total_params*100:.1f}%)")

# ============================================================
# 3. 构造测试输入
# ============================================================
B, N, D = 2, 256, 768

image_embeds_A = torch.randn(B, N, D, device=DEVICE)
image_embeds_B = torch.randn(B, N, D, device=DEVICE)
diff_enhance = torch.randn(B, N, D, device=DEVICE)

# LayerNorm 稳定输入
image_embeds_A = torch.nn.functional.layer_norm(image_embeds_A, [D])
image_embeds_B = torch.nn.functional.layer_norm(image_embeds_B, [D])
diff_enhance = torch.nn.functional.layer_norm(diff_enhance, [D])

text_input = [
    "The building was demolished.",
    "A new road was constructed.",
]

# ============================================================
# 4. 估算 FLOPs
# ============================================================
print("\n" + "=" * 70)
print("🔢 FLOPs 估算")
print("=" * 70)

# Q-Former 参数量
qformer_params = sum(p.numel() for n, p in model.named_parameters() if 'Qformer' in n)
opt_params = sum(p.numel() for n, p in model.named_parameters() if 'opt_model' in n)
proj_params = sum(p.numel() for n, p in model.named_parameters() if 'opt_proj' in n)
ctx_params = sum(p.numel() for n, p in model.named_parameters()
                 if any(x in n for x in ['context1', 'context2', 'gate1', 'gate2', 'context3']))

print(f"\nQ-Former 参数: {format_params(qformer_params)}")
print(f"OPT 模型参数:  {format_params(opt_params)}")
print(f"投影层参数:    {format_params(proj_params)}")
print(f"Context 模块:  {format_params(ctx_params)}")

# 手动估算 FLOPs
num_qformer_layers = len([n for n, p in model.named_parameters()
                          if 'Qformer.bert.encoder.layer' in n and 'self' in n])

num_queries = 32
num_image_tokens = N * 2  # A + B
ffn_dim = 3072

# Cross-attention: queries × image_tokens × dim × 4 (Q,K,V,O)
flops_cross = num_queries * num_image_tokens * D * 4
# Self-attention: queries × queries × dim × 4
flops_self = num_queries * num_queries * D * 4
# FFN: queries × dim × ffn_dim × 2
flops_ffn = num_queries * D * ffn_dim * 2

total_qformer_flops = num_qformer_layers * (flops_cross + flops_self + flops_ffn)

# OPT 生成
opt_hidden = model.opt_model.config.hidden_size
opt_layers = model.opt_model.config.num_hidden_layers
opt_ffn = opt_hidden * 4
num_gen_tokens = 20

per_token_opt = opt_layers * (opt_hidden * opt_hidden * 4 + opt_hidden * opt_ffn * 2)
total_opt_flops = num_gen_tokens * per_token_opt

total_flops = total_qformer_flops + total_opt_flops

print(f"\nQ-Former 层数 (估算): {num_qformer_layers}")
print(f"Q-Former FLOPs (估算): {total_qformer_flops/1e9:.2f} GFLOPs")
print(f"OPT 生成 FLOPs (估算):  {total_opt_flops/1e9:.2f} GFLOPs")
print(f"总 FLOPs (估算):        {total_flops/1e9:.2f} GFLOPs")

# ============================================================
# 5. 推理延迟测试 (generate)
# ============================================================
print("\n" + "=" * 70)
print("⏱️  推理延迟测试 (generate)")
print("=" * 70)

model.eval()

# Warmup
with torch.no_grad():
    for _ in range(20):
        _ = model.generate(
            image_embeds_A=image_embeds_A,
            image_embeds_B=image_embeds_B,
            diff_enhance=diff_enhance,
            max_new_tokens=20,
        )

torch.cuda.synchronize()

# 正式测试
latencies = []
with torch.no_grad():
    for _ in range(100):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        captions = model.generate(
            image_embeds_A=image_embeds_A,
            image_embeds_B=image_embeds_B,
            diff_enhance=diff_enhance,
            max_new_tokens=20,
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # ms

latencies = np.array(latencies)

print(f"\n{'指标':<20} {'数值':>15}")
print("-" * 40)
print(f"{'Mean Latency':<20} {latencies.mean():>12.2f} ms")
print(f"{'Std Dev':<20} {latencies.std():>12.2f} ms")
print(f"{'Min Latency':<20} {latencies.min():>12.2f} ms")
print(f"{'Max Latency':<20} {latencies.max():>12.2f} ms")
print(f"{'P50 Latency':<20} {np.percentile(latencies, 50):>12.2f} ms")
print(f"{'P95 Latency':<20} {np.percentile(latencies, 95):>12.2f} ms")
print(f"{'FPS (单样本)':<20} {1000.0 / latencies.mean():>12.2f}")
print(f"{'FPS (batch=2)':<20} {1000.0 / latencies.mean() * B:>12.2f}")

latency_results = {
    'mean_ms': latencies.mean(),
    'std_ms': latencies.std(),
    'min_ms': latencies.min(),
    'max_ms': latencies.max(),
    'p50_ms': np.percentile(latencies, 50),
    'p95_ms': np.percentile(latencies, 95),
    'fps': 1000.0 / latencies.mean(),
}

# ============================================================
# 6. 训练 Forward + Backward 时间
# ============================================================
print("\n" + "=" * 70)
print("🔥 训练 Forward + Backward 时间")
print("=" * 70)

model.train()

# Warmup
for _ in range(10):
    model.zero_grad()
    out = model(
        image_embeds_A=image_embeds_A,
        image_embeds_B=image_embeds_B,
        diff_enhance=diff_enhance,
        text_input=text_input,
    )
    loss = out["loss"]
    loss.backward()

torch.cuda.synchronize()

# 正式测试
forward_times = []
backward_times = []

for _ in range(50):
    model.zero_grad()

    # Forward
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = model(
        image_embeds_A=image_embeds_A,
        image_embeds_B=image_embeds_B,
        diff_enhance=diff_enhance,
        text_input=text_input,
    )
    loss = out["loss"]
    torch.cuda.synchronize()
    forward_times.append((time.perf_counter() - start) * 1000)

    # Backward
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    backward_times.append((time.perf_counter() - start) * 1000)

forward_times = np.array(forward_times)
backward_times = np.array(backward_times)

print(f"\n{'指标':<25} {'数值':>15}")
print("-" * 45)
print(f"{'Forward Mean':<25} {forward_times.mean():>12.2f} ms")
print(f"{'Forward Std':<25} {forward_times.std():>12.2f} ms")
print(f"{'Backward Mean':<25} {backward_times.mean():>12.2f} ms")
print(f"{'Backward Std':<25} {backward_times.std():>12.2f} ms")
print(f"{'Total Mean':<25} {(forward_times + backward_times).mean():>12.2f} ms")

train_times = {
    'forward_mean_ms': forward_times.mean(),
    'forward_std_ms': forward_times.std(),
    'backward_mean_ms': backward_times.mean(),
    'backward_std_ms': backward_times.std(),
    'total_mean_ms': (forward_times + backward_times).mean(),
}

# ============================================================
# 7. 不同 Batch Size 下的延迟测试
# ============================================================
print("\n" + "=" * 70)
print("📈 不同 Batch Size 下的推理延迟")
print("=" * 70)

batch_sizes = [1, 2, 4, 8]
D_test = 768
N_test = 256

print(f"\n{'Batch Size':<15} {'Mean Latency (ms)':>20} {'FPS':>15}")
print("-" * 55)

for bs in batch_sizes:
    img_a = torch.randn(bs, N_test, D_test, device=DEVICE)
    img_b = torch.randn(bs, N_test, D_test, device=DEVICE)
    diff = torch.randn(bs, N_test, D_test, device=DEVICE)

    img_a = torch.nn.functional.layer_norm(img_a, [D_test])
    img_b = torch.nn.functional.layer_norm(img_b, [D_test])
    diff = torch.nn.functional.layer_norm(diff, [D_test])

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model.generate(
                image_embeds_A=img_a,
                image_embeds_B=img_b,
                diff_enhance=diff,
                max_new_tokens=20,
            )

    torch.cuda.synchronize()

    latencies_bs = []
    with torch.no_grad():
        for _ in range(50):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model.generate(
                image_embeds_A=img_a,
                image_embeds_B=img_b,
                diff_enhance=diff,
                max_new_tokens=20,
            )
            torch.cuda.synchronize()
            latencies_bs.append((time.perf_counter() - start) * 1000)

    avg_lat = np.mean(latencies_bs)
    fps = 1000.0 / avg_lat * bs
    print(f"{bs:<15} {avg_lat:>18.2f} {fps:>15.2f}")

    del img_a, img_b, diff
    torch.cuda.empty_cache()

# ============================================================
# 8. 显存占用
# ============================================================
print("\n" + "=" * 70)
print("💾 显存占用")
print("=" * 70)

# 推理显存
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model.eval()
with torch.no_grad():
    _ = model.generate(
        image_embeds_A=image_embeds_A,
        image_embeds_B=image_embeds_B,
        diff_enhance=diff_enhance,
        max_new_tokens=20,
    )

infer_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
print(f"\n推理峰值显存: {infer_memory:.2f} GB")

# 训练显存
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model.train()
out = model(
    image_embeds_A=image_embeds_A,
    image_embeds_B=image_embeds_B,
    diff_enhance=diff_enhance,
    text_input=text_input,
)
loss = out["loss"]
loss.backward()

train_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
print(f"训练峰值显存: {train_memory:.2f} GB")

# ============================================================
# 9. 生成测试
# ============================================================
print("\n" + "=" * 70)
print("📝 生成测试")
print("=" * 70)

model.eval()
with torch.no_grad():
    captions = model.generate(
        image_embeds_A=image_embeds_A,
        image_embeds_B=image_embeds_B,
        diff_enhance=diff_enhance,
        max_new_tokens=20,
    )

for i, c in enumerate(captions):
    print(f"  样本 {i}: {c}")

# ============================================================
# 10. 总结
# ============================================================
print("\n" + "=" * 70)
print("📋 总结")
print("=" * 70)
print(f"""
┌─────────────────────┬─────────────────────────────────┐
│ 总参数量           │ {format_params(total_params):>10}                    │
│ 可训练参数         │ {format_params(trainable_params):>10} ({trainable_params/total_params*100:>5.1f}%)               │
│ 冻结参数           │ {format_params(frozen_params):>10} ({frozen_params/total_params*100:>5.1f}%)               │
│ 估算 FLOPs        │ {total_flops/1e9:>9.2f} G                  │
│ 推理延迟 (mean)    │ {latency_results['mean_ms']:>9.2f} ms                 │
│ 推理 FPS (batch=1) │ {latency_results['fps']:>9.2f}                     │
│ 训练 F+B (mean)    │ {train_times['total_mean_ms']:>9.2f} ms                 │
│ 推理峰值显存       │ {infer_memory:>9.2f} GB                  │
│ 训练峰值显存       │ {train_memory:>9.2f} GB                  │
└─────────────────────┴─────────────────────────────────┘
""")

print("✅ 测试完成！")