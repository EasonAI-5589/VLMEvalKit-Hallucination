# VLMEvalKit 集成 Attention-based Decoding 方法指南

## 1. 概述

本文档说明如何在 VLMEvalKit 框架中使用集成的 Attention-based Decoding 方法（PAI、OPERA、AllPath）来缓解 LLaVA 模型的幻觉问题。

### 1.1 三种方法对比

| 方法 | 论文 | 核心机制 | 适用场景 |
|------|------|----------|----------|
| **PAI** | ECCV 2024 | 注意力增强 + 对比解码 (CFG) | 通用场景，易于使用 |
| **OPERA** | CVPR 2024 | 过度信任惩罚 + 回溯机制 | 高精度需求 |
| **AllPath** | NeurIPS 2025 | 头级别干预（增强/抑制特定 heads） | 有预训练头部配置 |

### 1.2 支持的解码策略

| 策略 | 参数 | 说明 |
|------|------|------|
| **Greedy** | `do_sample=False, num_beams=1` | 默认，速度最快 |
| **Beam Search** | `do_sample=False, num_beams=N` | 更好的生成质量 |
| **Nucleus Sampling** | `do_sample=True, top_p=P` | 更多样的输出 |

---

## 2. 快速开始

### 2.1 注意力方法

```bash
# PAI 方法
python run.py --data POPE --model llava_v1.5_7b --method pai

# OPERA 方法
python run.py --data MME --model llava_v1.5_7b --method opera

# AllPath 方法
python run.py --data POPE --model llava_v1.5_13b --method allpath
```

### 2.2 解码策略

```bash
# Greedy Decoding（默认）
python run.py --data POPE --model llava_v1.5_7b

# Beam Search
python run.py --data POPE --model llava_v1.5_7b --decoding beam --num-beams 5

# Nucleus Sampling
python run.py --data POPE --model llava_v1.5_7b --decoding nucleus --top-p 0.9 --temperature 0.7
```

### 2.3 组合使用

```bash
# PAI + Beam Search
python run.py --data POPE --model llava_v1.5_7b --method pai --decoding beam --num-beams 5

# AllPath + Nucleus Sampling
python run.py --data POPE --model llava_v1.5_7b --method allpath --decoding nucleus
```

### 2.4 输出目录

| 参数组合 | 输出目录 |
|---------|---------|
| `--model llava_v1.5_7b` | `outputs/llava_v1.5_7b/` |
| `--model llava_v1.5_7b --method pai` | `outputs/llava_v1.5_7b_pai/` |
| `--model llava_v1.5_7b --decoding beam` | `outputs/llava_v1.5_7b_beam/` |
| `--model llava_v1.5_7b --method pai --decoding beam` | `outputs/llava_v1.5_7b_pai_beam/` |

---

## 3. 验证脚本

### 3.1 完整验证脚本

```bash
#!/bin/bash
# run_eval.sh - 完整的评估脚本

# 设置环境
export CUDA_VISIBLE_DEVICES=0

# ============================================
# 1. Baseline（原始 LLaVA）
# ============================================
# Greedy Decoding
python run.py --data POPE --model llava_v1.5_7b

# Beam Search
python run.py --data POPE --model llava_v1.5_7b --decoding beam --num-beams 5

# Nucleus Sampling
python run.py --data POPE --model llava_v1.5_7b --decoding nucleus --top-p 0.9 --temperature 0.7

# ============================================
# 2. PAI 方法
# ============================================
python run.py --data POPE --model llava_v1.5_7b --method pai

# PAI + Beam Search
python run.py --data POPE --model llava_v1.5_7b --method pai --decoding beam --num-beams 5

# ============================================
# 3. OPERA 方法
# ============================================
python run.py --data POPE --model llava_v1.5_7b --method opera

# ============================================
# 4. AllPath 方法
# ============================================
python run.py --data POPE --model llava_v1.5_7b --method allpath
```

### 3.2 单独验证命令

```bash
# 验证 PAI
python run.py --data POPE --model llava_v1.5_7b --method pai

# 验证 OPERA
python run.py --data POPE --model llava_v1.5_7b --method opera

# 验证 AllPath
python run.py --data POPE --model llava_v1.5_7b --method allpath

# 验证 Beam Search
python run.py --data POPE --model llava_v1.5_7b --decoding beam --num-beams 5

# 验证 Nucleus Sampling
python run.py --data POPE --model llava_v1.5_7b --decoding nucleus --top-p 0.9 --temperature 0.7
```

### 3.3 多 GPU 验证

```bash
# 使用多 GPU
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run.py \
    --data POPE --model llava_v1.5_7b --method pai
```

### 3.4 批量对比脚本

```bash
#!/bin/bash
# compare_methods.sh - 对比不同方法

DATA="POPE"
MODEL="llava_v1.5_7b"

echo "=== Running Baseline ==="
python run.py --data $DATA --model $MODEL

echo "=== Running PAI ==="
python run.py --data $DATA --model $MODEL --method pai

echo "=== Running OPERA ==="
python run.py --data $DATA --model $MODEL --method opera

echo "=== Running AllPath ==="
python run.py --data $DATA --model $MODEL --method allpath

echo "=== Running Beam Search ==="
python run.py --data $DATA --model $MODEL --decoding beam --num-beams 5

echo "=== Running Nucleus Sampling ==="
python run.py --data $DATA --model $MODEL --decoding nucleus

echo "=== All evaluations completed ==="
```

---

## 4. 命令行参数

### 4.1 注意力方法

```bash
--method {baseline,pai,opera,allpath}
    # baseline: 原始 LLaVA（默认）
    # pai: PAI 方法
    # opera: OPERA 方法
    # allpath: AllPath 方法
```

### 4.2 解码策略

```bash
--decoding {greedy,beam,nucleus}
    # greedy: 贪婪解码（默认）
    # beam: Beam Search
    # nucleus: Nucleus Sampling

--num-beams N       # Beam 数量（默认 5）
--top-p P           # Top-p 值（默认 0.9）
--temperature T     # 温度（默认 0.7）
```

---

## 5. 实现架构

### 5.1 文件结构

```
vlmeval/
├── vlm/
│   └── llava/
│       ├── llava.py                  # 原始 LLaVA
│       ├── llava_hallucination.py    # 支持幻觉缓解的 LLaVA
│       └── __init__.py
│
└── run.py                            # --method 和 --decoding 参数
```

### 5.2 核心类

```python
class LLaVA_Hallucination(BaseModel):
    def __init__(
        self,
        model_path: str = "liuhaotian/llava-v1.5-7b",
        decoding_method: str = "baseline",  # baseline, pai, opera, allpath
        # PAI 参数
        pai_alpha: float = 0.2,
        pai_gamma: float = 1.1,
        # OPERA 参数
        opera_scale_factor: float = 50.0,
        opera_threshold: int = 15,
        # AllPath 参数
        allpath_in_scale: float = 2.0,
        allpath_de_scale: float = 0.0,
        # 解码参数
        do_sample: bool = False,
        temperature: float = 0,
        top_p: float = None,
        num_beams: int = 1,
        **kwargs
    )
```

### 5.3 run.py 关键函数

```python
def get_decoding_kwargs(decoding='greedy', num_beams=5, top_p=0.9, temperature=0.7):
    """根据解码策略返回生成参数"""

def build_model_with_method(model_name, method=None, decoding='greedy', ...):
    """构建带有幻觉缓解方法的模型"""
```

---

## 6. 方法详细说明

### 6.1 PAI

- **注意力增强**：增强对图像 token 的注意力
- **对比解码**：生成时减去纯文本条件的 logits
- **参数**：`pai_alpha=0.2`, `pai_gamma=1.1`

### 6.2 OPERA

- **过度信任惩罚**：基于注意力模式计算惩罚
- **回溯机制**：检测汇总 token 并回滚
- **参数**：`opera_scale_factor=50.0`, `opera_threshold=15`

### 6.3 AllPath

- **头部干预**：增强好头、抑制幻觉头
- **参数**：`allpath_in_scale=2.0`, `allpath_de_scale=0.0`

---

## 7. Python API

```python
from vlmeval.vlm import LLaVA_Hallucination

# PAI
model = LLaVA_Hallucination(
    model_path="liuhaotian/llava-v1.5-7b",
    decoding_method="pai",
    pai_alpha=0.3,
)

# OPERA
model = LLaVA_Hallucination(
    model_path="liuhaotian/llava-v1.5-7b",
    decoding_method="opera",
    num_beams=5,
)

# AllPath
model = LLaVA_Hallucination(
    model_path="liuhaotian/llava-v1.5-7b",
    decoding_method="allpath",
    allpath_in_scale=2.5,
)

# 生成
message = [
    {"type": "image", "value": "/path/to/image.jpg"},
    {"type": "text", "value": "Is there a dog in this image?"}
]
response = model.generate(message)
```

---

## 8. 性能

| 方法 | 相对速度 | 额外内存 |
|------|----------|----------|
| Baseline | 1.0x | - |
| PAI | ~1.5x 慢 | +10% |
| OPERA | ~3-5x 慢 | +50% |
| AllPath | ~1.2x 慢 | +5% |

---

## 9. 已知限制与待解决问题

### 9.1 OPERA 方法限制

**问题**：当前 OPERA 实现是简化版本，缺少完整的回溯机制。

**原因**：完整的 OPERA 需要修改 HuggingFace transformers 库的 `beam_search` 函数，以支持：
- 基于注意力模式的过度信任惩罚计算
- 检测到汇总 token 时的回溯（rollback）机制

**当前行为**：使用标准 beam search，会输出警告信息。

**解决方案**：如需完整 OPERA 功能，请安装 OPERA 官方仓库的修改版 transformers：
```bash
# 从 OPERA 官方仓库安装修改版 transformers
git clone https://github.com/shikiw/OPERA.git
cd OPERA
pip install -e transformers-4.36.2
```

### 9.2 AllPath 头部配置

**问题**：AllPath 方法需要预训练的头部配置（`hallu_heads` 和 `good_heads`），默认为空。

**原因**：AllPath 通过增强"好头"、抑制"幻觉头"来工作，这些头部需要通过分析预先识别。

**当前行为**：如果不提供头部配置，AllPath 不会产生任何效果。

**解决方案**：使用 Python API 手动指定头部配置：
```python
from vlmeval.vlm import LLaVA_Hallucination

# 示例头部配置（需要根据实际分析结果设置）
hallu_heads = {
    10: [5, 12, 18],  # 第10层的幻觉头
    15: [3, 7, 22],   # 第15层的幻觉头
}
good_heads = {
    8: [1, 4, 9],     # 第8层的好头
    12: [2, 6, 11],   # 第12层的好头
}

model = LLaVA_Hallucination(
    model_path="liuhaotian/llava-v1.5-7b",
    decoding_method="allpath",
    allpath_hallu_heads=hallu_heads,
    allpath_good_heads=good_heads,
    allpath_in_scale=2.0,   # 好头增强倍数
    allpath_de_scale=0.0,   # 幻觉头抑制倍数
)
```

### 9.3 问题状态汇总

| 问题 | 状态 | 影响 |
|------|------|------|
| OPERA 简化版本 | ⚠️ [TODO] 待完善 | 效果可能不如论文报告 |
| AllPath 无默认头部配置 | ⚠️ [TODO] 待配置 | 需手动提供头部配置 |
| PAI 方法 | ✅ 完整实现 | 可直接使用 |

---

## 10. 参考资料

- **PAI**: [Paying More Attention to Image](https://arxiv.org/abs/2407.21771) (ECCV 2024)
- **OPERA**: [OPERA: Alleviating Hallucination](https://arxiv.org/abs/2311.17911) (CVPR 2024)
- **AllPath**: [Exploring and Mitigating Hallucinations](https://arxiv.org/abs/) (NeurIPS 2025)
