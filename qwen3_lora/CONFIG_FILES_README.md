# Qwen3 8B LoRA 配置文件说明

## 概述
本文件夹包含Qwen3 8B模型LoRA微调相关的配置文件，这些文件是从MindFormers项目中提取的核心配置。

## 文件列表

### 1. `predict_qwen3_8b_stage1_eval.yaml`
**作用**: Qwen3 8B模型推理评估配置文件

**关键配置项**:
- **模型配置**: 指定使用Qwen3 8B模型
- **推理参数**: batch_size、max_length等推理相关参数
- **评估指标**: 评估任务和指标设置
- **数据路径**: 评估数据集路径配置

**使用场景**:
- 模型推理性能评估
- 微调后模型效果验证
- 批量预测任务配置

### 2. `finetune_qwen3_8b_lora_stage1.yaml`
**作用**: Qwen3 8B模型LoRA微调配置文件

**关键配置项**:
- **LoRA参数**: rank、alpha、dropout等LoRA特定参数
- **训练配置**: learning_rate、batch_size、epochs等训练参数
- **模型保存**: checkpoint保存策略
- **数据加载**: 训练数据预处理和加载配置

**使用场景**:
- Qwen3 8B模型的LoRA微调
- 特定任务的模型适配
- 资源受限环境下的模型优化

## 配置文件关系

```
原始模型 (Qwen3 8B)
    ↓
使用 finetune_qwen3_8b_lora_stage1.yaml 进行LoRA微调
    ↓
微调后的模型
    ↓
使用 predict_qwen3_8b_stage1_eval.yaml 进行评估
    ↓
评估结果和模型性能报告
```

## 使用示例

### LoRA微调命令示例
```bash
# 使用配置文件进行LoRA微调
python run_qwen3_8b_stage1_finetune.sh \
    --config qwen3_lora/finetune_qwen3_8b_lora_stage1.yaml \
    --data_path /path/to/training_data
```

### 推理评估命令示例
```bash
# 使用配置文件进行推理评估
python run_qwen3_8b_stage1_eval_base.sh \
    --config qwen3_lora/predict_qwen3_8b_stage1_eval.yaml \
    --model_path /path/to/finetuned_model
```

## 配置参数详解

### `finetune_qwen3_8b_lora_stage1.yaml` 关键参数
```yaml
model:
  type: Qwen3ForCausalLM
  config:
    hidden_size: 4096
    num_attention_heads: 32
    num_hidden_layers: 32
    
lora_config:
  r: 8                    # LoRA秩
  lora_alpha: 32         # LoRA缩放系数
  lora_dropout: 0.1      # Dropout率
  
train_config:
  learning_rate: 2e-4    # 学习率
  batch_size: 4          # 批次大小
  num_epochs: 3          # 训练轮数
```

### `predict_qwen3_8b_stage1_eval.yaml` 关键参数
```yaml
inference:
  batch_size: 1          # 推理批次大小
  max_length: 2048       # 最大生成长度
  temperature: 0.7       # 温度参数
  top_p: 0.9            # Top-p采样
  
evaluation:
  metrics: ["bleu", "rouge", "accuracy"]  # 评估指标
  dataset: "eval_dataset"                 # 评估数据集
```

## 注意事项

1. **路径配置**: 配置文件中涉及的数据路径需要根据实际环境调整
2. **硬件要求**: LoRA微调对显存要求较低，但仍需确保硬件满足要求
3. **版本兼容**: 配置文件与特定版本的MindFormers和Qwen3模型兼容
4. **参数调优**: 可根据具体任务调整LoRA参数和训练参数

## 相关脚本

本文件夹中的其他脚本文件与这些配置文件配合使用：

- `run_qwen3_8b_stage1_finetune.sh` - 使用LoRA配置进行微调
- `run_qwen3_8b_stage1_eval_base.sh` - 使用推理配置进行评估
- `run_qwen3_8b_stage1_eval_merged.sh` - 合并模型的评估
- `run_qwen3_8b_stage1_merge_lora.sh` - LoRA权重合并

## 更新历史

- **2026-03-17**: 从MindFormers项目提取并添加到本仓库
- **来源**: `/home/leon/Code/mindformers-r1.8.0/configs/qwen3/`

## 参考资料

- [MindFormers官方文档](https://gitee.com/mindspore/mindformers)
- [Qwen3模型文档](https://huggingface.co/Qwen/Qwen3-8B)
- [LoRA微调原理](https://arxiv.org/abs/2106.09685)

---
*文档维护: Bailongma Agent*
*最后更新: 2026-03-17*