# Qwen3-8B 单卡 LoRA 微调配置说明

## 目的

基于官方配置文件 [`configs/qwen3/finetune_qwen3.yaml`](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3.yaml) 复制出单卡测试版本 [`configs/qwen3/finetune_qwen3_8b_lora_1card_debug.yaml`](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_debug.yaml)，目标是：

- 使用 `Qwen3-8B`
- 使用 `LoRA` 微调
- 使用单卡尽快跑通流程
- 只使用 `100` 条测试数据
- 保留官方文件末尾的 profiling 配置段

## 当前使用的配置文件

- 配置文件：[configs/qwen3/finetune_qwen3_8b_lora_1card_debug.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_debug.yaml)
- 代码根目录：`/mnt/hzl/10_qwen3_transfer/mindformers-r1.8.0`
- 模型路径：`/mnt/hzl/10_qwen3_transfer/model/Qwen3-8B`
- 数据路径：`/mnt/hzl/10_qwen3_transfer/data/alpaca-gpt4-data/alpaca_gpt4_data_test_100.json`

## 相对官方 YAML 的主要修改

### 1. 切换为 Qwen3-8B

将：

```yaml
pretrained_model_dir: "/path/to/Qwen3-32B"
```

改为：

```yaml
pretrained_model_dir: "/mnt/hzl/10_qwen3_transfer/model/Qwen3-8B"
```

### 2. 切换为单卡运行

将并行相关配置改为单卡最小化配置：

```yaml
auto_trans_ckpt: False
use_parallel: False
```

```yaml
parallel_config:
  data_parallel: 1
  model_parallel: 1
  pipeline_stage: 1
  micro_batch_num: 1
  use_seq_parallel: False
  gradient_aggregation_group: 1
```

```yaml
parallel:
  parallel_mode: 0
  enable_alltoall: False
  full_batch: False
```

### 3. 切换为 LoRA 微调

在 `model.model_config` 下新增：

```yaml
pet_config:
  pet_type: lora
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.1
  lora_a_init: 'normal'
  lora_b_init: 'zeros'
  target_modules: '.*word_embeddings|.*linear_qkv|.*linear_proj|.*linear_fc1|.*linear_fc2'
  freeze_include: ['*']
  freeze_exclude: ['*lora*']
```

含义：

- 只训练 LoRA 参数
- 基座参数冻结
- 覆盖 embedding、attention、MLP 相关线性层

### 4. 切换为本地 100 条测试数据

将数据集从在线 HF 数据集改为本地 `json`：

```yaml
data_loader:
  type: HFDataLoader
  load_func: 'load_dataset'
  path: "json"
  data_files: "/mnt/hzl/10_qwen3_transfer/data/alpaca-gpt4-data/alpaca_gpt4_data_test_100.json"
```

并保留：

```yaml
handler:
  - type: take
    n: 100
```

### 5. 将序列长度统一为 1024

最终生效的序列长度配置有三处，必须一致：

```yaml
AlpacaInstructDataHandler.seq_length: 1024
PackingHandler.seq_length: 1024
model.model_config.seq_length: 1024
```

这是本次排障中最关键的一点。

### 6. 训练参数保持简单

```yaml
runner_config:
  epochs: 1
  batch_size: 1
  gradient_accumulation_steps: 1
```

```yaml
lr_schedule:
  type: ConstantWarmUpLR
  learning_rate: 5.e-6
```

```yaml
runner_wrapper:
  type: MFTrainOneStepCell
  scale_sense: 1.0
  use_clip_grad: True
  max_grad_norm: 1.0
```

### 7. 保留官方文件末尾 profiling 配置

保留了官方文件最后一段：

```yaml
profile: False
profile_start_step: 1
profile_stop_step: 10
init_start_profile: False
profile_communication: False
profile_memory: True
```

当前只是保留配置，没有开启 profiling。

## 本次报错与修复

### 现象

首次单卡编译时报错：

```text
For 'Mul', input1.shape and input2.shape need to broadcast
input1.shape = [1, 1, 1024]
input2.shape = [1, 4096, 4096]
```

### 原因

数据处理侧已经改成 `1024`，但模型构图侧仍按 `4096` 生成 causal mask。

也就是：

- 数据侧长度：`1024`
- 模型侧 `lower_triangle_mask`：`4096 x 4096`

最终在 [`mindformers/parallel_core/training_graph/transformer/mask_generate.py`](/home/leon/Code/mindformers-r1.8.0/mindformers/parallel_core/training_graph/transformer/mask_generate.py) 中相乘时报 shape mismatch。

### 修复

在 `model.model_config` 中补上：

```yaml
seq_length: 1024
```

修复后，数据侧和模型侧的序列长度保持一致，训练可正常编译并启动。

## 本次运行结果

从日志看，本次单卡 LoRA 微调已经跑通：

- 成功完成编译
- 成功进入训练
- 成功输出 step 日志
- `overflow cond: False`
- `loss_scale: 1.0`
- 成功保存 checkpoint
- 训练正常结束

关键日志摘要：

```text
step 1/25, loss: 15.994116, overflow cond: False, global_norm: [18.362312]
step 2/25, loss: 15.751992, overflow cond: False, global_norm: [16.522688]
step 3/25, loss: 16.604145, overflow cond: False, global_norm: [17.509201]
step 4/25, loss: 15.940238, overflow cond: False, global_norm: [14.831752]
step 5/25, loss: 16.594877, overflow cond: False, global_norm: [17.419353]
step 6/25, loss: 15.939915, overflow cond: False, global_norm: [14.462115]
step 7/25, loss: 14.734194, overflow cond: False, global_norm: [14.444327]
```

checkpoint 保存相关日志：

```text
Saving ckpt......
global_batch_size: 1
epoch_num: 25
step_num: 25
global_step: 25
Training Over!
```

## 结果解读

- 单卡 `Qwen3-8B + LoRA + 100条数据` 已可跑通
- 当前配置适合作为后续精度分析和 profiling 的基础版本
- 当前日志中未见 `overflow`
- step 1 耗时显著偏大，属于首次编译和图初始化，后续 step 稳定在约 `575ms ~ 581ms`

## 后续建议

如果后面继续做实验，建议在这份 YAML 基础上分叉：

1. `smoke` 版
   只用于快速回归跑通

2. `accuracy_debug` 版
   增加更细的 loss / grad / overflow 观察

3. `profile` 版
   将 `profile: False` 改为 `True`，单独做性能分析

当前这份文件建议保留为“单卡最小可跑通基线配置”。
