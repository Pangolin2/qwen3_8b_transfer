# Qwen3-8B 单卡 LoRA 实验记录

## 目标

本说明文档用于持续记录 `Qwen3-8B` 在 `MindFormers r1.8.0` 上的单卡 `LoRA` 微调实验，包括：

- 配置文件用途
- 当前实验阶段
- YAML 修改点
- 启动命令
- 运行结果
- 后续精度分析计划

后续实验继续在这份文档上更新，不另起文档。

## 环境与路径

- 代码目录：`/mnt/hzl/10_qwen3_transfer/mindformers-r1.8.0`
- 本地工作目录：`/home/leon/Code/mindformers-r1.8.0`
- 模型目录：`/mnt/hzl/10_qwen3_transfer/model/Qwen3-8B`
- 数据集：`/mnt/hzl/10_qwen3_transfer/data/alpaca-gpt4-data/alpaca_gpt4_data_test_100.json`

## 实验阶段总览

### 阶段 1：单卡 LoRA 跑通

目标：

- 基于官方 `Qwen3` 微调 YAML 复制新文件
- 切换到 `Qwen3-8B`
- 切换到单卡
- 使用本地 `100` 条数据
- 使用 `LoRA`
- 跑通训练、保存 checkpoint

结果：

- 已完成
- 训练成功启动
- loss 正常输出
- `overflow=False`
- checkpoint 保存成功

### 阶段 2：精度分析

目标：

- 不直接上 `Dump`
- 先做基础稳定性分析
- 重点观察：
  - `loss`
  - `global_norm`
  - `local_norm`
  - `overflow`
  - `loss_scale`
- 先做第一组实验：`local_norm + nan/inf check`

## 配置文件列表

### 1. 阶段 1 基线配置

- 文件：[finetune_qwen3_8b_lora_1card_debug.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_debug.yaml)

用途：

- 单卡 LoRA 最小可跑通版本
- 保留官方文件末尾 profiling 配置
- 不做额外精度分析增强

### 2. 阶段 2 第一组实验配置

- 文件：[finetune_qwen3_8b_lora_1card_acc_group1.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_acc_group1.yaml)

用途：

- 在阶段 1 基线配置上增加基础精度分析能力
- 开启 `local_norm`
- 开启 `nan/inf` 检查
- 增加 `TrainingStateMonitor`

## 阶段 1 配置修改摘要

相对官方 [`finetune_qwen3.yaml`](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3.yaml) 的主要修改：

1. 切换模型

```yaml
pretrained_model_dir: "/mnt/hzl/10_qwen3_transfer/model/Qwen3-8B"
```

2. 切换为单卡

```yaml
use_parallel: False
auto_trans_ckpt: False
```

```yaml
parallel_config:
  data_parallel: 1
  model_parallel: 1
  pipeline_stage: 1
  micro_batch_num: 1
```

3. 切换为本地 100 条 json 数据

```yaml
data_loader:
  path: "json"
  data_files: "/mnt/hzl/10_qwen3_transfer/data/alpaca-gpt4-data/alpaca_gpt4_data_test_100.json"
```

```yaml
handler:
  - type: take
    n: 100
```

4. 启用 LoRA

```yaml
pet_config:
  pet_type: lora
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.1
  target_modules: '.*word_embeddings|.*linear_qkv|.*linear_proj|.*linear_fc1|.*linear_fc2'
  freeze_include: ['*']
  freeze_exclude: ['*lora*']
```

5. 统一序列长度为 1024

```yaml
AlpacaInstructDataHandler.seq_length: 1024
PackingHandler.seq_length: 1024
model.model_config.seq_length: 1024
```

## 阶段 1 关键问题与修复

### 问题

首次训练编译报错：

```text
For 'Mul', input1.shape = [1, 1, 1024]
input2.shape = [1, 4096, 4096]
```

### 原因

数据侧长度已经改成 `1024`，但模型侧 `seq_length` 仍然沿用原始 `4096`。

### 修复

在 `model.model_config` 中补充：

```yaml
seq_length: 1024
```

## 阶段 1 运行结果

摘要：

- step 1 首次编译耗时高
- step 2 之后基本稳定
- `overflow=False`
- `loss_scale=1.0`
- checkpoint 保存成功

关键日志摘录：

```text
step 1/25, loss: 15.994116, per_step_time: 153259ms, overflow cond: False, loss_scale: 1.0, global_norm: [18.362312]
step 2/25, loss: 15.751992, per_step_time: 577ms, overflow cond: False, loss_scale: 1.0, global_norm: [16.522688]
step 3/25, loss: 16.604145, per_step_time: 581ms, overflow cond: False, loss_scale: 1.0, global_norm: [17.509201]
step 4/25, loss: 15.940238, per_step_time: 578ms, overflow cond: False, loss_scale: 1.0, global_norm: [14.831752]
step 5/25, loss: 16.594877, per_step_time: 576ms, overflow cond: False, loss_scale: 1.0, global_norm: [17.419353]
step 6/25, loss: 15.939915, per_step_time: 575ms, overflow cond: False, loss_scale: 1.0, global_norm: [14.462115]
step 7/25, loss: 14.734194, per_step_time: 577ms, overflow cond: False, loss_scale: 1.0, global_norm: [14.444327]
```

checkpoint：

```text
Saving ckpt......
global_batch_size: 1
step_num: 25
global_step: 25
Training Over!
```

结论：

- 单卡 Qwen3-8B LoRA 路径已跑通
- 当前基线配置可作为后续精度分析基础版本

## 阶段 2 实验思路

参考官方精度调优文档：

- 官方文档：https://www.mindspore.cn/mindformers/docs/zh-CN/r1.8.0/advanced_development/precision_optimization.html

当前策略：

1. 先不启用 `Dump`
2. 先做基础问题排查
3. 优先观察：
   - `step1/step2 loss`
   - `global_norm`
   - `local_norm`
   - `overflow`
   - `loss_scale`
4. 先做第一组实验：
   - `local_norm=True`
   - `check_for_nan_in_loss_and_grad=True`

## 阶段 2 第一组实验说明

### 配置目标

第一组实验用于做最基础的精度稳定性检查。

相对阶段 1 基线配置，新增：

1. 顶层开启 `nan/inf` 检查

```yaml
check_for_nan_in_loss_and_grad: True
```

2. 打开 `local_norm`

```yaml
runner_wrapper:
  local_norm: True
```

3. 增加 `TrainingStateMonitor`

```yaml
callbacks:
  - type: MFLossMonitor
  - type: TrainingStateMonitor
  - type: CheckpointMonitor
```

说明：

- `MFLossMonitor` 负责打印 `loss / overflow / loss_scale / global_norm`
- `TrainingStateMonitor` 负责触发更完整的监控逻辑，包括 `nan/inf` 边界检查
- `local_norm=True` 会使训练 wrapper 返回 `local_norm`，并由 callback 打印

### 本组实验要看什么

重点看：

- `step1 loss`
- `step2 loss`
- `global_norm`
- `local_norm`
- `overflow cond`
- `loss_scale`

判断标准：

- 不应出现 `nan/inf`
- 不应出现 `overflow=True`
- `global_norm` 不应异常尖刺
- `local_norm` 不应出现明显异常值

## 运行命令

### 阶段 1 基线跑通

```bash
cd /mnt/hzl/10_qwen3_transfer/mindformers-r1.8.0
python run_mindformer.py \
  --config configs/qwen3/finetune_qwen3_8b_lora_1card_debug.yaml \
  --run_mode finetune
```

### 阶段 2 第一组实验

```bash
cd /mnt/hzl/10_qwen3_transfer/mindformers-r1.8.0
python run_mindformer.py \
  --config configs/qwen3/finetune_qwen3_8b_lora_1card_acc_group1.yaml \
  --run_mode finetune
```

## 阶段 2 记录模板

执行第一组实验后，建议记录以下内容，后续继续补到本文件中：

```text
实验名称：
配置文件：
是否成功启动：
step1 loss：
step2 loss：
step10 loss：
最后一步 loss：
overflow：
loss_scale：
global_norm 范围：
local_norm 是否正常：
是否保存 checkpoint：
结论：
```

## 下一步计划

第一组实验完成后，继续做第二组实验：

- 动态 loss scale 对照实验

计划新增配置文件：

- `finetune_qwen3_8b_lora_1card_acc_group2.yaml`

本文件后续继续追加：

- 第一组实验结果
- 第二组实验配置与结果
- 是否需要进入 `Dump` 分析
