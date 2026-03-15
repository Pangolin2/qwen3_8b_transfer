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

### 3. 阶段 2 第二组实验配置

- 文件：[finetune_qwen3_8b_lora_1card_acc_group2.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_acc_group2.yaml)

用途：

- 在第一组基础上做动态 loss scale 对照实验
- 保留 `local_norm`
- 保留 `nan/inf` 检查
- 将 `scale_sense` 从固定 `1.0` 改为 `AdaptiveLossScaleUpdateCell`

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

## 阶段 2 第一组实验结果

实验名称：

- `acc_group1`

配置文件：

- [finetune_qwen3_8b_lora_1card_acc_group1.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_acc_group1.yaml)

实验结果表：

| step | loss | global_norm | overflow | loss_scale |
| ---- | ---: | ----------: | -------- | ---------: |
| 1 | 15.994116 | 18.363276 | False | 1.0 |
| 2 | 15.762886 | 16.694190 | False | 1.0 |
| 3 | 16.583115 | 17.367466 | False | 1.0 |
| 4 | 15.922192 | 14.777558 | False | 1.0 |
| 5 | 16.606695 | 17.294086 | False | 1.0 |
| 6 | 15.940705 | 14.446381 | False | 1.0 |
| 7 | 14.742739 | 14.557197 | False | 1.0 |
| 8 | 16.535093 | 16.793005 | False | 1.0 |
| 9 | 18.153065 | 16.866010 | False | 1.0 |
| 10 | 15.514065 | 14.986671 | False | 1.0 |
| 11 | 14.842097 | 13.528371 | False | 1.0 |
| 12 | 15.031343 | 13.323236 | False | 1.0 |
| 13 | 15.078642 | 15.839686 | False | 1.0 |
| 14 | 13.316998 | 9.804013 | False | 1.0 |
| 15 | 16.386261 | 18.083832 | False | 1.0 |
| 16 | 20.476391 | 24.483217 | False | 1.0 |
| 17 | 15.943971 | 16.552122 | False | 1.0 |
| 18 | 15.183918 | 15.458350 | False | 1.0 |
| 19 | 16.258022 | 18.503250 | False | 1.0 |
| 20 | 14.880547 | 17.422707 | False | 1.0 |
| 21 | 14.626083 | 14.995673 | False | 1.0 |
| 22 | 15.866027 | 16.819584 | False | 1.0 |
| 23 | 15.227503 | 16.916020 | False | 1.0 |
| 24 | 16.030104 | 14.749761 | False | 1.0 |
| 25 | 17.689260 | 25.118725 | False | 1.0 |

初步结论：

- 第一组实验未出现 `nan/inf`
- 全程 `overflow=False`
- `loss_scale` 固定为 `1.0`，无异常
- `global_norm` 大体处于可接受波动范围
- step 16 和 step 25 的 `global_norm` 明显偏高，但未引发 overflow，属于后续对照实验需要重点关注的点
- 当前可以进入第二组实验，观察动态 loss scale 是否会带来更平稳的数值表现

建议重点对比项：

- step 1 到 step 5 的 `loss/global_norm`
- step 16 和 step 25 附近的 `global_norm`
- 动态 loss scale 是否发生变化
- 是否出现新的 overflow

## 阶段 2 第二组实验说明

第二组实验用于做动态 loss scale 对照。

相对第一组实验，唯一核心改动是：

```yaml
runner_wrapper:
  type: MFTrainOneStepCell
  scale_sense:
    type: AdaptiveLossScaleUpdateCell
    loss_scale_value: 65536
    scale_factor: 2
    scale_window: 1000
  use_clip_grad: True
  max_grad_norm: 1.0
  local_norm: True
```

实验目标：

- 观察动态 loss scale 是否会被调整
- 对比固定 `loss_scale=1.0` 与动态 loss scale 的稳定性差异
- 观察 `global_norm` 是否更平稳
- 检查是否出现新的 `overflow`

预期现象：

- 如果当前固定 `1.0` 已足够稳定，第二组可能不会显著更好
- 但第二组能帮助确认这个模型在当前配置下是否对 loss scale 敏感
- 如果动态 loss scale 频繁调整，说明当前数值空间更接近边界

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

### 阶段 2 第二组实验

```bash
cd /mnt/hzl/10_qwen3_transfer/mindformers-r1.8.0
python run_mindformer.py \
  --config configs/qwen3/finetune_qwen3_8b_lora_1card_acc_group2.yaml \
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

当前下一步：

- 执行第二组实验 `acc_group2`
- 收集 `step1/step2/step10/最后一步` 的：
  - `loss`
  - `global_norm`
  - `overflow`
  - `loss_scale`

本文件后续继续追加：

- 第二组实验结果
- 第一组与第二组对比结论
- 是否需要进入长稳实验
- 是否需要进入 `Dump` 分析

## 阶段 2 第二组实验结果

实验名称：

- `acc_group2`

配置文件：

- [finetune_qwen3_8b_lora_1card_acc_group2.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_acc_group2.yaml)

已知关键结果：

- 最后一步日志：

```text
step 25/25, loss: 17.659529, overflow cond: False, loss_scale: 65536.0, global_norm: [24.985922]
```

- checkpoint 保存成功
- 训练正常结束

初步结论：

- 第二组动态 loss scale 实验已跑通
- 训练过程中未出现 `overflow`
- `loss_scale` 在本次实验中保持为 `65536.0`
- 这说明当前这组 `100` 条数据、`Qwen3-8B` 单卡 LoRA 配置下，数值空间较稳定，没有触发动态 loss scale 的缩放逻辑

### 第一组与第二组对比结论

第一组：

- 固定 `loss_scale=1.0`
- 无 overflow
- 训练稳定结束

第二组：

- 动态 `loss_scale` 初始值 `65536.0`
- 无 overflow
- `loss_scale` 未发生回退
- 训练稳定结束

对比结论：

- 当前这组小样本实验下，固定 loss scale 和动态 loss scale 都能稳定运行
- 从目前结果看，没有证据表明第二组在稳定性上明显优于第一组
- 同时也没有证据表明当前任务已经逼近数值边界

因此，当前阶段可以得出：

- 这份单卡 Qwen3-8B LoRA 配置在 `100` 条样本、`seq_length=1024` 下数值稳定
- 后续如果继续做精度分析，重点应转向“长稳训练”而不是继续纠结 loss scale

## 关于 Dump 的定位

本次实验中测试了 `Dump`，主要目的仅为了解功能和产物形式，不作为当前问题定位主手段。

说明：

- 当前实验没有出现 `nan/inf`
- 没有 `overflow`
- `loss/global_norm` 虽有波动，但整体可训练
- 因此暂时不需要把 `Dump` 作为主线调试工具

结论：

- 当前 `Dump` 仅作为介绍性验证使用
- 正式精度排查仍以：
  - `loss`
  - `global_norm`
  - `local_norm`
  - `overflow`
  - `loss_scale`
  为主

只有当后续长稳实验出现：

- `nan/inf`
- `overflow`
- 某步 loss 明显异常
- 某步 global norm 明显爆炸

时，再进入 `Dump` 的逐层定位阶段更合适

## 下一步建议

建议进入阶段 3：长稳实验。

优先顺序：

1. 长稳实验 A：学习率置零
   - 目的：排除优化器更新影响
   - 观察 loss 是否仍异常漂移

2. 长稳实验 B：恢复 `lr=5e-6`
   - 目的：观察真实训练场景下长步数稳定性
   - 重点看 `global_norm` 是否持续 spike

如果后续长稳阶段出现异常，再决定是否正式启用 `Dump` 做逐层分析。
