# Qwen3-8B 单卡 LoRA 性能实验记录

## 目标

本说明文档用于记录 `Qwen3-8B` 在单卡 `LoRA` 场景下的性能实验，当前优先关注：

1. 单卡性能基线
2. 重计算对照实验

其余性能实验如数据处理对照、序列长度敏感性、多卡通信分析，当前优先级降低，后续再补。

## 参考资料

- 官方性能优化文档：https://www.mindspore.cn/mindformers/docs/zh-CN/dev/advanced_development/performance_optimization.html
- MindSpore 性能调试文档：https://www.mindspore.cn/docs/zh-CN/r1.8/migration_guide/performance_optimization.html

代码对应位置：

- profiling callback 构建：[trainer.py](/home/leon/Code/mindformers-r1.8.0/mindformers/trainer/trainer.py)
- `ProfileMonitor` 实现：[callback.py](/home/leon/Code/mindformers-r1.8.0/mindformers/core/callback/callback.py)
- 训练日志输出（step time / throughput）：[callback.py](/home/leon/Code/mindformers-r1.8.0/mindformers/core/callback/callback.py)

## 当前场景

- 模型：`Qwen3-8B`
- 微调方式：`LoRA`
- 卡数：`1`
- 数据量：`100` 条
- `seq_length`：`1024`
- 目标：建立单卡性能基线，理解重计算带来的时间/内存权衡

## 为什么先做这两组实验

当前阶段不需要直接做多卡并行优化，原因：

- 目标是理解单卡训练的稳定态性能
- 单卡下没有通信瓶颈干扰
- 更容易区分：
  - 首 step 编译开销
  - 稳定态 step 耗时
  - 是否值得开启重计算

当前实验需要特别注意：

- `step 1` 的时间不能作为性能基线
- 基线应以 `step 2` 之后的稳定态 `per_step_time` 为准

## 配置文件列表

### 1. 单卡性能基线

- 文件：[finetune_qwen3_8b_lora_1card_perf_baseline.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_perf_baseline.yaml)

用途：

- 建立单卡稳定态性能基线
- 打开轻量 profiling
- 不开启重计算

### 2. 单卡重计算对照

- 文件：[finetune_qwen3_8b_lora_1card_perf_recompute.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_perf_recompute.yaml)

用途：

- 与基线配置做对照
- 评估 `recompute=True` 带来的性能变化

## 两组实验的核心差异

### 基线配置

```yaml
recompute_config:
  recompute: False
  select_recompute: False
  parallel_optimizer_comm_recompute: False
  mp_comm_recompute: False
```

### 重计算配置

```yaml
recompute_config:
  recompute: True
  select_recompute: False
  parallel_optimizer_comm_recompute: False
  mp_comm_recompute: False
```

除这一处差异外，其他核心配置保持一致，便于做对照。

## profiling 配置

两组性能实验都使用相同的 profiling 配置：

```yaml
profile: True
profile_start_step: 5
profile_stop_step: 8
init_start_profile: False
profile_communication: False
profile_memory: False
profile_output: ...
profiler_level: 1
profile_rank_ids: [0]
profile_pipeline: False
with_stack: False
data_simplification: True
mstx: False
```

说明：

- `step 5 ~ 8` 用于避开首 step 编译开销
- 单卡下 `profile_communication=False`
- 当前先不打开 `profile_memory`
- `profiler_level=1` 足够做基线分析

## 运行命令

### 单卡性能基线

```bash
cd /mnt/hzl/10_qwen3_transfer/mindformers-r1.8.0
python run_mindformer.py \
  --config configs/qwen3/finetune_qwen3_8b_lora_1card_perf_baseline.yaml \
  --run_mode finetune
```

### 单卡重计算对照

```bash
cd /mnt/hzl/10_qwen3_transfer/mindformers-r1.8.0
python run_mindformer.py \
  --config configs/qwen3/finetune_qwen3_8b_lora_1card_perf_recompute.yaml \
  --run_mode finetune
```

## 每组实验要收集的结果

建议至少记录：

- `step1 per_step_time`
- `step2 ~ step25` 稳定态的典型 `per_step_time`
- `train_throughput_per_npu`
- profile 输出目录
- profile 中的主要热点模块
- 是否成功保存 checkpoint

建议记录模板：

```text
实验名称：
配置文件：
唯一改动：
是否成功运行：
step1 per_step_time：
稳定态 per_step_time：
稳定态吞吐：
profile 输出目录：
主要热点：
checkpoint 是否成功：
结论：
```

## 对比时重点看什么

### 基线实验重点

- 稳定态 step 时间
- 吞吐
- profile 中 attention / matmul 等算子占比
- 是否存在明显迭代间隙

### 重计算实验重点

- 与基线相比，稳定态 step 时间增加了多少
- 是否出现明显吞吐下降
- 是否换来更低的内存压力

当前由于暂未专门开 `profile_memory`，本轮主要先看时间代价。

## 预期结论

在当前这个单卡 `Qwen3-8B + LoRA + 100条数据 + seq_length=1024` 场景下，合理预期是：

- 基线配置更快
- 重计算配置更慢

如果结果确实如此，说明：

- 当前显存并不紧张
- 后续没有必要为了“保险”默认开启重计算

如果重计算带来的时间损失很小，则说明：

- 后续上更长序列时，重计算可能是可接受的折中选项

## 后续计划

本轮完成后，再决定是否继续：

1. 数据处理对照
2. 序列长度敏感性对照
3. 多卡性能实验

当前优先级最低的是多卡通信分析。
