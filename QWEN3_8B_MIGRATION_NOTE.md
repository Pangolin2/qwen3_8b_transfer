# Qwen3-8B 迁移说明

## 目标

本说明文档用于梳理 `Qwen3-8B` 在 `MindFormers r1.8.0` 中的迁移步骤、代码落点和 `YAML` 修改方法，便于后续统一撰写实验手册。

本文档重点回答 4 个问题：

- `Qwen3-8B` 在当前仓库里需要迁移哪些部分
- 每一步迁移对应哪些代码目录
- 哪些步骤是“复用已有实现”，哪些步骤需要新增配置
- `YAML` 文件应该怎么改，为什么这么改

## 参考资料

- 官方开发迁移文档（r1.8.0）：https://www.mindspore.cn/mindformers/docs/zh-CN/r1.8.0/advanced_development/dev_migration.html
- 官方开发迁移文档（master）：https://www.mindspore.cn/mindformers/docs/zh-CN/master/advanced_development/dev_migration.html
- `Qwen3` 模型文档：[README.md](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/README.md)

## 当前迁移定位

结合官方迁移文档和当前仓库代码，`Qwen3-8B` 在本项目中的迁移，不是“从零实现一个全新的模型”，而是“复用仓库里已经存在的 `Qwen3` 实现，并将其适配到本地 `Qwen3-8B` Hugging Face 模型目录和当前实验目标”。

这意味着迁移工作的重心不是重写模型主体，而是：

- 确认已有 `Qwen3` 模型配置与 `8B` 规格是否对齐
- 确认推理与训练网络能够正确加载 `Qwen3-8B` 权重
- 确认 tokenizer 和 generation config 从 Hugging Face 目录正确加载
- 通过 `YAML` 将推理、微调、性能测试链路串起来

## 总体流程

参考官方开发迁移文档，可以将 `Qwen3-8B` 迁移拆分为 6 步：

1. 确认迁移基线与目录结构
2. 迁移模型配置
3. 迁移模型实现
4. 迁移 tokenizer
5. 迁移权重
6. 迁移 `YAML` 与数据处理流程

对于当前仓库，最重要的结论是：

- 步骤 2 到步骤 5 的核心代码已经存在
- 当前真正需要重点维护的是步骤 6，即 `YAML` 的派生和验证流程
- 迁移后的首个验证动作应是单卡推理验证，而不是直接启动微调

## 步骤 1：确认迁移基线与目录结构

### 这一步要解决什么问题

官方迁移文档首先强调，要先找到结构最接近的已有模型作为迁移基线。

对 `Qwen3-8B` 来说，当前仓库已经存在完整的 `Qwen3` 代码与配置，因此最合理的基线不是其他模型，而是仓库内已有的 `Qwen3`。

### 需要关注的代码与目录

- `Qwen3` 文档：[README.md](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/README.md)
- 统一启动入口：[run_mindformer.py](/home/leon/Code/mindformers-r1.8.0/run_mindformer.py)
- 官方推理配置：[predict_qwen3.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/predict_qwen3.yaml)
- 官方微调配置：[finetune_qwen3.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3.yaml)

### 结论

当前 `Qwen3-8B` 迁移的起点应当是：

- 以官方 `predict_qwen3.yaml` 为推理基线
- 以官方 `finetune_qwen3.yaml` 为训练基线
- 只针对 `8B` 权重路径、单卡实验目标、LoRA 参数、本地数据集和性能分析需求做定向修改

## 步骤 2：迁移模型配置

### 官方迁移文档关注点

官方开发迁移文档指出，模型配置迁移是第一关键步骤。需要重点核对的通常包括：

- `vocab_size`
- `hidden_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `intermediate_size`
- `max_position_embeddings`
- 特殊 token id
- RoPE 参数

### 当前仓库的代码落点

- 配置类：[configuration_qwen3.py](/home/leon/Code/mindformers-r1.8.0/mindformers/models/qwen3/configuration_qwen3.py)

### 这份配置类在迁移中承担什么作用

`Qwen3Config` 是 `Qwen3` 模型的统一配置入口。模型构建时，训练网络、推理网络、生成配置、tokenizer 自动加载链路，都会依赖这个配置类。

因此迁移 `Qwen3-8B` 时，需要优先确认本地 Hugging Face 模型目录中的 `config.json` 能否与这里的配置定义正确对齐。

### 在当前项目里，哪些参数是“必须重点核对”的

建议至少核对以下字段是否和 `Qwen3-8B` 官方权重一致：

- `vocab_size`
- `hidden_size`
- `intermediate_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `max_position_embeddings`
- `rope_theta`
- `bos_token_id`
- `eos_token_id`
- `pad_token_id`

### 这一步与 `YAML` 的关系

这一步虽然主要落在 Python 配置类，但最终要落实到 `YAML`。

对于当前实验，最常见的配置相关 `YAML` 修改点有：

1. 模型目录路径

```yaml
pretrained_model_dir: "/mnt/hzl/10_qwen3_transfer/model/Qwen3-8B"
```

作用：

- 指向本地 Hugging Face 模型目录
- 让 `config.json`、`generation_config.json`、tokenizer 和 safetensors 权重一起被读取

2. 实验使用的序列长度

```yaml
model:
  model_config:
    seq_length: 1024
```

作用：

- 对训练任务指定当前实验实际使用的序列长度
- 避免数据侧和模型侧长度不一致

注意：

- `seq_length` 是实验配置，不等价于模型的原始 `max_position_embeddings`
- 当前单卡 LoRA 实验为了控时与控显存，将训练长度设为 `1024`

### 实际经验

当前实验里已经出现过一个直接由 `YAML` 配置不一致导致的错误：

```text
For 'Mul', input1.shape = [1, 1, 1024]
input2.shape = [1, 4096, 4096]
```

根因就是：

- 数据侧改成了 `1024`
- 模型侧 `seq_length` 仍保留更长长度配置

因此，模型配置迁移完成后，必须在训练 `YAML` 中再次检查与实验长度相关的字段是否全部同步。

## 步骤 3：迁移模型实现

### 官方迁移文档关注点

官方开发迁移文档要求模型类满足统一接口，并尽量复用已有模型实现。

### 当前仓库的代码落点

- 模型入口：[modeling_qwen3.py](/home/leon/Code/mindformers-r1.8.0/mindformers/models/qwen3/modeling_qwen3.py)
- 训练实现：[modeling_qwen3_train.py](/home/leon/Code/mindformers-r1.8.0/mindformers/models/qwen3/modeling_qwen3_train.py)
- 推理实现：[modeling_qwen3_infer.py](/home/leon/Code/mindformers-r1.8.0/mindformers/models/qwen3/modeling_qwen3_infer.py)
- 公共预训练模型基类：[utils.py](/home/leon/Code/mindformers-r1.8.0/mindformers/models/qwen3/utils.py)

### 当前实现的关键特点

1. 训练与推理网络分离

`Qwen3ForCausalLM` 会根据 `RUN_MODE` 自动切换：

- `predict` 时使用推理实现
- 其他模式使用训练实现

这个切换逻辑在 [modeling_qwen3.py](/home/leon/Code/mindformers-r1.8.0/mindformers/models/qwen3/modeling_qwen3.py#L39)。

2. HF 与 MindFormers 权重名映射已实现

在 [utils.py](/home/leon/Code/mindformers-r1.8.0/mindformers/models/qwen3/utils.py) 中，`weight_mapping` 已经定义了 HF 参数名到 MindFormers 参数名的对应关系。

3. 当前迁移重点不是改模型类，而是验证现有类能否正确加载 `Qwen3-8B`

因此，对当前 `Qwen3-8B` 实验，不建议一开始改模型实现。更合理的顺序是：

1. 先用推理 `YAML` 验证能否加载并生成文本
2. 再用训练 `YAML` 验证能否完成单卡 LoRA 跑通
3. 只有在权重加载、shape 或 forward 路径报错时，再回头定位模型代码

### 与 `YAML` 的关系

模型实现迁移本身主要不是改 `YAML`，但 `YAML` 决定了使用哪条运行路径。

最关键的相关字段是：

```yaml
run_mode: 'predict'
```

和

```yaml
run_mode: 'finetune'
```

作用：

- `predict` 会走推理网络
- `finetune` 会走训练网络

因此，迁移验证阶段必须至少准备两份 `YAML`：

- 一份推理验证配置
- 一份训练验证配置

## 步骤 4：迁移 tokenizer

### 官方迁移文档关注点

官方文档把 tokenizer 迁移列为单独步骤，因为：

- tokenizer 决定输入编码方式
- tokenizer 变化会影响推理文本表现和训练数据预处理

### 当前仓库的实际实现

当前 `Qwen3` 没有单独新增 `qwen3_tokenizer.py`，而是复用了通用自动加载逻辑：

- tokenizer 构建入口：[build_tokenizer.py](/home/leon/Code/mindformers-r1.8.0/mindformers/models/build_tokenizer.py)

在 [build_tokenizer.py](/home/leon/Code/mindformers-r1.8.0/mindformers/models/build_tokenizer.py#L72) 中可以看到：

- 当 `use_legacy: False` 时
- 直接通过 Hugging Face `AutoTokenizer.from_pretrained(pretrained_model_dir)` 加载 tokenizer

### 这对当前迁移意味着什么

这说明 `Qwen3-8B` 当前不是“开发一个新 tokenizer”，而是“保证本地 Hugging Face 模型目录完整且可加载”。

建议至少保证模型目录包含：

- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors.index.json`
- `model-*.safetensors`

### 与 `YAML` 的关系

tokenizer 迁移最关键的 `YAML` 字段是：

```yaml
use_legacy: False
pretrained_model_dir: "/mnt/hzl/10_qwen3_transfer/model/Qwen3-8B"
```

作用：

- 启用 HF 兼容路径
- 直接从模型目录中读取 tokenizer 和相关配置

### 训练 `YAML` 中 tokenizer 的额外落点

在当前单卡 LoRA 训练配置中，tokenizer 还出现在数据处理链路里：

```yaml
handler:
  - type: AlpacaInstructDataHandler
    seq_length: 1024
    padding: False
    tokenizer:
      trust_remote_code: True
      padding_side: 'right'
```

作用：

- 使用模型目录中的 tokenizer 对训练数据编码
- 控制 padding 方向和 tokenizer 远程代码信任行为

因此，推理链路和训练链路都依赖同一个模型目录中的 tokenizer 文件。

## 步骤 5：迁移权重

### 官方迁移文档关注点

官方文档要求准备权重，并在需要时完成 PyTorch/Hugging Face 与 MindSpore 权重格式之间的转换。

### 当前仓库的实际情况

`Qwen3` 当前已经支持直接从 Hugging Face 模型目录加载权重。

相关说明在：

- 权重转换说明：[toolkit/weight_convert/qwen3/README.md](/home/leon/Code/mindformers-r1.8.0/toolkit/weight_convert/qwen3/README.md)

相关脚本在：

- MindFormers 训练权重反转 HF：[reverse_mcore_qwen3_weight_to_hf.py](/home/leon/Code/mindformers-r1.8.0/toolkit/weight_convert/qwen3/reverse_mcore_qwen3_weight_to_hf.py)
- safetensors 合并工具：[unified_safetensors.py](/home/leon/Code/mindformers-r1.8.0/toolkit/safetensors/unified_safetensors.py)

### 当前实验应采用哪条路径

对 `Qwen3-8B` 当前实验，更推荐的路径是：

1. 直接使用 Hugging Face 原始模型目录
2. 通过 `pretrained_model_dir` 加载权重
3. 先验证推理
4. 再验证训练

原因：

- 这样最省步骤
- 可以减少离线转换链路引入的新问题
- 更适合当前“先验证迁移是否成功，再做 LoRA 和性能测试”的实验目标

### 与 `YAML` 的关系

这一部分最关键的 `YAML` 字段有两个。

1. 模型权重目录

```yaml
pretrained_model_dir: "/mnt/hzl/10_qwen3_transfer/model/Qwen3-8B"
```

2. 是否自动转换权重

```yaml
auto_trans_ckpt: False
```

对于当前单卡实验，这个设置的含义是：

- 不做分布式自动切分转换
- 直接按单卡路径加载本地 Hugging Face 权重

如果后续切换到分布式训练，再考虑是否打开：

```yaml
auto_trans_ckpt: True
```

### 推理验证为什么能作为权重迁移是否成功的第一关

因为推理验证同时覆盖了：

- 配置文件加载
- tokenizer 加载
- generation config 加载
- safetensors 权重加载
- 推理网络 forward

只要推理能正常返回相关文本，说明权重迁移链路大概率已经打通。

## 步骤 6：迁移 `YAML` 与数据处理流程

### 为什么这一部分最重要

官方开发迁移文档特别强调，`YAML` 是任务配置的总入口。它不仅包含模型配置，还包含：

- 训练参数
- 推理参数
- 数据集配置
- 并行配置
- 上下文配置
- callback 配置

对当前 `Qwen3-8B` 项目，迁移工作的绝大部分可操作内容，最终都落实在 `YAML` 上。

### 当前项目中建议维护的 `YAML` 层级

建议将 `YAML` 分为 4 类：

1. 官方基线 `YAML`
2. 迁移验证 `YAML`
3. 训练跑通 `YAML`
4. 实验派生 `YAML`

对应文件如下：

- 官方推理基线：[predict_qwen3.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/predict_qwen3.yaml)
- 官方训练基线：[finetune_qwen3.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3.yaml)
- 迁移后推理验证：[predict_qwen3_8b_verify.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/predict_qwen3_8b_verify.yaml)
- 单卡 LoRA 跑通：[finetune_qwen3_8b_lora_1card_debug.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_debug.yaml)
- 单卡性能基线：[finetune_qwen3_8b_lora_1card_perf_baseline.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_perf_baseline.yaml)
- 单卡纯性能测量：[finetune_qwen3_8b_lora_1card_perf_measure.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_perf_measure.yaml)
- 单卡重计算对照：[finetune_qwen3_8b_lora_1card_perf_recompute.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_perf_recompute.yaml)

### 这一阶段最关键的 `YAML` 修改点

下面按用途拆开说明。

### 6.1 推理验证 `YAML` 修改点

基线文件：

- [predict_qwen3.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/predict_qwen3.yaml)

派生文件：

- [predict_qwen3_8b_verify.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/predict_qwen3_8b_verify.yaml)

重点修改项如下。

1. 输出目录

```yaml
output_dir: './output_qwen3_8b_predict_verify'
```

作用：

- 将迁移验证日志和输出结果与其他任务隔离

2. 模型目录

```yaml
pretrained_model_dir: '/mnt/hzl/10_qwen3_transfer/model/Qwen3-8B'
```

作用：

- 指向当前迁移后的 `Qwen3-8B` Hugging Face 权重目录

3. 单卡推理

```yaml
use_parallel: False
parallel_config:
  data_parallel: 1
  model_parallel: 1
```

作用：

- 先最小化验证迁移链路
- 避免多卡并行问题干扰迁移判断

4. 顶层生成配置

```yaml
generation_config:
  max_length: 128
  do_sample: False
  top_k: 1
  top_p: 1.0
  temperature: 1.0
  repetition_penalty: 1.0
  use_past: True
```

作用：

- 覆盖模型目录默认 `generation_config.json`
- 避免默认 `max_length` 太短导致验证时只生成很短的回复

特别说明：

- 这里必须使用顶层 `generation_config`
- 不能只写 `generation`
- 当前项目中真正会覆盖模型默认生成参数的是顶层 `generation_config`

5. 关闭 legacy tokenizer 路径

```yaml
use_legacy: False
```

作用：

- 走 Hugging Face 兼容加载路径
- 直接使用模型目录中的 tokenizer 与生成配置

### 6.2 训练跑通 `YAML` 修改点

基线文件：

- [finetune_qwen3.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3.yaml)

派生文件：

- [finetune_qwen3_8b_lora_1card_debug.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_debug.yaml)

重点修改项如下。

1. 模型目录改为 `Qwen3-8B`

```yaml
pretrained_model_dir: "/mnt/hzl/10_qwen3_transfer/model/Qwen3-8B"
```

2. 切换到单卡

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

作用：

- 降低迁移验证复杂度
- 先证明单卡训练路径可用

3. 数据集改为本地小样本

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

作用：

- 快速验证训练链路
- 方便观察 loss、global_norm、overflow

4. 统一实验序列长度

```yaml
handler:
  - type: AlpacaInstructDataHandler
    seq_length: 1024
  - type: PackingHandler
    seq_length: 1024
```

```yaml
model:
  model_config:
    seq_length: 1024
```

作用：

- 保证数据处理与模型前向一致
- 避免 shape mismatch

5. 开启 LoRA

```yaml
model:
  model_config:
    pet_config:
      pet_type: lora
      lora_rank: 8
      lora_alpha: 16
      lora_dropout: 0.1
      target_modules: '.*word_embeddings|.*linear_qkv|.*linear_proj|.*linear_fc1|.*linear_fc2'
      freeze_include: ['*']
      freeze_exclude: ['*lora*']
```

作用：

- 将训练目标切换为参数高效微调
- 冻结主干，仅训练 LoRA 参数

### 6.3 性能实验 `YAML` 修改点

当前已维护 3 份性能相关配置：

- [finetune_qwen3_8b_lora_1card_perf_baseline.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_perf_baseline.yaml)
- [finetune_qwen3_8b_lora_1card_perf_measure.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_perf_measure.yaml)
- [finetune_qwen3_8b_lora_1card_perf_recompute.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_perf_recompute.yaml)

三者关系如下。

1. `perf_baseline`

作用：

- 建立单卡稳态性能基线
- 打开轻量 profile

关键修改：

```yaml
profile: True
profile_start_step: 5
profile_stop_step: 8
```

2. `perf_measure`

作用：

- 纯测稳态吞吐
- 避免 profile 和 ckpt 干扰

关键修改：

```yaml
callbacks:
  - type: MFLossMonitor

profile: False
```

以及：

```yaml
num_parallel_workers: 8
prefetch_size: 4
```

3. `perf_recompute`

作用：

- 与基线配置对照
- 评估重计算的时间代价

关键修改：

```yaml
recompute_config:
  recompute: True
```

### 6.4 为什么要特别强调 `YAML`

对当前 `Qwen3-8B` 项目，`YAML` 是迁移工作的主轴，而不只是“启动参数文件”。

原因如下：

1. `YAML` 决定了使用哪套运行路径

- `predict` 还是 `finetune`
- 单卡还是多卡
- 是否启用 profile
- 是否启用 recompute

2. `YAML` 决定了是否正确对齐模型、数据和 tokenizer

- `pretrained_model_dir`
- `seq_length`
- `handler`
- `use_legacy`

3. `YAML` 决定了实验结论是否可信

- 如果 profile 和 ckpt 逻辑混在一起，性能结论会被污染
- 如果生成参数没写在真正生效的位置，推理验证结论会失真

因此，手册撰写时建议把 `YAML` 单列成迁移说明的重点章节，而不是附录。

## 建议的迁移验证顺序

结合当前项目经验，建议按下面顺序推进：

1. 先检查 Hugging Face 模型目录完整性
2. 用 [predict_qwen3_8b_verify.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/predict_qwen3_8b_verify.yaml) 做单卡推理验证
3. 推理通过后，用 [finetune_qwen3_8b_lora_1card_debug.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_debug.yaml) 做单卡 LoRA 跑通
4. 跑通后再进入性能和精度实验

## 迁移验证闭环

为了让迁移验证形成一个真正可执行、可复盘的闭环，建议将验证流程固化为下面 4 个环节。

### 环节 1：模型目录完整性检查

目标：

- 确认 `Qwen3-8B` Hugging Face 模型目录可用于 MindFormers 推理与训练

建议检查的文件：

- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors.index.json`
- `model-*.safetensors`

当前目录：

- 模型目录：`/mnt/hzl/10_qwen3_transfer/model/Qwen3-8B`

通过标准：

- 配置文件齐全
- tokenizer 文件齐全
- safetensors 权重完整

### 环节 2：推理验证

目标：

- 验证配置、tokenizer、权重、推理网络和生成参数是否整体打通

使用文件：

- 推理验证配置：[predict_qwen3_8b_verify.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/predict_qwen3_8b_verify.yaml)
- 推理验证脚本：[run_qwen3_8b_verify_infer.sh](/home/leon/Code/mindformers-r1.8.0/scripts/run_qwen3_8b_verify_infer.sh)

建议命令：

```bash
cd /mnt/hzl/10_qwen3_transfer/mindformers-r1.8.0
bash scripts/run_qwen3_8b_verify_infer.sh '请用中文用三句话介绍你自己，并说明你能帮助用户完成哪些任务。'
```

这一环节为什么重要：

- 它同时覆盖模型目录读取、tokenizer 读取、generation config 覆盖、权重加载和推理前向
- 是判断迁移是否成功的第一道关

重点检查项：

1. 日志中的 `Generation Config`

预期应接近：

```text
max_length: 128
do_sample: False
top_k: 1
top_p: 1.0
use_past: True
```

如果仍然出现：

```text
max_length: 20
do_sample: True
top_p: 0.95
```

则说明：

- 实际生效的还不是验证配置中的生成参数
- 需要优先回看是否使用了最新 `YAML`

2. 推理结果

通过标准：

- 任务可以正常启动
- 无权重加载错误
- 返回文本与提示词相关
- 不应只返回 `Sure`、`好的` 这类明显过短回复
- 正常生成 `text_generation_result.txt`

### 环节 3：训练跑通验证

目标：

- 在推理链路已通过的前提下，验证训练链路可用

使用文件：

- 训练跑通配置：[finetune_qwen3_8b_lora_1card_debug.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_debug.yaml)

这一环节主要验证：

- 数据集是否能被正确加载
- tokenizer 是否能用于训练数据编码
- `seq_length` 是否一致
- LoRA 配置是否能正常插入训练网络
- loss、global_norm、overflow 是否正常输出

通过标准：

- 训练能启动
- 首 step 编译后可持续迭代
- `overflow=False`
- `loss_scale` 正常
- checkpoint 可保存

特别强调：

- 如果推理验证未通过，不要直接进入训练验证
- 否则很容易把模型目录或权重问题误判成训练问题

### 环节 4：实验派生验证

目标：

- 在跑通配置基础上，进入性能、精度、对照实验

相关文件：

- 单卡性能基线：[finetune_qwen3_8b_lora_1card_perf_baseline.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_perf_baseline.yaml)
- 单卡纯性能测量：[finetune_qwen3_8b_lora_1card_perf_measure.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_perf_measure.yaml)
- 单卡重计算对照：[finetune_qwen3_8b_lora_1card_perf_recompute.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_perf_recompute.yaml)
- 精度分析第一组：[finetune_qwen3_8b_lora_1card_acc_group1.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_acc_group1.yaml)
- 精度分析第二组：[finetune_qwen3_8b_lora_1card_acc_group2.yaml](/home/leon/Code/mindformers-r1.8.0/configs/qwen3/finetune_qwen3_8b_lora_1card_acc_group2.yaml)

通过标准：

- 每个实验都应在“推理通过 + 训练跑通”后开展
- 每份派生 `YAML` 应只承载单一实验目标
- 结果分析时要能明确追溯到对应配置文件

## 推理脚本在闭环中的作用

推理脚本不是简单的便捷启动工具，而是迁移闭环中的标准入口。

脚本文件：

- [run_qwen3_8b_verify_infer.sh](/home/leon/Code/mindformers-r1.8.0/scripts/run_qwen3_8b_verify_infer.sh)

它在闭环中的作用有 3 点：

1. 固化推理验证入口

- 避免每次手动拼接命令
- 保证推理验证参数一致

2. 固化默认 prompt

- 默认使用更适合验证长回复的中文 prompt
- 避免因提示词过短导致误判模型回复质量

3. 固化配置文件路径

- 默认绑定 `predict_qwen3_8b_verify.yaml`
- 将迁移验证和其他预测任务分开

因此，在迁移手册中，推理脚本应被视为“迁移验证闭环的一部分”，而不是附属工具。

## 推荐的闭环记录模板

建议后续在手册或实验记录中固定记录以下信息：

实验名称：
`Qwen3-8B 迁移验证`

模型目录：

使用配置：

使用脚本：

输入 prompt：

Generation Config 是否符合预期：

推理结果：

是否生成 `text_generation_result.txt`：

训练跑通配置：

训练是否成功启动：

step2 后是否稳定：

后续派生实验：

结论：

## 当前项目推荐的闭环主线

结合当前已维护的脚本和配置，建议后续所有 `Qwen3-8B` 迁移验证都走下面这条标准主线：

1. 检查 `/mnt/hzl/10_qwen3_transfer/model/Qwen3-8B` 目录完整性
2. 运行 [run_qwen3_8b_verify_infer.sh](/home/leon/Code/mindformers-r1.8.0/scripts/run_qwen3_8b_verify_infer.sh) 做推理验证
3. 查看日志中的 `Generation Config` 和最终输出文本
4. 推理通过后，运行 `finetune_qwen3_8b_lora_1card_debug.yaml` 做单卡训练跑通
5. 跑通后再进入性能和精度实验

这条链路的价值在于：

- 每一步都可独立判断是否通过
- 每一步都能明确回溯到对应的 `YAML` 和脚本
- 能最大限度降低定位问题时的混淆

## 当前项目中建议重点保留的迁移产物

建议后续手册中持续保留下列内容：

- 推理验证配置文件
- 单卡训练跑通配置文件
- 性能基线与对照配置文件
- 迁移后模型目录结构说明
- 关键 `YAML` 修改点及原因
- 首轮推理和训练验证日志摘要

## 结论

`Qwen3-8B` 在当前仓库里的迁移，本质上是“基于已有 `Qwen3` 实现的配置化迁移”。真正需要反复打磨的不是模型主干代码，而是下面三件事：

- Hugging Face 模型目录是否完整可加载
- `YAML` 是否把推理、训练、数据和实验目标正确串起来
- 每一阶段是否先通过最小验证，再进入下一阶段

在这三件事里，最应被强调的是 `YAML`。因为当前实验里已经反复证明：

- 一处 `seq_length` 不一致，会直接导致训练编译失败
- 生成参数写错层级，会导致推理验证得到过短回复
- profile 和 ckpt 配置混在一起，会污染性能结论

因此，后续实验手册在写迁移章节时，应把 `YAML` 修改点放在最核心的位置描述。
