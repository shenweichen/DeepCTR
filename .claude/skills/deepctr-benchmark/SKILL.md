---
name: deepctr-benchmark
description: >-
  运行 DeepCTR 模型 benchmark 套件（benchmarks/），在真实或自带数据上端到端验证、
  对比 CTR 模型并产出排序后的 leaderboard。当被要求 benchmark、对比、验证或评测
  DeepCTR 模型（单任务 CTR / 多任务 / 序列），或在改动代码后重跑、扩展 benchmark 时使用。
  Use to benchmark, compare, or validate DeepCTR CTR models.
---

# DeepCTR benchmark 套件

一个自包含的 harness（`benchmarks/`），训练 DeepCTR 模型并写出排序的 leaderboard。
三个 track：`single`（21 个单任务 CTR 模型，Criteo）、`multitask`
（SharedBottom/ESMM/MMOE/PLE，Census-Income）、`sequence`（DIN/BST/DSIN，DIEN 在
TF≥2.0 上跳过）。完整文档见 `benchmarks/README.md`，历史结果见 `benchmarks/RESULTS.md`。

## 每次必做的两件事

1. **强制 CPU**：加 `CUDA_VISIBLE_DEVICES=""`。本机 GPU 存在 CUDA/PTX 不兼容，在 GPU 上
   每个模型都会以 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` 失败。harness 会逐模型隔离错误，
   所以在 GPU 上你会得到一张全部 `failed` 的 leaderboard 而非崩溃——解决办法永远是走 CPU。
2. **用 legacy Keras**：`TF_USE_LEGACY_KERAS=1`（DeepCTR 面向 Keras 2 API；在 TF≥2.16 上
   即 `tf-keras`）。套件自身会设置它,但用 pytest 调用时要显式设上。

长时间 sweep 放后台跑并轮询日志——真实数据上 21 个单任务模型的完整 sweep 在 CPU 上需要
几分钟到约 30 分钟。

## 本机数据

真实数据在 `benchmarks/data/`（git-ignored）。若缺失,需重新生成或下载（见 README;内置的
Criteo DAC URL 已失效——匿名 HF 镜像会限流,下载 `reczoo/Criteo_x1` 需要 `HF_TOKEN`）。

| 文件 | 样本数 | 用途 |
| ---- | ---- | --- |
| `benchmarks/data/criteo_x1_500k.csv` | 500,000 | 单任务,AUC 有意义 |
| `benchmarks/data/criteo_x1_200k.csv` | 200,000 | 单任务,更快 |
| `benchmarks/data/census-income.data`（+ `.test`） | 199,523（+99,762） | 多任务,官方切分 |
| `examples/criteo_sample.txt` | 200 | 仅冒烟——AUC 是噪声 |

## 命令

```bash
# 快速冒烟（1 epoch、每 track 2 个模型、自带数据）—— 验证 harness 正常
CUDA_VISIBLE_DEVICES="" python -m benchmarks.benchmark --track all --quick

# 单任务,真实 Criteo 500k —— 主力对比
CUDA_VISIBLE_DEVICES="" python -m benchmarks.benchmark --track single \
  --data-path benchmarks/data/criteo_x1_500k.csv \
  --epochs 1 --batch-size 1024 --val-split 0 --seed 2020

# 多任务,真实 Census,使用其官方 train/test 切分
CUDA_VISIBLE_DEVICES="" python -m benchmarks.benchmark --track multitask \
  --data-path benchmarks/data/census-income.data --epochs 3 --batch-size 1024

# 子集 / 排除慢模型;例如丢掉交互密集的几个以省时间
#   --models DeepFM,DCN,xDeepFM     只跑这些
#   --exclude FiBiNET,ONN,DeepFEFM,FwFM,FGCNN   仅 ONN 就有 77M 参数 / 约 245s
```

leaderboard 打印到 stdout,并写出 `benchmarks/results/<track>_<dataset>.csv` 和 `.md`。
要长期保存一次运行结果,把数字抄进 `benchmarks/RESULTS.md`（`results/` 目录是 git-ignored）。

## 训练 / 测试集划分 —— 这点要做对

- **默认 = 随机切分**,`--test-size 0.2`,固定 `--seed`。对 Criteo_x1 / DAC 是正确的:该数据
  已匿名化并打散、无时间戳,所以随机切分无穿越且为标准做法。
- **`--temporal-split`（可选 `--time-col COL`）**:按时间留出——最近 `test_size` 比例作为测试集,
  杜绝未来行泄漏进训练。有 `--time-col` 时按它排序,否则信任文件既有顺序。仅用于带时间顺序的
  日志;自带的 Criteo 没有时间戳。
- **Census 官方切分**:当 `--data-path` 为 `census-income.data` 且同目录存在 `census-income.test`
  时,loader 自动使用官方划分（编码器在并集上 fit）。不要再把它打散重切。
- **`--val-split`** 从训练集中切出验证集,但**没有接任何早停 / checkpoint**——它不影响训练过程
  也不参与模型选择。固定 epoch 的短跑应设 `--val-split 0`,让训练用满整个训练集;只有在加了
  EarlyStopping callback 时才保留它。

## 结果解读

- 报告的 AUC/LogLoss 来自 hold-out 的**测试集**（`fit` 从未见过）。
- 要有足够数据才有真实排名:200 行自带数据上 AUC≈0.5 是噪声;到 200k–500k 时,深层交互模型
  （DeepFEFM/FiBiNET/DeepFM/xDeepFM）才与简单模型拉开。更多 epoch 会抬高绝对 AUC,但很少改变
  头部梯队。
- 汇报 leaderboard 时务必同时注明切分方式与 epoch 数——排名只在相同 数据/切分/epoch 下可比。

## 测试

```bash
CUDA_VISIBLE_DEVICES="" TF_USE_LEGACY_KERAS=1 python -m pytest tests/benchmark_test.py -q
```

覆盖每个 track 在小数据上的运行,以及切分保证（时序留出把最近的行作为测试集;Census 使用
官方 `.test`）。
