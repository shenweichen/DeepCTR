> **Status:** original design plan for the model-onboarding pipeline, kept in-repo
> for reference across machines. For day-to-day **usage** see
> [`benchmarks/onboard/README.md`](./README.md) and the repo-root `AGENTS.md`.
>
> A few details evolved during implementation:
> - Templates are split per category: `model_single.py.tmpl`,
>   `model_sequence.py.tmpl`, `model_multitask.py.tmpl`, `layer.py.tmpl`,
>   `test_single.py.tmpl`, `test_sequence.py.tmpl`, `test_multitask.py.tmpl`.
> - `discover --refresh` prints a tool-agnostic research prompt (no hard
>   dependency on any specific agent's web-search tool).
> - OneTrans's positional-encoding class was renamed to
>   `SinusoidalPositionEncoding` to avoid a name collision with
>   `deepctr.layers.sequence.PositionEncoding`; a serializable `TokenSlice`
>   layer replaced an inline `Lambda`.
> - `docs.py` updates README / Features / rst+toctree / History / RESULTS;
>   `docs/source/index.rst` is left to manual edits (no stable counter anchor).

---

# 计划：DeepCTR 新模型自动化加入流程（model onboarding pipeline）

## Context（为什么做这件事）

DeepCTR 收录的模型停留在几年前，没有跟进近年（2021+）业界/学界的新 CTR 模型，影响项目的影响力与可持续发展。用户希望建立一条**自动化的新模型加入流程**，让"发现 → 实现 → 验证 → 文档"四个环节标准化、可复用，从而让项目能持续吸收新模型、保持活力。

现状暴露的核心痛点（以正在手工加入的 **OneTrans** 为活证据）：手工加一个模型要改 6 个分散的集成点，极易漏接。OneTrans 当时只接了一半——
- ✅ `deepctr/layers/onetrans.py`、`deepctr/models/sequence/onetrans.py` 已实现；
- ✅ 已在 `deepctr/layers/__init__.py` import、`deepctr/models/sequence/__init__.py` import；
- ❌ **未**注册到 `deepctr/layers/__init__.py` 的 `custom_objects`（导致 save/load 失败）；
- ❌ **未**在 `deepctr/models/__init__.py` 的 import 行与 `__all__` 中导出（`from deepctr.models import OneTrans` 会失败）；
- ❌ 无单测、无 `benchmarks/registry.py` 条目、无文档。

目标产物：一个**独立 CLI 工具** `python -m benchmarks.onboard <command>`，覆盖发现/实现脚手架/benchmark 验证/文档四阶段，并用 **OneTrans + FinalMLP + MaskNet** 跑通验证。运行方式为独立 CLI，不强绑 CI。

---

## 总体架构

新增子包 `benchmarks/onboard/`，复用现有 benchmark 基础设施（`benchmarks/registry.py`、`benchmark.py`、`common.py`、`metrics.py`）与模型基建（`deepctr/inputs.py` 的 `build_input_features`/`input_from_feature_columns`/`combined_dnn_input`/`get_linear_logit`、`deepctr/layers`、`tests/utils.py` 的 `check_model`/`get_test_data`）。

```
benchmarks/onboard/
  __init__.py
  __main__.py          # CLI 入口：python -m benchmarks.onboard <cmd>
  candidates.json      # 候选模型知识库（发现阶段产物，受版本控制）
  discover.py          # 阶段1：发现
  scaffold.py          # 阶段2：脚手架/接线
  verify.py            # 阶段3：benchmark 验证（正确性+效果）
  docs.py              # 阶段4：文档自动更新
  audit.py             # 横切：集成完整性校验（抓 OneTrans 式半接线）
  templates/           # 模型/单测/layer 代码模板（按 category 拆分）
```

CLI 子命令：`discover` / `scaffold` / `verify` / `docs` / `audit` / `onboard`（串联）。

---

## 阶段 1 — 发现（`discover.py`）

**产物**：`benchmarks/onboard/candidates.json`，每条候选含固定 schema：
`name, paper_title, year, venue, category(single|sequence|multitask), one_liner, ref_impl_url, paper_metric(如 {dataset:"Criteo", AUC:0.8xx}), difficulty, status(candidate|implemented|skipped)`。

**实现**：
- `discover --refresh`：打印工具无关的研究提示，让执行 agent 用自身联网检索能力扫 arXiv、paperswithcode CTR 榜、近年 KDD/RecSys/CIKM/SIGIR/WWW、FuxiCTR/BARS 等基准库，抽取候选并按"与 DeepCTR feature-column 接口契合度 + 引用度 + 是否已收录"排序，合并进 `candidates.json`（去重、保留 `status`）。
- `discover --list`：离线列出/筛选候选，标出未实现项。

> 说明：模型选型本质需要人/LLM 判断，CLI 负责把"检索 → 结构化 → 排序 → 标记状态"自动化，并产出可评审的候选表。

---

## 阶段 2 — 实现脚手架与接线（`scaffold.py`）

消灭 6 点手工 checklist。`scaffold <name> --category single|sequence|multitask [--with-layer] [--wire-only]`：

1. 从 `templates/` 生成：
   - `deepctr/models/<name>.py`（或 `sequence/`、`multitask/` 子目录），遵循工厂函数范式：`build_input_features → input_from_feature_columns → combined_dnn_input → [模型核心 # TODO] → DNN → Dense(1) → PredictionLayer → Model`（参考 `deepctr/models/wdl.py`）。核心交互层留明确 `# TODO` 占位，由人/LLM 补全数学。
   - 需自定义层时 `deepctr/layers/<name>.py`（`--with-layer`）。
   - `tests/models/<Name>_test.py`，用 `get_test_data` + `check_model`（照搬 `tests/models/DeepFM_test.py` 范式）。
2. **自动接线全部注册点**（锚点安全插入，幂等）：
   - `deepctr/models/__init__.py`：import 行 + `__all__`；
   - 子包 `deepctr/models/{sequence,multitask}/__init__.py`：import；
   - `deepctr/layers/__init__.py`：layer import + **`custom_objects` 字典**（OneTrans 漏的就是这一项）；
   - `benchmarks/registry.py`：对应 `SINGLE_TASK_MODELS` / `MULTITASK_MODELS` / `SEQUENCE_MODELS` 加 builder（sequence 自动生成 `_build_<name>` 处理 head/embedding 约束）。
- `--wire-only`：模型已存在（如 OneTrans）时只补接线 + 生成缺失的单测/registry 条目，不重生成模型主体。

---

## 阶段 3 — benchmark 验证（`verify.py`）

`verify <name>` 产出"正确性 + 效果"双结论与报告（`benchmarks/onboard/reports/<name>.md`）。

- **正确性**：跑 `pytest tests/models/<Name>_test.py`（编译/训练 1 epoch/save-load 往返——直接验证 `custom_objects` 是否齐全）；并对该模型跑 `audit`。
- **效果**：复用 `benchmarks.benchmark.run_single/run_multitask/run_sequence`，跑 `<新模型 + 同 track 基线（single=DeepFM / sequence=DIN / multitask=SharedBottom）>`，比较 AUC；若候选含 `paper_metric` 则与论文报告值对标，给出 verdict。
- 强制 `CUDA_VISIBLE_DEVICES="" TF_USE_LEGACY_KERAS=1`。

---

## 横切 — 集成完整性校验（`audit.py`）

`audit [--name X]`：自动发现 `deepctr.models` 下所有模型符号，逐项断言全接线，输出差距表：
可 `from deepctr.models import X` ✓ / 在 `__all__` ✓ / 自定义层在 `custom_objects` ✓ / 有 `tests/models/X_test.py` ✓ / 有 `benchmarks/registry.py` builder ✓。
这能立刻把 OneTrans 当时的半接线状态报出来。

---

## 阶段 4 — 文档自动更新（`docs.py`）

`docs <name>` 用锚点插入，幂等更新：
- `README.md` 模型表追加一行（名称 + 论文链接）；
- `docs/source/Features.md` 追加 `### <Name>` 小节；
- 新建 `docs/source/deepctr.models[.sequence|.multitask].<name>.rst` autodoc stub，并加入 `docs/source/deepctr.models.rst` 的 toctree；
- `docs/source/History.md` 追加变更行；
- `benchmarks/RESULTS.md` 记录该模型最近一次 `verify` 结果。

---

## 已落地的模型

1. **OneTrans**（序列）— 走 `scaffold --wire-only` 补全接线，`verify` 阶段抓出并修复 4 个真实 bug（positional `add_weight`、`tensorflow.python.keras` import、`Lambda` 序列化 → `TokenSlice`、`PositionEncoding` 类名冲突 → `SinusoidalPositionEncoding`）。
2. **FinalMLP**（AAAI 2023）、**MaskNet**（DLP-KDD 2021）（单任务）— 走完整 `discover → scaffold → 补核心数学 → verify → docs`，证明全链路可复用。

---

## 关键复用点（不要重造）

- 模型骨架：`deepctr/inputs.py`、`deepctr/layers`（`DNN`/`PredictionLayer`/`FM`/`InteractingLayer` 等）。
- 范式样例：`deepctr/models/wdl.py`（单任务）、`deepctr/models/sequence/din.py`/`bst.py`（序列）、`deepctr/models/multitask/mmoe.py`（多任务）。
- 测试：`tests/utils.py::check_model`/`get_test_data`、`tests/utils_mtl.py::check_mtl_model`/`get_mtl_test_data`。
- benchmark：`benchmarks/registry.py`、`benchmarks/benchmark.py::run_single/run_multitask/run_sequence`、`benchmarks/common.py`、`benchmarks/metrics.py`。
- feature 定义：`deepctr/feature_column.py`。

---

## 验证（端到端如何测）

```bash
export CUDA_VISIBLE_DEVICES="" TF_USE_LEGACY_KERAS=1

# 接线完整性
python -m benchmarks.onboard audit

# 新模型全链路（以 FinalMLP 为例）
python -m benchmarks.onboard discover --list
python -m benchmarks.onboard scaffold FinalMLP --category single
#   ↳ 人/LLM 补全 deepctr/models/finalmlp.py 的核心交互
python -m benchmarks.onboard verify FinalMLP
python -m benchmarks.onboard docs FinalMLP

# 回归
python -m pytest tests/benchmark_test.py -q
python -m benchmarks.benchmark --track all --quick
```

通过标准：`audit` 全绿；新模型可 `from deepctr.models import X`、单测 save-load 往返通过、benchmark 跑出 AUC、文档各处出现新模型条目；既有测试不回归。

---

## 不在本次范围

- 不接 CI 闸门（独立 CLI 取向）。
- `discover` 的全自动联网研究依赖外部检索，首批候选以人工/研究填充为准。
- 不改 `deepctr/estimator/` 旧 Estimator 路径。
