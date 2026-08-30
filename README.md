# VeriGraph3D（验图三维智能体）

VeriGraph3D 是一个面向研究的多模态 3D 编辑智能体原型。它用可执行动态场景图连接任务理解、规划、执行和验证，并把视觉判断与 3D 引擎确定性状态结合起来。当前版本优先实现项目书中的 MVP：移动、旋转、缩放、放置、材质颜色、相机、几何验证及动作回滚。

## 快速运行

要求 Python 3.10+，核心包没有第三方运行时依赖。

```powershell
python -m pip install -e .
python -m verigraph3d.cli
pytest
```

真实参考图验证需要安装视觉依赖：

```powershell
python -m pip install -e ".[visual]"
```

批量运行项目书中的六组消融实验：

```powershell
verigraph3d-experiment examples/tasks.json --output experiment_results.json
```

论文规模的可复现 MVP 扩充集包含 64 个任务、10 个类别和 2 个复用场景：

```powershell
python tools/generate_expanded_tasks.py --output examples/tasks_expanded.json
verigraph3d-experiment examples/tasks_expanded.json --active-view `
  --output artifacts/expanded_ablation_results.json
```

类别覆盖移动、旋转、缩放、支撑面放置、材质颜色、相机、灯光、对象创建、授权删除和复合长序列。数据协议支持显式安全语义动作，因此创建/删除与多动作任务不需要生成任意 Blender Python。扩充集还包含 4 个确定性故障注入任务和 5 个主动观察任务。

从中派生的真实 Blender 分层子集包含 24 项任务，仍覆盖上述 10 类能力。每项任务都从同一个只读 ABWS 基准场景开始，执行结果保存到独立目录：

```powershell
python tools/generate_blender_tasks.py --output examples/blender_tasks_expanded.json
verigraph3d-abws-experiment artifacts/abws_blender24_fixture.blend `
  examples/blender_tasks_expanded.json `
  --output-directory artifacts/blender24_full `
  --blender-executable E:\Blender\blender.exe
```

批处理报告记录源文件 SHA-256、每项任务的最终 revision、动作数、修复/回滚次数、耗时、输出 `.blend` 和按类别汇总。当前固定场景实验为 24/24 成功，共执行 26 个真实 Blender 原子动作；源场景执行前后 SHA-256 一致。

追加主动视角对照：

```powershell
verigraph3d-experiment examples/tasks.json --active-view --output experiment_results.json
```

命令会同时生成完整 JSON、UTF-8 CSV 和 Markdown 论文表格。结果包含逐任务轨迹、任务成功率及 Wilson 95% 置信区间、约束满足率、误接受率、无效动作率、严重物理违规率、失败归因准确率、局部修复成功率、动作/修复/回滚/观察次数和运行成本，并给出 Full 与各消融组的配对 McNemar 精确检验。真实成功率由不参与消融的 oracle 验证器独立计算，避免纯视觉组把内部结构错误误报为成功。数据集使用带版本号的 JSON 格式，可在同一文件中定义复用场景、多项任务及可复现故障注入规则。

也可以保存完整实验轨迹：

```powershell
python -m verigraph3d.cli --instruction "把Cup放到Table上，并把Cup改成红色" --output run.json
```

## 研究阶段与代码子问题

| 阶段 | 可独立验证的子问题 | 主要代码 |
|---|---|---|
| 1 基线与任务集 | 结构化任务、批量消融和统一运行指标 | `examples/tasks.json`, `dataset.py`, `experiments.py` |
| 2 确定性场景状态 | 读取状态、计算关系、检测穿模/悬空 | `state.py`, `backends.py` |
| 3 动态场景图 | 事实溯源、目标、动作历史、增量重建 | `graph.py` |
| 4 几何约束求解 | 语义动作参数化、碰撞规避、执行预检 | `solver.py`, `planning.py` |
| 5 混合验证 | 硬约束优先、视觉属性验证、统一报告 | `verification.py` |
| 6 失败恢复 | 动作级归因、局部修复、快照回滚 | `recovery.py`, `execution.py` |
| 7 主动视角 | 候选视角、信息增益减观察成本 | `active_view.py` |
| 8 完整实验 | 轨迹、成本、测试与可复现入口 | `agent.py`, `tests/` |

## 核心闭环

`VeriGraph3DAgent.run()` 执行以下流程：读取真实状态 → 解析目标 → 生成语义动作 → 几何求解 → 执行前检查 → 保存快照并执行 → 确定性/视觉混合验证 → 失败归因与局部修复 → 更新动态图。

所有编辑通过动作白名单执行，不接受任意 Blender Python 代码。`MemoryBackend` 用于无 Blender 的算法实验；在 Blender 的脚本环境中可换用 `BlenderBackend`。后续接入 VLM 时，只需实现 `VisualVerifier` 协议，不影响几何与执行模块。

## 当前边界

- AABB 关系和碰撞检测适用于 MVP 简单刚体，复杂网格应接入 BVH/物理引擎窄相位检测。
- 内置文本解析器是可复现基线，只覆盖少量中英文模板；真实多模态理解应作为独立适配器接入。
- 内置视觉验证器检查结构化视觉属性；参考图相似度需要接入 VLM 或视觉编码器。
- Blender/ABWS 原子后端已覆盖白名单图元创建、授权删除、基础变换、相机参数与朝向、灯光参数、材质基础色和版本化 `.blend` 检查点；外部资产导入及复杂材质节点仍是下一迭代内容。

## 扩展接口

- `SceneBackend`：替换 Blender 或仿真环境。
- `VisualVerifier`：接入任意 VLM/视觉模型。
- `GoalSpec` / `Constraint`：作为规划、求解和验证共享契约。
- `SemanticAction`：限定安全、可审计的动作空间。

## 接入 AgentBlender World State

项目提供 `AgentBlenderStateMapper` 和 `AgentBlenderWorldStateBackend`。推荐边界是：ABWS 负责稳定对象 ID、场景 revision、事务日志和权威状态；VeriGraph3D 负责多模态目标、动态图规划、视觉—几何混合验证与恢复决策。适配层不会复制 ABWS 源码，也不会形成两个相互竞争的世界真值。

当 ABWS 已安装且在 Blender 内提供实时提取函数时：

```python
from verigraph3d.agentblender import AgentBlenderInstalledReader, AgentBlenderWorldStateBackend
from verigraph3d.backends import BlenderBackend

backend = AgentBlenderWorldStateBackend(
    write_backend=BlenderBackend(),
    state_reader=AgentBlenderInstalledReader(extractor=your_abws_extract_function),
)
```

如果 ABWS 通过原子 JSON 状态存储或 API 落盘，可改用 `AgentBlenderJsonReader("outputs/states/scene.json")`。映射器兼容 Pydantic 对象、列表/ID 字典两类集合，保留 `abws_id`、revision、原始对象记录、关系和约束。写操作仍通过现有动作白名单执行，渲染、检查点和主动视角能力会委托给原 Blender 后端。

需要让 ABWS 同时掌管事务提交时，使用 `AgentBlenderTransactionBackend`。它会把求解后的单步动作转换为包含 `base_revision`、稳定对象 ID、read/write set、前置条件、后置条件和回滚策略的事务计划。只有 ABWS 明确返回 `committed/success/accepted` 才会接受结果；拒绝时，failed constraint、property diff 和验证详情会进入 VeriGraph3D 的动作级失败归因。`undo_transaction`/`redo_transaction` 回调用于连接 ABWS 的持久事务存储。不同 ABWS 版本若使用不同 Pydantic 字段，可通过 `plan_builder` 注入精确转换，而无需修改智能体主循环。

`AgentBlenderRuntimeBridge` 可以自动绑定 runtime 常见的 `read_state/get_state/current_state`、`preview_and_commit/execute_plan`、`undo/redo` 接口。若事务函数的参数带 Pydantic 类型注解，桥接器会调用其 `model_validate`，将 JSON 计划转换成真正的 ABWS `EditPlan`。

Blender 后台入口可以直接启用该桥接器：

```powershell
& 'E:\Blender\blender.exe' --background scene.blend `
  --python scripts/run_blender_task.py -- `
  examples/blender_tasks.json blender_place_and_color_001 `
  --agentblender-src 'D:\AgentBlender-World-State\src' `
  --agentblender-runtime-factory 'my_abws_bootstrap:create_runtime' `
  --output artifacts/abws_run.json
```

`create_runtime()` 应返回已连接当前 Blender 场景和 ABWS `TransactionStore` 的 runtime。factory 把上游版本特定的初始化限制在一个小文件中，VeriGraph3D 的规划、验证和恢复代码无需依赖 ABWS 内部类名。

本仓库现已在 `external/AgentBlender-World-State` 中包含经过兼容修改的 ABWS 源码。对真实 Blender 场景，推荐先使用已经完成联合验证的世界状态模式：

```powershell
& 'E:\Blender\blender.exe' --background scene.blend `
  --python scripts/run_blender_task.py -- `
  examples/abws_blender_tasks.json abws_move_chair_001 `
  --agentblender-src external/AgentBlender-World-State/src `
  --agentblender-world-state `
  --output artifacts/abws_bridge_run.json
```

该模式使用 ABWS 的实时提取器作为权威读状态，以稳定 `abws_id` 追踪对象，以 BlenderBackend 的固定动作白名单修改当前场景，并在动作后递增 ABWS revision、重新提取和混合验证。ABWS `BOTTOM_CENTER` 与 Blender 中心原点之间的转换、缩放维度重复计算和 `PLACE_ON` 半穿模问题也已在上游适配代码中修正。

`--agentblender-runtime-factory` 适合纯状态事务实验；普通 `SceneRuntime` 只提交内存 `WorldState`，不能被误当作当前 `.blend` 已经修改。完整的候选 `.blend` 原子提升仍应使用 ABWS `BlenderProcess.execute_transaction()`。

宿主端原子任务入口已经封装了上述流程：

```powershell
$env:PYTHONPATH='src'
python -m verigraph3d.abws_atomic_cli `
  artifacts/abws_fixture.blend `
  examples/abws_blender_tasks.json abws_move_chair_001 `
  --abws-src external/AgentBlender-World-State/src `
  --blender-executable 'E:\Blender\blender.exe' `
  --output-blend artifacts/abws_atomic_result.blend `
  --report artifacts/abws_atomic_result.json
```

该入口保证原始 `.blend` 不被覆盖。每个动作产生候选文件，经 ABWS 实际状态 diff、write guard 和后置条件验证后才成为下一修订；任务最终接受时才复制为 `--output-blend`。

上游仓库当前页面未显示开源许可证，因此本项目只提供接口级适配，不直接复制或再分发其代码。正式合并源码前应由仓库作者补充许可证。

## 故障恢复与主动观察实验

`FaultInjectingBackend` 可以按动作 ID 注入一次性执行异常、无操作或 Z 轴偏移，用于可重复地测量失败归因准确率和局部修复成功率。`AgentConfig(use_active_view=True)` 会在任务包含 `uncertainties` 时计算候选视角的信息增益；只有收益高于观察成本时才执行相机观察，并记录在 `observations` 指标和动作轨迹中。

在 Blender 后端中，`visible_from` 使用相机视锥投影和对象中心/包围盒角点的多点射线检测，并把可见率、来源 `engine_ray_cast` 写入场景图。主动视角模块会临时评估候选相机的真实几何可见率，恢复相机后再执行收益最高且超过观察成本的视角。

## Blender 后台运行

在已安装 Blender 的环境中，可以直接运行一个数据集任务：

```powershell
blender --background scene.blend --python scripts/run_blender_task.py -- examples/tasks.json place_and_color_001 --output blender_run.json --checkpoint checkpoints/before.blend
```

当任务提供 `reference_images` 时，智能体会在动作后渲染，并使用像素、颜色直方图、边缘和粗粒度构图四类证据进行真实图像验证。检查点通过 Blender 的 `copy=True` 保存，不替换当前打开的原始场景。

本机 Blender 不在 `PATH` 时，可显式运行真实集成测试：

```powershell
$env:BLENDER_EXECUTABLE='E:\Blender\blender.exe'
pytest -q -m blender
```

## 接入 VLM API

`VLMClient` 是供应商无关接口，目前提供两种实现：官方 Responses 风格的 `ResponsesVLMClient`，以及适合 vLLM、LM Studio 等服务的 `ChatCompletionsVLMClient`。客户端支持本地图片 Data URL、JSON Schema 输出、超时、错误封装和 Token 用量统计。

通过环境变量配置，不要把 API Key 写入代码或任务文件：

```powershell
$env:VERIGRAPH_VLM_PROVIDER='responses'
$env:VERIGRAPH_VLM_BASE_URL='https://api.openai.com/v1'
$env:VERIGRAPH_VLM_MODEL='your-vision-model'
$env:VERIGRAPH_VLM_API_KEY='your-key'
```

在真实 Blender 任务中启用 VLM 任务理解和视觉验证：

```powershell
& 'E:\Blender\blender.exe' --background scene.blend `
  --python scripts/run_blender_task.py -- `
  examples/blender_tasks.json blender_place_and_color_001 `
  --vlm-task-understanding --vlm-visual-verification `
  --output artifacts/vlm_run.json
```

本地兼容服务可以将 `VERIGRAPH_VLM_PROVIDER` 设为 `chat`、将 `BASE_URL` 指向本机服务。若服务不支持 JSON Schema，将 `VERIGRAPH_VLM_STRUCTURED_OUTPUTS=false`，系统会把 Schema 放入提示词并继续严格解析返回 JSON。完整配置见 `examples/vlm.env.example`。

只验证配置、不发送请求：

```powershell
verigraph3d-vlm-check
```

发送一次可能产生费用的文本或图像健康检查：

```powershell
verigraph3d-vlm-check --live
verigraph3d-vlm-check --live --image artifacts/reference.png
```

HTTP 客户端会对超时、429 和 5xx 错误进行指数退避重试。`BudgetedVLMClient` 可限制最大调用次数和输入/输出 Token。启用 VLM 视觉验证时，Blender 入口默认按 40% 确定性图像指标和 60% VLM 语义评分进行融合，并合并两类差异证据。

## 场景图导出

场景图现在包含对象、容器、相机、灯光、材质、约束和动作节点，并支持 `in_front_of`、`attached_to`、`uses_material`、`acts_on` 等执行语义边。可导出 JSON 或 Graphviz DOT：

```powershell
verigraph3d-graph examples/tasks.json place_and_color_001 --format dot --output artifacts/task_graph.dot
verigraph3d-graph examples/tasks.json place_and_color_001 --format json --output artifacts/task_graph.json
```

`ExecutableSceneGraph.diff(previous)` 会分别报告新增、删除和变化的节点与事实，可用于动作级错误定位和论文案例可视化。

实验时请固定任务文件、Python/Blender 版本、随机种子和模型版本，并保存 CLI 输出的完整轨迹。
