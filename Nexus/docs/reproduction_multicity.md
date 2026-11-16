# Nexus 四城验证子集评测

这是在前期 8 个调试场景之后进行的独立子集评测，不代表论文全量指标已复现。

冻结清单为 `artifacts/nuplan_evaluation/multicity128_plan_v2.json`：4 城市各 32 个场景，共 32 段行程，每段行程 4 个间隔至少 15 秒的场景。排除前期调试使用的整段行程。选择仅依据数据库元信息，不使用模型预测结果。候选必须同时满足 nuPlan 官方 `valid_scenes` 首尾过滤规则，以及历史至少 2 秒、未来至少 8 秒的条件。

主设置保留官方 ResetVAugmentor，map/sma/clip 权重均为 1，full_sequence 日程、32 步采样。随机种子标签为 0、1、2，每个场景分别进行行为预测和终点条件行为预测。默认每批 4 个场景。

## 在新终端运行

```bash
source /inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/nexus_repro/cache_env.sh &&
conda activate "$NEXUS_WORKDIR/envs/nexus" &&
export PATH="$CONDA_PREFIX/bin:$PATH" &&
hash -r &&
cd "$NEXUS_WORKDIR/Nexus" &&
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 &&
set -o pipefail &&
python -u scripts/pipeline/run_nuplan_multicity.py \
  --plan artifacts/nuplan_evaluation/multicity128_plan_v2.json \
  --run-dir artifacts/nuplan_evaluation/multicity128_official_v2 \
  --seeds 0 1 2 \
  --batch-size 4 \
  2>&1 | tee -a artifacts/nuplan_evaluation/multicity128_v2_latest.log
```

推理所需文件已在服务器本地，无需为这条命令重新选择代理节点。依赖导入、模型加载、地图读取和 CPU 指标计算期间，GPU 使用率可能暂时降低；无法保证平台不会因其空闲策略回收实例。

`EVAL_PROGRESS` 表示进度，`NUPLAN_MULTICITY_OK` 表示整个任务完成。中断后再次执行相同命令，按保存记录恢复；已经持久化的场景预测不会全部重跑，部分完成的批次可能重算。恢复时不要改变种子、批大小、清单和脚本。

## 输出与统计口径

输出目录为 `artifacts/nuplan_evaluation/multicity128_official_v2`：

- `run.json`：状态、预测文件校验和、随机数设置和实际 GPU 等。
- `config.yaml`、`frozen_plan.json`：运行配置和冻结的场景清单。
- `prediction_*_seed*.pt`：每个场景每个种子的两项任务预测。
- `official_metrics_by_seed.json`：官方实现的逐种子误差、碰撞和地图代理指标。
- `audited_displacement_by_city_seed.json`：按城市、种子汇总的未来半秒采样点位置误差；排除自车、历史和无效帧，终点条件任务另排除每辆车已给定的最后有效点。

官方误差按整秒采样并进行时域平均，审计误差对全部有效未来半秒采样点加权，两者不应混用。官方地图指标是近邻道路点距离代理，碰撞统计也是原实现口径。批量推理时间不能直接当成论文的单场景延迟。

GPU 推理由用户终端启动。v2 评测已完成：128 个场景、384 个场景与种子组合、768 条场景级任务输出。`run.json` 状态为 `complete`，模型参数版本检查通过，训练更新为 0；384 份预测文件的 SHA256 已逐一校验通过。

三个种子均值与样本标准差：普通预测 1–8 秒整秒误差为 9.1939 ± 0.0157 米，给定终点任务 1–7 秒整秒误差为 3.1860 ± 0.0144 米。排除已知终点等条件点的半秒轨迹点审计误差分别约为 8.3238 米和 3.1269 米。两种口径不能混用，也不应把给定终点任务 8 秒处的零误差解释为独立终点预测成功。

报告导出入口为 `export_nuplan_multicity_report.py`，输出至 `/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/report/Nexus四城定量复现结果_2026-10-21`，包含 Markdown、图表、指标 JSON、离线 HTML 和同名完整 ZIP 包。当前完成的是四城子集评测，尚未完成全验证集或论文全部数值对齐。

## 首次启动故障修复

v1 首次运行在特征提取前发生 `KeyError: '0e380cdb0a0c5f74'`，保存预测记录数为 0。原因是 v1 选样只检查了历史和未来时间范围，没有应用 nuPlan 查询中 `row_num >= 3 AND row_num < scene_count - 1` 的首尾场景过滤。该 token 在数据库中存在，但被官方场景构建器排除。

v1 清单和失败目录保留用于追溯，正式运行使用 v2 清单与新输出目录。场景类型也与官方查询统一使用 `MAX(type)`。评测启动时先验证全部 token，再加载模型；若构建器仍遗漏 token，会输出明确的缺失列表。

`check_nuplan_plan_builder.py` 使用真正的 nuPlan 场景构建器进行 CPU 集成验证，结果保存至 `artifacts/nuplan_evaluation/multicity128_plan_v2_builder_check.json`。这项检查验证 token、日志名、时间戳和场景类型，不进行 GPU 推理。
