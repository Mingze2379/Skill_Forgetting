# MATH 固定位置回放与 Shapley 首轮

本实验直接使用普通 SFT，不执行技能记忆抽取与增强。训练仍由 LlamaFactory 完成；生成式评估复用上游 `math.evaluate`。默认配置见 `configs/replay_shapley_pilot.json`。

## 固定的实验定义

- 模型：Qwen2.5-3B-Instruct，LoRA rank 16、alpha 32、dropout 0.05，学习率 `1e-4`。
- 顺序：代数 → 计数与概率 → 几何。各阶段用 512 条唯一训练题、128 个优化步，每阶段重新建立 LoRA/优化器，并合并导出模型。
- 起点：几何阶段结束后的同一个完整模型。八个中级代数分支各自重新建立 LoRA，只保存 adapter，评估时使用共享起点 + `--lora_path`。
- 每分支：512 条题、64 个优化步，单 GPU、batch size 1、梯度累积 8。这是一次完整的数据遍历；数据随机种子和模型随机种子均为 42。
- A=代数，B=计数与概率，C=几何。每簇预留 64 个互不重叠的位置；激活某簇时，在这些位置用固定的旧题替换固定的新题。旧题来自前三阶段实际使用过的 512 题。其他位置保持不变。
- 验证：每域从原生 train 按难度层级留出 100 题。训练/回放排除验证题、四域原生 test 中的重复题和跨域重复题；原生 test 内容单独保留。首轮数据中发现并排除了一条训练/test 重复题。
- 效用：`v(S) = (acc_algebra + acc_counting + acc_geometry + acc_intermediate) / 4`。准确率取值为 0–1；相对空集增益另列。

固定样本数和优化步数不能保证相同的监督 token 数。训练入口对实际执行的每个 micro-batch 统计 `labels[:, 1:] != -100` 的数量，包括重复遍历和截断后的真实标签；统计结果写入 `loss_tokens.json`，最终表同时保留 loss tokens、实际样本访问数和优化步数。没有将估算 token 数当作实际训练预算。

## 数据与命令

在可访问 Hugging Face 的机器下载原生切分。`--parquet` 读取官方的 native-split Parquet 导出，避免旧数据集加载脚本与新版 datasets 的兼容问题。

```bash
export PYTHONPATH="$PWD/src"
python -m skill_forgetting.math.download --source hendrycks_math --parquet
python -m skill_forgetting.math.prepare \
  --run_root runs/math_raw --by_type_dir data/hendrycks_math_splits
python -m skill_forgetting.experiments.prepare \
  --config configs/replay_shapley_pilot.json --raw_root runs/math_raw
```

准备器导出每域普通 SFT、验证原题、原生 test、八份分支 SFT、固定回放题清单和位置清单。训练与推理共用 `math.plain` 中的用户提示；使用相同的 Qwen chat template。

```bash
# 查看顺序训练和一个分支的参数；不加载模型
python -m skill_forgetting.experiments.run \
  --config configs/replay_shapley_pilot.json --dry_run baseline
python -m skill_forgetting.experiments.run \
  --config configs/replay_shapley_pilot.json --dry_run branch --subset AB

# 单次训练入口可以独立指定起点、数据、步数、种子、输出目录
python -m skill_forgetting.training.single_sft \
  --model /path/to/checkpoint --data_dir /path/to/sft \
  --max_steps 64 --seed 42 --output_dir /path/to/adapter
```

## HPC3 执行

项目代码、数据与结果在 `/data/user/msun411/cl/Skill_Forgetting`，自动同步到 Mac。模型、训练环境、完整 checkpoint 和 adapter 放在 `/data/user/msun411/cl-artifacts`，避免大体积权重反复同步回 Mac。配置中的绝对路径可按主机修改；改动预算或路径后应建立新的准备目录。

训练环境固定为 LlamaFactory 0.9.5、Transformers 4.57.6、PEFT 0.18.1、Accelerate 1.11.0、datasets 3.6.0、Math-Verify 0.8.0。HPC3 的训练/评估环境分别使用共享 PyTorch 2.7.1 与 vLLM 0.17.1 的独立 overlay 环境。计算作业启用离线模式。

额外依赖通过[校内软件镜像仓](https://docs.hpc.hkust-gz.edu.cn/docs/hpc3/on-campus-repository)安装，PyPI 地址为 `http://harbor.internal.com:8081/repository/pypi-hkust/simple`。镜像配置写在项目虚拟环境内；共享 PyTorch 和 NVIDIA 驱动由平台提供。GPU 预检作业 820208 已确认 H100 80GB、CUDA 12.8 和 BF16 运算正常。

```bash
cd /data/user/msun411/cl/Skill_Forgetting
export PYTHONPATH="$PWD/src"
PY=/data/user/msun411/cl-artifacts/envs/sft/bin/python

# 仅生成调度脚本
"$PY" -m skill_forgetting.experiments.slurm \
  --config configs/replay_shapley_pilot.json --dry_run

# 确认模型和依赖就绪后提交。已经提交的目录会拒绝重复提交。
"$PY" -m skill_forgetting.experiments.slurm \
  --config configs/replay_shapley_pilot.json

# 小测试正在排队时，可以预先提交依赖链；只有小测试成功才会运行。
"$PY" -m skill_forgetting.experiments.slurm \
  --config configs/replay_shapley_pilot.json --dependency_job SMOKE_JOB_ID
```

调度使用 `acd_u`：顺序基线一个 GPU，成功后启动八个单 GPU 分支，最多同时八张 GPU、64 核 CPU；最后一个 CPU 作业读取完成的分数表。依赖使用 `afterok`；失败的训练不会触发后续实验。作业 ID 保存在 `runs/replay_shapley_pilot/slurm/jobs.json`，日志也在此目录。

完成的训练和评估可以复用缓存；入口检查数据指纹、起点权重、参数、生成预算和评分器。未完成或参数不一致的目录会报错，避免把不同实验的输出拼成同一张表。

## 评分与交付物

每个答案保留上游规范化字符串评分与 Math-Verify 的数学等价评分。主评分接受字符串匹配或数学等价成立；不可解析的答案保留解析状态。生成长度截断、两种评分器不一致的原题/答案/预测和固定种子抽样的 20 条复核样本均写入每个子集的结果目录。

`report` 只接受覆盖完整验证集的八份实际分数；缺行、混用起点、种子、样本数、步数或评分器时拒绝输出结果。

```bash
python -m skill_forgetting.experiments.report --run_root runs/replay_shapley_pilot
```

- `report/eight_subsets.csv`：8 行，包含四域验证准确率、v、相对空集增益、实际 loss tokens 与训练预算。
- `report/shapley.json`：精确三簇 Shapley、各域 Shapley、单簇增益排序、二阶/三阶交互项，以及回放预算 0/64/128/192 下实际得分最高的所有并列子集。
- `report/results.md`：便于查看的结果表与两种排序。
- `results/<subset>/disagreements.jsonl`、`audit_sample.jsonl`：评分分歧与待人工复核样本。
- `baseline_validation/score.json`：进入中级代数之前的四域验证分数，辅助判断起点是否已经学会旧域。

这是单 checkpoint、单 seed 的探索性首轮；排序与交互值不代表统计显著性。原生 test 不参与八分支估值或子集选择。先在验证表上确定固定回放预算与子集，再开展最终 test 报告及第二 checkpoint 实验。

## 验证

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

CPU 测试覆盖原生 test 保留、验证隔离、固定位置干预、回放来自已见数据、随机种子复现、提示一致性、数学等价、预测覆盖检查、精确 Shapley 和 SLURM 资源配置。真实 GPU 训练另行验证；CPU 测试产生的合成分数不会写入实验报告。
