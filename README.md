# 学区归属模型：按区调节与均衡训练实验

这是用于在另一台双 RTX 5090 机器上接续实验的公有仓库，保存 2026-11-10 停机时的代码与输入快照。它不是已确认的模型改进发布。输入数据约 29 MiB，无历史概率输出、Conda 环境或大型中间张量。

## 在服务器上开始

```bash
git clone https://github.com/1489214241/school-zone-adaptation.git
cd school-zone-adaptation
```

公有仓库可以匿名克隆和拉取，无需服务器 GitHub 登录。只有推送修改才需要另行配置权限；不要复制本机令牌或把令牌写入仓库。

让服务器上的 Codex 先读 `AGENTS.md`、`HANDOFF.md` 和 `docs/EXPERIMENT_PLAN.md`。本仓库不会自动启动训练，无 CI、自动任务或后台运行器。

## 环境

建议独立 Python 3.11 环境。先根据目标机驱动、系统和 RTX 5090 支持情况安装适配的 GPU 版 PyTorch，再安装 `requirements-data.txt` 中的数据处理依赖。保存实际 Python、PyTorch、CUDA、驱动和 GPU 型号版本。

`experiment_state/source_environment.json` 是原机实测版本；原机 `torch 2.6.0+cu118` 不能作为 5090 安装指令。原代码关闭 TF32、启用确定性计算，不应为提速而未经验证地改变实验条件。

## 内容

- `zoning/`、`scripts/`：原有实现，未改训练目标或算法。
- `data/`：可独立加载的数据、政策观察、道路和社区证据。
- `configs/`：原模型配置；`experiment_configs/`：本轮四组冻结配置。
- `docs/EXPERIMENT_PLAN.md`：开跑前的比较与判定标准。
- `experiment_state/completed_districts.json`：已在原机落盘的区及 SHA-256；对应概率留在原机。
- `SOURCE_MANIFEST.json`：交付文件哈希。

原 `scripts/export_data.py`、`scripts/verify.py` 的旧程序对照功能依赖原工作区，不适合在此独立仓库直接执行。正常数据加载和本轮训练使用仓库内 `data/`。

## 两张卡

当前代码使用默认可见 CUDA 设备，不会自动跨两卡训练。适合为每卡分配一个独立进程、互不重叠的任务。例如 Linux 的 `CUDA_VISIBLE_DEVICES=0` 和 `CUDA_VISIBLE_DEVICES=1` 各启动一个任务；各进程看到的本地设备都是 `cuda:0`。必须先单进程测量耗时和峰值显存，再决定调度。

## 结果与同步

首次克隆后用 `git pull --ff-only` 获取更新。`outputs/` 被忽略，不随代码推送。用户已授权通过 GitHub 回传本轮结果。历史基线从 Release 下载，新结果上传独立 Release；具体流程见 `docs/GITHUB_TRANSFER.md`。服务器仍缺少原机已完成的试验概率，完整合并评分在原机执行。
