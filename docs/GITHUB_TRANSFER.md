# GitHub 数据下载与结果回传

用户已于 2026-11-10 明确授权将本轮所需历史基线及服务器实验结果公开上传本仓库。无需用户手动传数据。

## 下载历史基线

仓库 `data/` 随 clone 下载。历史基线另放 Release，避免扩大 Git 历史：

- Release：https://github.com/1489214241/school-zone-adaptation/releases/tag/baseline-pre-v1-20261110
- 完整 32 份：https://github.com/1489214241/school-zone-adaptation/releases/download/baseline-pre-v1-20261110/historical_baselines_32.zip
- 完整包 SHA-256：https://github.com/1489214241/school-zone-adaptation/releases/download/baseline-pre-v1-20261110/historical_baselines_32.zip.sha256
- 先导单区：https://github.com/1489214241/school-zone-adaptation/releases/download/baseline-pre-v1-20261110/baseline_pilot_k1_s1_yuhua.zip

先验证外部 SHA-256，再解压到仓库根目录，得到 `outputs/main_pre_v1/{k0_s1,k1_s1,k1_s2,k1_s3}/probs/<区>.npz`。包内另有逐文件哈希、形状和种子清单。这是历史 BASE 比较基线，不是原机已完成的所有试验结果。不要覆盖新环境复跑概率；使用独立目录存放复跑结果。

## 服务器结果上传（已获用户授权）

1. 检查服务器自己的 GitHub 写入凭据（例如 `gh auth status`）。公有仓库可匿名下载，上传需要该账号对仓库的写权限。不得复制原机令牌、把令牌写进文件或打印凭据。若没有权限，明确说明需用户在服务器登录；继续不依赖上传的已授权计算。
2. 续跑补丁、任务清单、环境版本、先导核对与最终报告提交新的 `server-results/<时间戳>` 分支。不强推，不直接覆盖 main，不将新的训练输出伪装为原机历史结果。
3. 概率结果按原目录结构打成 ZIP，包含逐区配置、种子、完成清单、结果 SHA-256、耗时、峰值显存、实际代码 Git commit 和基线数值差异记录。可以按组分包，避免单包过大。仅上传本实验相关文件，日志先去除令牌、个人目录等无关信息。
4. 生成每个 ZIP 的 `.sha256`。创建唯一 Release 标签，例如 `server-results-YYYYMMDD-HHMMSS`，目标为服务器结果分支的实际 commit，上传 ZIP 和校验文件。不使用覆盖旧资产的参数，不删除已有 Release。
5. 上传后通过 GitHub API 核对资产大小，下载并核对 SHA-256，报告 Release 链接、结果分支、commit、完成/失败任务数和仍需原机处理事项。
6. 若任务未全部完成，明确标为 partial，完整后另建发布，不把部分结果标成最终结论。原机下载新增概率，与保留的历史结果合并，统一按冻结方案评分。

## 运行边界

先解决代理与 GPU 设备访问，核对数据与基线，再续跑。网络失败不能假定下载完成，CUDA 无设备不能靠安装包解决。用户已在服务器指令中授权训练；本文的结果上传授权不改变数据、训练目标、阈值、精度设置或评价标准。
