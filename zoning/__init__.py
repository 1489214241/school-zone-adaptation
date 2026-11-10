"""小区—学校归属补全。

六步，各一个模块：
    data      读 data/ 下导出好的数据
    rules     第一、二步：政策文字的原子观察 → 每个（小区，学校）判入 / 排除 / 不知道
    evidence  第三步：把规则结果和地理信息整理成模型的两类输入
    model     第四步：参照检索网络（参照一路、学校一路，线性项＋小网络，一起做 softmax）
    train     第五步：训练题、权重、线性起点、训练循环
    predict   第六步：规则判出的照原判，其余取概率最高的学校
    evaluate  计分与空间块重抽区间
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[_k] = '1'
import torch  # noqa: E402

# 结果可复现：与项目里旧程序相同的设置
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
