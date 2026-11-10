"""可选的第七步：限定范围的证据修正（默认不用；scripts/correct.py 调用）。

已训练好的模型冻结不动（各种子平均的学校概率 P）。对带某类专门证据的（小区, 学校）加一个修正分：
    最终分 = log P(小区, 学校) + 修正(小区, 学校)，再在各学校上 softmax。
限定范围（scoped=True）：修正只在带证据的（小区, 学校）上非零；一个小区若没有任何学校带证据，输出直接拷贝原概率，逐位不变。
不限定（scoped=False，对照）：同一个修正头作用于所有（小区, 学校）。

现在只有一类证据：名字（小区名里含有该校政策文字中的楼盘/单元名；data 里 name_match 的前三列：只对上这一所、对上多所、对上的最长字数）。
第四列"别的学校对上、这所没对上"是小区层面的标志，不算这所学校带证据。

训练题：其余七区规则没判出、有官方答案的小区（限定范围时只有带证据的小区影响损失）。它们的原模型概率来自嵌套模型：
对目标区 t、来源区 o，另训一个以 o 为目标、来源不含 t 的模型（o 自己的官方答案也不用），所以目标区的官方答案不进入修正的任何输入。
"""
import numpy as np
import torch
from torch import nn

FEATURES = ['只对上这一所', '对上多所', '对上最长字数/10', 'log原概率/10', '是原模型第一名', 'log原第一名概率/10', '1/对上的学校数', '到校址公里数/5（最多2）']


def name_mask(dist):
    """[n, 学校数]：这所学校在这个小区上带名字证据。"""
    return (dist.name_match[..., :3] != 0).any(-1)


def features(dist, P, mask):
    """[n, 学校数, 8]。P：原模型概率（冻结）。"""
    nm = dist.name_match[..., :3]; lp = np.log(np.clip(P, 1e-6, None)) / 10
    top = (P == P.max(1, keepdims=True)).astype(np.float32); lpt = np.broadcast_to(lp.max(1, keepdims=True), lp.shape)
    k = mask.sum(1, keepdims=True); invk = np.broadcast_to(np.where(k > 0, 1 / np.maximum(k, 1), 0), lp.shape)
    dkm = np.minimum(np.linalg.norm(dist.school_xy[None] - dist.coords[:, None], axis=-1) / 1000, 10) / 5
    Z = np.concatenate([nm, lp[..., None], top[..., None], lpt[..., None], invk[..., None], dkm[..., None]], -1).astype(np.float32)
    assert Z.shape[-1] == len(FEATURES) and np.isfinite(Z).all()
    return Z


class Correction(nn.Module):
    """修正分 ＝ 线性项 ＋ 小网络（8→16→1）。线性项和小网络最后一层都从 0 起步，训练开始时输出等于原模型。"""
    def __init__(self, nf, seed, width=16):
        super().__init__(); torch.manual_seed(seed)
        self.lin = nn.Linear(nf, 1); self.net = nn.Sequential(nn.Linear(nf, width), nn.SiLU(), nn.Linear(width, 1))
        for m in [self.lin, self.net[-1]]: nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, logp, Z, mask):
        r = (self.lin(Z) + self.net(Z)).squeeze(-1)
        return torch.log_softmax(logp + r * mask, -1)

    def penalty(self):
        return sum((p ** 2).sum() for n_, p in self.named_parameters() if n_.endswith('weight'))


def _tensors(dist, P, mask, ids, labels, scoped):
    q = np.asarray(ids, int); m = mask[q] if scoped else np.ones_like(mask[q])
    return dict(logp=torch.from_numpy(np.log(np.clip(P[q], 1e-30, None)).astype(np.float32)), Z=torch.from_numpy(features(dist, P, mask)[q]),
                mask=torch.from_numpy(m.astype(np.float32)), y=torch.tensor([labels[i] for i in q], dtype=torch.long), groups=[(dist.name, dist.groups[i]) for i in q])


def fit(train_sets, cfg, seed, scoped):
    """train_sets：[(District, 冻结概率 P, 证据 mask, {小区: 学校})]，每个来源区一组。cfg：correction 段。"""
    bs = []
    for dist, P, mask, lab in train_sets:
        ids = sorted(lab)
        if scoped: ids = [q for q in ids if mask[q].any()]           # 不带证据的小区修正为 0，对损失没有影响
        if ids: bs.append(_tensors(dist, P, mask, ids, lab, scoped))
    allg = [g for b in bs for g in b['groups']]; uniq, inv, cnt = np.unique(np.array([f'{a}|{b}' for a, b in allg]), return_inverse=True, return_counts=True)
    w = 1 / cnt[inv]; w = w / w.sum(); st = 0
    for b in bs: n = len(b['groups']); b['w'] = torch.tensor(w[st:st + n], dtype=torch.float32); st += n
    model = Correction(len(FEATURES), seed, cfg['width']); opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    for _ in range(cfg['steps']):
        opt.zero_grad(); loss = cfg['reg'] * model.penalty()
        for b in bs:
            lp = model(b['logp'], b['Z'], b['mask']); loss = loss - (lp[torch.arange(len(b['y'])), b['y']] * b['w']).sum()
        assert torch.isfinite(loss); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
    return model


def apply(model, dist, P, mask, scoped):
    """返回修正后的概率 [n, 学校数]；限定范围时没有证据的小区直接拷贝原概率。"""
    rows = np.flatnonzero(mask.any(1)) if scoped else np.arange(dist.n); out = P.copy()
    if len(rows):
        t = _tensors(dist, P, mask, rows, {int(q): 0 for q in rows}, scoped)
        with torch.no_grad(): out[rows] = model(t['logp'], t['Z'], t['mask']).exp().numpy()
    return out
