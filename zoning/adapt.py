"""按区调整（试验开关，默认都不启用；方案：科研推进/按区调整与均衡训练_20261110/方案.md）。

区概况 z：7 个数，只用该区的已知小区（规则判出＋核验参照）和政策证据，不用官方答案：
    1 有参照的学校占比            2 是参照的楼盘组占全区楼盘组的比例
    3 已知小区之间"抄最近的别组参照"的一致率（不足 5 个记 0.5）
    4 其中最近参照隔主干/次干/铁路河流时的一致率     5 不隔时的一致率（不足 5 个时取第 3 项）
    6 政策里有四至状态的学校占比   7 名字与某校政策名单部分对上的小区占比
各减 0.5。model.condition = true 时，模型用 z 调节每个证据块的增益（model.RetrievalScorer）。
train.reference_dropout = r 时，训练每步每区随机藏掉一部分参照楼盘（比例 ρ~U(0, r)），z 按剩下的重算；预测时不藏。
"""
import numpy as np
import torch
from .model import context

ZDIM = 7


class Support:
    def __init__(self, x):
        sup = x.supervised; self.n_ref = len(x.ref_sites); self.S = x.school_n; ps = x.pair[sup]
        self.D = torch.from_numpy(ps[:, :, x.pair_blocks['distance'].start].copy())                 # [已知小区, 参照] 距离
        cr = ps[:, :, x.crossing_cols]
        self.cross = torch.from_numpy(cr[..., [0, 1, 4]].sum(-1) > 0) if cr.shape[-1] == 5 else torch.zeros(self.D.shape, dtype=torch.bool)
        self.own = torch.from_numpy(x.self_mask[sup].copy()); self.y = torch.from_numpy(x.y[sup].astype(np.int64)); self.rs = torch.from_numpy(x.ref_schools.astype(np.int64))
        rg = x.groups[x.ref_sites]; _, inv = np.unique(rg, return_inverse=True)
        self.ginv = torch.from_numpy(inv.astype(np.int64)); self.ng = int(inv.max()) + 1 if len(inv) else 0; self.total_groups = len(np.unique(x.groups))
        z6 = z7 = 0.
        if 'policy' in x.anchor_blocks:
            pol = x.anchor[:, :, x.anchor_blocks['policy']]; z6 = float((pol[..., [1, 2, 4, 5, 7, 8, 10, 11]].sum(-1) > 0).any(0).mean())
        if 'name' in x.anchor_blocks:
            unknown = x.y < 0; z7 = float((x.anchor[:, :, x.anchor_blocks['name']].sum((1, 2)) > 0)[unknown].mean()) if unknown.any() else 0.
        self.static = (z6, z7)

    def descriptor(self, keep=None):
        """keep：[参照] 哪些参照还在（None＝全部）。返回 [7]。"""
        z3 = z4 = z5 = .5; z1 = z2 = 0.
        if self.n_ref:
            keep = torch.ones(self.n_ref, dtype=torch.bool) if keep is None else keep
            known = (self.own & keep[None]).any(1) | ~self.own.any(1); ok = ~self.own & keep[None]; has = ok.any(1) & known
            j = self.D.masked_fill(~ok, torch.inf).argmin(1); agree = (self.rs[j] == self.y).float(); cr = self.cross[torch.arange(len(j)), j]
            rate = lambda m, fb: float(agree[m].mean()) if int(m.sum()) >= 5 else fb
            z3 = rate(has, .5); z4 = rate(has & cr, z3); z5 = rate(has & ~cr, z3)
            z1 = float((torch.bincount(self.rs[keep], minlength=self.S) > 0).float().mean()); z2 = len(torch.unique(self.ginv[keep])) / self.total_groups
        return torch.tensor([z1, z2, z3, z4, z5, *self.static], dtype=torch.float32) - .5

    def sample_keep(self, gen, max_rate):
        """随机藏掉一部分参照楼盘：先抽比例 ρ~U(0, max_rate)，再逐楼盘组独立决定。"""
        rho = float(torch.rand(1, generator=gen)) * max_rate
        return (torch.rand(self.ng, generator=gen) >= rho)[self.ginv] if self.ng else torch.zeros(0, dtype=torch.bool)


def support(x):
    if '_support' not in x.__dict__: x.__dict__['_support'] = Support(x)
    return x.__dict__['_support']


def masked(t, keep):
    """把一批训练题里被藏掉的参照去掉，重算周边统计和两路网络输入。"""
    valid = t['valid'] & keep.to(t['valid'].device)[None]
    px, ax = context(t['pair'], t['anchor'], valid, t['gb'], t['mem'][:len(t['gb'])], t['rs'], *t['meta'])
    return dict(t, valid=valid, px=px, ax=ax)
