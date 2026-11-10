"""第四步：参照检索网络。

对一个待判小区，候选是附近的规则已知小区（参照）和各所学校：
    参照一路：每个参照一个分 ＝ 线性项 ＋ 小网络（32→16→1）；输入是这一行证据、参照所属学校那一行证据、该校的周边统计
    学校一路：每所学校一个分 ＝ 线性项 ＋ 小网络；输入是这一行证据和该校的周边统计
所有候选一起做 softmax，再按"候选属于哪所学校"加总，得到每所学校的概率。
周边统计（每所学校 6 个数）：500/1000 米内该校参照的份额、数量，跨路份额，有没有参照。
线性项的起点由 LinearRetrieval 先拟合（train.warm_start）。
"""
import numpy as np
import torch
from torch import nn
from . import DEVICE


def mlp(dim, width=32):
    net = nn.Sequential(nn.Linear(dim, width), nn.SiLU(), nn.Linear(width, width // 2), nn.SiLU(), nn.Linear(width // 2, 1))
    nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)   # 小网络从 0 输出起步，开始时等于线性模型
    return net


def context(pair, anchor, vt, gb, mem, ref_schools, dist_col, crossing_cols, np_):
    """周边统计（每所学校 6 个数）和两路小网络的输入。vt：[小区, 参照] 哪些参照可用；gb：参照的楼盘修正；mem：[参照, 学校]。"""
    v = vt.float()
    weights = torch.exp(gb)[None] * v
    dist = pair[:, :, dist_col] * 300
    w500 = weights * torch.exp(-dist / 500); w1000 = weights * torch.exp(-dist / 1000)
    m500 = w500 @ mem; m1000 = w1000 @ mem
    total500 = m500.sum(1, keepdim=True).clamp_min(1e-12); total1000 = m1000.sum(1, keepdim=True).clamp_min(1e-12)
    cross = (pair[:, :, crossing_cols].sum(-1) > 0).float(); crossmass = (w500 * cross) @ mem
    ctx = torch.stack([m500 / total500, m1000 / total1000, torch.log1p(m500) / 4, torch.log1p(m1000) / 4, crossmass / m500.clamp_min(1e-12), (v @ mem > 0).float()], dim=-1)
    px = torch.cat([pair, anchor[:, ref_schools], ctx[:, ref_schools]], dim=-1); ax = torch.cat([anchor, ctx], dim=-1)
    px = px.clone(); ax = ax.clone(); px[:, :, 0] /= 10; px[:, :, np_] /= 10; ax[:, :, 0] /= 10   # 距离列缩小，给小网络用
    return px, ax


def batch(x, ids, labels=None):
    """把一组小区整理成模型输入（在 CPU 上算，最后搬到 DEVICE）。labels：{小区: 学校}，覆盖默认答案（别区的官方答案）。"""
    q = np.asarray(ids, int); valid = ~x.self_mask[q]
    _, inv, cnt = np.unique(x.groups[q], return_inverse=True, return_counts=True); w = 1 / cnt[inv]; w = w / w.sum()   # 同一楼盘平分
    y = x.y.copy()
    for k, s in (labels or {}).items(): y[k] = s
    pair = torch.from_numpy(x.pair[q]); anchor = torch.from_numpy(x.anchor[q]); vt = torch.from_numpy(valid)
    meta = (x.pair_blocks['distance'].start, x.crossing_cols, x.np_)
    px, ax = context(pair, anchor, vt, torch.from_numpy(x.group_bias), x.membership[:len(x.ref_sites)], x.ref_schools, *meta)
    t = dict(pair=pair, anchor=anchor, valid=vt, y=torch.tensor(y[q], dtype=torch.long), weight=torch.tensor(w, dtype=torch.float32), px=px, ax=ax)
    t = {k: v_.to(DEVICE) for k, v_ in t.items()}
    t.update(ids=q, groups=x.groups[q], mem=x.membership.to(DEVICE), gb=torch.from_numpy(x.group_bias).to(DEVICE))
    t.update(rs=torch.from_numpy(x.ref_schools.astype(np.int64)).to(DEVICE), meta=meta)       # 试验开关用（adapt.masked 重算周边统计）
    return t


def _school_mask(model, al, valid):
    """消融开关（设置 model.paths = ["pair"]）：去掉学校一路，学校不再当候选；一个可用参照都没有的小区仍保留学校一路。默认不屏蔽。"""
    if not getattr(model, 'school_off', False): return al
    return al.masked_fill(valid.any(1, keepdim=True), -torch.inf)


class LinearRetrieval(nn.Module):
    """只有线性项的同一结构，用来给 RetrievalScorer 定起点（越近越好：距离列初值 −1）。在 CPU 上算。"""
    def __init__(self, x):
        super().__init__(); self.x = x
        self.p = nn.Parameter(torch.zeros(x.np_)); self.a = nn.Parameter(torch.zeros(x.na)); self.offset = nn.Parameter(torch.tensor(-2.))
        with torch.no_grad(): self.p[0] = -1; self.a[0] = -1
        self.initial = torch.cat([self.p.detach().clone(), self.a.detach().clone(), self.offset.detach()[None]])

    def forward(self, t):
        pl = (t['pair'] @ self.p) + torch.from_numpy(self.x.group_bias)[None]
        pl = pl.masked_fill(~t['valid'], -torch.inf); al = _school_mask(self, t['anchor'] @ self.a + self.offset, t['valid'])
        return torch.softmax(torch.cat([pl, al], 1), 1) @ self.x.membership

    def penalty(self):
        return (torch.cat([self.p, self.a, self.offset[None]]) - self.initial).square().sum()


class RetrievalScorer(nn.Module):
    def __init__(self, np_, na, start, seed, width=32):
        super().__init__(); torch.manual_seed(seed); self.np = np_; self.na = na
        self.p = nn.Parameter(start.p.detach().clone()); self.a = nn.Parameter(start.a.detach().clone()); self.offset = nn.Parameter(start.offset.detach().clone())
        self.pc = nn.Parameter(torch.zeros(na + 6)); self.ac = nn.Parameter(torch.zeros(6))      # 参照所属学校的证据、周边统计的线性项
        self.pnet = mlp(np_ + na + 6, width); self.anet = mlp(na + 6, width)
        self.register_buffer('start', torch.cat([self.p.detach().clone(), self.a.detach().clone(), self.offset.detach()[None]]))
        self.cond = False

    def enable_condition(self, pair_blocks, anchor_blocks, zdim):
        """试验开关 model.condition：用区概况 z 调节各证据块的增益 g = 1 + U·z（U 从 0 起步，起点与不调节时相同）。
        参照一路：各证据块、参照所属学校的证据、周边统计各一个增益；学校一路：各证据块、周边统计各一个增益；学校一路偏置另加 u·z。"""
        def colmap(blocks, width, extra):
            m = torch.full((width + sum(extra),), -1, dtype=torch.long)
            for i, sl in enumerate(blocks.values()): m[sl] = i
            st = width
            for j, e in enumerate(extra): m[st:st + e] = len(blocks) + j; st += e
            assert (m >= 0).all(); return m
        self.register_buffer('col_p', colmap(pair_blocks, self.np, [self.na, 6])); self.register_buffer('col_a', colmap(anchor_blocks, self.na, [6]))
        self.U_p = nn.Parameter(torch.zeros(len(pair_blocks) + 2, zdim)); self.U_a = nn.Parameter(torch.zeros(len(anchor_blocks) + 1, zdim)); self.u_off = nn.Parameter(torch.zeros(zdim))
        self.cond = True

    def forward(self, t):
        if self.cond: return self._forward_conditioned(t)
        pl = t['pair'] @ self.p + t['gb'][None] + t['px'][:, :, self.np:] @ self.pc + self.pnet(t['px']).squeeze(-1)    # 参照一路
        al = t['anchor'] @ self.a + self.offset + t['ax'][:, :, self.na:] @ self.ac + self.anet(t['ax']).squeeze(-1)     # 学校一路
        pl = pl.masked_fill(~t['valid'], -torch.inf); al = _school_mask(self, al, t['valid'])
        return torch.softmax(torch.cat([pl, al], 1), 1) @ t['mem']                                                      # 一起 softmax，按学校加总

    def _forward_conditioned(self, t):
        z = t['z']; gp = 1 + (self.U_p @ z)[self.col_p]; ga = 1 + (self.U_a @ z)[self.col_a]
        pair = t['pair'] * gp[:self.np]; px = t['px'] * gp; anchor = t['anchor'] * ga[:self.na]; ax = t['ax'] * ga
        pl = pair @ self.p + t['gb'][None] + px[:, :, self.np:] @ self.pc + self.pnet(px).squeeze(-1)
        al = anchor @ self.a + self.offset + self.u_off @ z + ax[:, :, self.na:] @ self.ac + self.anet(ax).squeeze(-1)
        pl = pl.masked_fill(~t['valid'], -torch.inf); al = _school_mask(self, al, t['valid'])
        return torch.softmax(torch.cat([pl, al], 1), 1) @ t['mem']

    def regularization(self, reg):
        diff = torch.cat([self.p, self.a, self.offset[None]]) - self.start
        params = [p for net in [self.pnet, self.anet] for p in net.parameters()]
        out = .01 * diff.square().sum() + reg * (self.pc.square().sum() + self.ac.square().sum()) + reg * sum(p.square().mean() for p in params)
        if self.cond: out = out + reg * (self.U_p.square().sum() + self.U_a.square().sum() + self.u_off.square().sum())
        return out
