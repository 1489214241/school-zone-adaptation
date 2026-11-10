"""第三步：把规则结果和地理信息整理成模型的两类输入。

候选有两种：附近的规则已知小区（参照），以及各所学校。每个候选一行证据，按"块"组织：

参照一路（小区 × 参照小区）                       学校一路（小区 × 学校）
    distance          距离、方向（5）                 distance          到校址的距离、方向（5）
    named_reference   参照是名单点名的（1）           policy            小区对该校四边的状态、是否被排除（13）
    policy            小区对参照学校四边的状态、排除（13）  within_2km        2 公里内（1）
    agreement         与参照在各边状态相同（4）        community_clause  所在社区被该校整社区点名（4，只雨花）
    disagreement      与参照在各边状态相反（4）        name              小区名与该校政策名单部分对上（4）
    crossing          连线跨主干/次干/三级/支路/铁路河流（5）
    within_2km        2 公里内（1）
    learned_boundary  学出的分界：连线跨的路附近，规则已知小区对是否多为异校（6）
    same_community    与参照同社区 / 不同社区（2，只雨花）
"""
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import shapely
import torch
from shapely import STRtree
from scipy.spatial import cKDTree
from .data import DATA

PAIR_BLOCKS = ['distance', 'named_reference', 'policy', 'agreement', 'disagreement', 'crossing', 'within_2km', 'learned_boundary', 'same_community']
SCHOOL_BLOCKS = ['distance', 'policy', 'within_2km', 'community_clause', 'name']
RANK = {'local': 0, 'tertiary': 1, 'secondary': 2, 'major': 3, 'rail': 4, 'river': 4}


# ---------------- 道路、河流、铁路 ----------------
class Lines:
    def __init__(self, city):
        L = pickle.load(open(DATA / 'lines' / f'{city}.pkl', 'rb')); self.L = L
        self.key = [(l['name'] or 'wid%d' % l['wid']) for l in L]; self.cls = [l['cls'] for l in L]
        self.geoms = np.array([shapely.linestrings(l['xy']) for l in L], dtype=object); self.tree = STRtree(self.geoms); self.by = {}
        for c in ['major', 'secondary', 'tertiary', 'local']:
            self.by[c] = STRtree(self.geoms[[i for i, x in enumerate(self.cls) if x == c]])
        self.by['rr'] = STRtree(self.geoms[[i for i, x in enumerate(self.cls) if x in ('rail', 'river')]])


_lines = {}


def lines(city):
    if city not in _lines: _lines[city] = Lines(city)
    return _lines[city]


def crossing(ls, coords, targets):
    """[n, m, 5] 直线 2 公里内跨各类线；[n, m, 1] 是否在 2 公里内。"""
    n, m = len(coords), len(targets); X = np.zeros((n, m, 5), np.float32); W = np.zeros((n, m, 1), np.float32)
    D = np.linalg.norm(coords[:, None] - targets[None], axis=-1); a, b = np.nonzero((D <= 2000) & (D > 1e-6))
    for st in range(0, len(a), 200000):
        aa, bb = a[st:st + 200000], b[st:st + 200000]; segs = shapely.linestrings(np.stack([coords[aa], targets[bb]], 1))
        for col, c in enumerate(['major', 'secondary', 'tertiary', 'local', 'rr']):
            hit = ls.by[c].query(segs, predicate='crosses')[0]; X[aa[hit], bb[hit], col] = 1
        W[aa, bb, 0] = 1
    return X, W


def _crossed_roads(ls, A, B):
    segs = shapely.linestrings(np.stack([A, B], 1)); si, li = ls.tree.query(segs, predicate='intersects')
    pts = shapely.centroid(shapely.intersection(segs[si], ls.geoms[li])); out = defaultdict(dict)
    for s, l, p in zip(si, li, pts):
        if (ls.cls[l] == 'local' and not ls.L[l]['name']) or p.is_empty: continue
        k = ls.key[l]
        if k not in out[s] or RANK[ls.cls[l]] > RANK[out[s][k][0]]: out[s][k] = (ls.cls[l], p.x, p.y)
    return out


def learned_boundary(ls, coords, groups, known, ref_sites, R=300, MAXLEN=800, LIMIT=1500):
    """学出的分界 [n, nref, 6]：小区→参照连线跨过的每条路，在交点 300 米内找跨过同一条路的规则已知小区对（单校、不同楼盘、不含本小区所在楼盘），
    看其中异校的比例。列：最高异校比例、最高等级路上的异校比例、支持对数、有观测、异校多的路数、同校多的路数。"""
    n = len(coords); lab = np.full(n, -1)
    for q, s in known.items():
        if len(s) == 1: lab[q] = s[0]
    kn = np.flatnonzero(lab >= 0); Dk = np.linalg.norm(coords[kn][:, None] - coords[kn][None], axis=-1)
    a, b = np.nonzero(np.triu(Dk <= MAXLEN, 1)); a, b = kn[a], kn[b]; keep = groups[a] != groups[b]; a, b = a[keep], b[keep]; per = defaultdict(list)
    for s, dd in _crossed_roads(ls, coords[a], coords[b]).items():
        for k, (c, x, y) in dd.items(): per[k].append((a[s], b[s], x, y))
    trees = {}
    for k, v in per.items():
        v = np.array(v, dtype=object); trees[k] = (cKDTree(np.array(v[:, 2:], float)), v[:, 0].astype(int), v[:, 1].astype(int))
    usites, inv = np.unique(ref_sites, return_inverse=True); F = np.zeros((n, len(usites), 6), np.float32)
    D = np.linalg.norm(coords[:, None] - coords[usites][None], axis=-1); qa, qb = np.nonzero(D <= LIMIT)
    for st in range(0, len(qa), 20000):
        aa, bb = qa[st:st + 20000], qb[st:st + 20000]
        for s, dd in _crossed_roads(ls, coords[aa], coords[usites[bb]]).items():
            qg = groups[aa[s]]; best = -1; hc = -1; hcs = 0; tot = 0; hi = 0; lo = 0
            for k, (c, x, y) in dd.items():
                if k not in trees: continue
                t, pa, pb = trees[k]; idx = t.query_ball_point([x, y], R)
                if not idx: continue
                pa_, pb_ = pa[idx], pb[idx]; ok = (groups[pa_] != qg) & (groups[pb_] != qg); pa_, pb_ = pa_[ok], pb_[ok]
                if len(pa_) < 2: continue
                v = float(np.mean(lab[pa_] != lab[pb_])); tot += len(pa_); best = max(best, v)
                if RANK[c] > hc or (RANK[c] == hc and v > hcs): hc = RANK[c]; hcs = v
                hi += v >= 0.67; lo += v < 0.34
            if tot > 0: F[aa[s], bb[s]] = [best, hcs, np.log1p(tot) / 4, 1, hi, lo]
    return F[:, inv]


# ---------------- 基本编码 ----------------
def geometry(delta, scale):
    d = np.linalg.norm(delta, axis=-1)
    return np.stack([np.minimum(d / scale, 50), np.log1p(d / 100), np.clip(delta[..., 0] / 1000, -5, 5), np.clip(delta[..., 1] / 1000, -5, 5), np.exp(-d / 500)], axis=-1).astype('float32')


def policy(st, negative):
    """四边状态各编成"不知道/在外/在内"三格，加上是否被排除，共 13 列。"""
    return np.concatenate([(st[..., None] == np.array([-1, 0, 1])).reshape(*st.shape[:-1], 12), negative[..., None]], axis=-1).astype('float32')


@dataclass
class Inputs:
    district: str
    pair: np.ndarray          # [n, 参照数, Fp]
    anchor: np.ndarray        # [n, 学校数, Fa]
    pair_blocks: dict         # 块名 → 列范围
    anchor_blocks: dict
    ref_sites: np.ndarray     # 每个参照是哪个小区
    ref_schools: np.ndarray   # 每个参照属于哪所学校
    group_bias: np.ndarray    # 同一楼盘有几个参照点，按 -log 个数修正，避免大楼盘占多数
    self_mask: np.ndarray     # [n, 参照数] 同楼盘的参照不能用
    groups: np.ndarray
    y: np.ndarray             # 规则判出的学校（多校取第一所），未判出为 -1
    supervised: np.ndarray    # 本区可当训练题的规则已知小区
    school_n: int
    membership: torch.Tensor = field(repr=False, default=None)   # [参照数＋学校数, 学校数] 每个候选属于哪所学校

    @property
    def n(self): return len(self.groups)

    @property
    def np_(self): return self.pair.shape[-1]

    @property
    def na(self): return self.anchor.shape[-1]

    @property
    def crossing_cols(self): return self.pair_blocks.get('crossing', slice(0, 0))   # 消融去掉跨路块时为空：跨路份额恒为 0


def build(dist, rules, cfg):
    """cfg：evidence 段（pair_blocks、school_blocks、exclusion_to_model、layout）。"""
    known = rules.known
    refs = sorted({(q, s, rules.kind[q]) for q, ss in known.items() if q not in dist.not_reference for s in ss})
    ri = np.array([r[0] for r in refs], int); rs = np.array([r[1] for r in refs], int); named = np.array([r[2] == 'named' for r in refs], bool)   # 没有参照时也是整数空数组
    groups = dist.groups; rgroups = groups[ri]; _, inv, cnt = np.unique(rgroups, return_inverse=True, return_counts=True)
    neg = np.zeros_like(rules.negative) if cfg['exclusion_to_model'] == 'none' else rules.negative if (dist.city == '长沙' or cfg['exclusion_to_model'] == 'all') else np.zeros_like(rules.negative)   # none：消融用，排除不交给模型
    st = rules.statuses; qst = st[:, rs]; rst = st[ri, rs]; n = dist.n; nr = len(ri)
    ls = lines(dist.lines_key or dist.city); usites, uinv = np.unique(ri, return_inverse=True)
    pc_, pw_ = crossing(ls, dist.coords, dist.coords[usites]); pc_, pw_ = pc_[:, uinv], pw_[:, uinv]
    _, sw_ = crossing(ls, dist.coords, dist.school_xy)   # 学校一路只用"2 公里内"，跨路类别在最终版里不用
    cn, strict = dist.community_name, dist.community_strict
    same = (cn[:, None] == cn[ri][None]) & (cn[:, None] != ''); both = strict[:, None] & strict[ri][None]
    known_rows = np.zeros(n, bool); known_rows[list(known)] = True
    P = dict(distance=geometry(dist.coords[ri][None] - dist.coords[:, None], 300), named_reference=np.broadcast_to(named[None, :, None], (n, nr, 1)).astype('float32'),
             policy=policy(qst, neg[:, rs]), agreement=((qst >= 0) & (rst[None] >= 0) & (qst == rst[None])).astype('float32'),
             disagreement=((qst >= 0) & (rst[None] >= 0) & (qst != rst[None])).astype('float32'), crossing=pc_, within_2km=pw_,
             learned_boundary=learned_boundary(ls, dist.coords, groups, known, ri),
             same_community=np.stack([same & both, (~same) & both], -1).astype('float32'))
    A = dict(distance=geometry(dist.school_xy[None] - dist.coords[:, None], 1000), policy=policy(st, neg), within_2km=sw_,
             community_clause=dist.community_school, name=np.where(known_rows[:, None, None], 0, dist.name_match).astype('float32'))
    if cfg.get('layout') == 'legacy':
        pair, pb, anchor, ab = legacy_layout(P, A)
    else:
        pair, pb = _stack(P, cfg['pair_blocks']); anchor, ab = _stack(A, cfg['school_blocks'])
    sup = np.array(sorted(q for q, s in known.items() if len(s) == 1 and q not in dist.not_training), int)
    x = Inputs(district=dist.name, pair=pair, anchor=anchor, pair_blocks=pb, anchor_blocks=ab, ref_sites=ri, ref_schools=rs,
               group_bias=-np.log(cnt[inv]).astype('float32'), self_mask=groups[:, None] == rgroups[None], groups=groups,
               y=np.array([known[q][0] if q in known else -1 for q in range(n)]), supervised=sup, school_n=dist.school_n)
    x.membership = torch.nn.functional.one_hot(torch.tensor(np.r_[rs, np.arange(dist.school_n)], dtype=torch.long), dist.school_n).float()
    assert np.isfinite(x.pair).all() and np.isfinite(x.anchor).all()
    return x


def _stack(blocks, names):
    cols, sl, i = [], {}, 0
    for k in names:
        a = blocks[k]; sl[k] = slice(i, i + a.shape[-1]); i += a.shape[-1]; cols.append(a)
    return np.concatenate(cols, -1).astype('float32'), sl


def legacy_layout(P, A):
    """两城最终版的列布局（参照 51 列、学校 48 列，被开关遮掉的列为 0），只用于和旧程序逐位核对（scripts/verify.py）。"""
    n, nr = P['distance'].shape[:2]; S = A['distance'].shape[1]
    env_p = np.zeros((n, nr, 16), np.float32); env_p[..., :5] = P['crossing']; env_p[..., [5, 6, 7, 15]] = P['within_2km']
    env_a = np.zeros((n, S, 16), np.float32); env_a[..., [5, 6, 7, 15]] = A['within_2km']     # 学校一路的跨路类别在最终版里被遮掉
    ex_a = np.zeros((n, S, 14), np.float32); ex_a[..., 6:10] = A['community_clause']; ex_a[..., 10:14] = A['name']
    pair = np.concatenate([P['distance'], P['named_reference'], P['policy'], P['agreement'], P['disagreement'], env_p, P['learned_boundary'], P['same_community']], -1).astype('float32')
    anchor = np.concatenate([A['distance'], A['policy'], env_a, ex_a], -1).astype('float32')
    return pair, dict(distance=slice(0, 5), crossing=slice(27, 32)), anchor, dict(distance=slice(0, 5))
