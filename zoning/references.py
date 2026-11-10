"""每校额外核验参照（K 个）：在规则判出的小区之外，每所学校再给 K 个核实过归属的楼盘，当作已知小区（参照，同时也是训练题）。

问题设定（科研推进/问题定义与参照数_20261109.md）：跨区少样本分类。支持集 = 规则判出的小区 ＋ 每校 K 个核验参照；
K 是"额外核验预算"，不是标准少样本里"每类只有 K 个"——规则判出的参照照常保留。

核实过程用官方答案模拟（等于假设每个都查得到、查得对）：
    每所学校的候选 = 规则没判出、有官方答案、整个楼盘组的答案都只有这一所学校的楼盘组；
    每校随机挑 min(K, 候选数) 个楼盘组，组内有答案的点都标成该校（kind = 'verified'）。
    随机数 = (seed, 区序号, K)，同一 seed、K 下每次相同；换 seed 就是换一批核验对象。
挑中的楼盘不再参加评价（evaluate.eval_rows 只评规则和核验都没判出的小区）。
核验参照不是政策点名的：evidence 里"参照是名单点名的"一列为 0。
"""
from collections import defaultdict
from dataclasses import replace
import numpy as np
from .data import ALL, EXTRA


def _index(name):
    order = ALL + [d for v in EXTRA.values() for d in v]
    return order.index(name) if name in order else sum(map(ord, name))


def candidates(dist, known):
    """{学校: [楼盘组，按名字排序]}。"""
    by_group = defaultdict(list)
    for q, t in dist.answers.items():
        if q not in known: by_group[dist.groups[q]].append((q, tuple(t)))
    out = defaultdict(list)
    for g, items in by_group.items():
        truths = {t for _, t in items}
        if len(truths) == 1:
            (t,) = truths
            if len(t) == 1 and t[0] >= 0: out[int(t[0])].append(g)
    return {s: sorted(gs) for s, gs in out.items()}


def pick(dist, known, k, seed):
    """返回 ({小区: [学校]}, {学校: 挑中的楼盘组})。"""
    if k <= 0: return {}, {}
    rng = np.random.default_rng([int(seed), _index(dist.name), int(k)]); cand = candidates(dist, known); chosen = {}
    for s in sorted(cand):
        gs = cand[s]; m = min(k, len(gs)); chosen[s] = sorted(rng.choice(gs, m, replace=False).tolist())
    keep = {g: s for s, gs in chosen.items() for g in gs}
    picked = {q: [keep[dist.groups[q]]] for q in dist.answers if q not in known and dist.groups[q] in keep}
    return picked, chosen


def apply(cfg, D, R):
    """cfg：设置文件的 references 段（k、seed）。返回 (并入核验参照后的 R, 记录)；k 缺省或为 0 时原样返回。"""
    if not cfg or cfg.get('k', 0) <= 0: return R, None
    R2, rec = {}, {}
    for d in D:
        p, chosen = pick(D[d], R[d].known, cfg['k'], cfg['seed']); R2[d] = augment(R[d], p)
        rec[d] = dict(points=len(p), schools=len(chosen), groups=sum(len(g) for g in chosen.values()), chosen={int(s): g for s, g in chosen.items()})
    return R2, rec


def augment(rules_out, picked):
    """把核验参照并入规则结果（规则判出的不动）。"""
    assert not (set(picked) & set(rules_out.known))
    known = dict(rules_out.known); kind = dict(rules_out.kind)
    for q, ss in picked.items(): known[q] = list(ss); kind[q] = 'verified'
    return replace(rules_out, known=known, kind=kind)
