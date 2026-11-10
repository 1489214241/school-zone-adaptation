"""第五步：训练。

训练题（一个目标区）：
    本区的规则已知小区（单校）；
    其余七区的规则已知小区；
    其余七区规则没判出、但有官方答案的小区（"跨区未覆盖监督"，提升主要来自这一部分）。
    目标区自己的官方答案从不进训练。
权重：每个楼盘组等权（组内平分；各批次按组数加权）。
起点：先用本区规则已知小区拟合一个只有线性项的模型（LinearRetrieval，100 步），作为线性项的起点；再整体训练 40 步。
"""
import numpy as np
import torch
from torch import nn
from . import DEVICE
from .data import CITIES, ALL, city_of
from .model import LinearRetrieval, RetrievalScorer, batch


def source_districts(target, which='all', pool=None):
    """别区训练题来自哪些区。all：其余七区（顺序：另一座城市的四区，再本城其余三区）；same_city：本城其余三区；other_city：另一座城市的四区。
    pool：可当来源的区（缺省＝现有八区；加了杭州新区时写在设置 train.items.pool 里）。"""
    pool = list(pool) if pool is not None else ALL
    other = [d for d in pool if city_of(d) != city_of(target)]; same = [d for d in pool if city_of(d) == city_of(target) and d != target]
    return {'all': other + same, 'same_city': same, 'other_city': other}[which]


def _cpu_tensors(x, ids):
    q = np.asarray(ids, int); _, inv, cnt = np.unique(x.groups[q], return_inverse=True, return_counts=True); w = 1 / cnt[inv]; w = w / w.sum()
    return dict(pair=torch.from_numpy(x.pair[q]), anchor=torch.from_numpy(x.anchor[q]), valid=torch.from_numpy(~x.self_mask[q]),
                y=torch.tensor(x.y[q], dtype=torch.long), weight=torch.tensor(w, dtype=torch.float32))


def warm_start(x, reg=.01, steps=100, lr=.04, school_off=False):
    """线性起点：只用本区规则已知小区。"""
    t = _cpu_tensors(x, x.supervised); model = LinearRetrieval(x); model.school_off = school_off; opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(); p = model(t)
        loss = -(p[torch.arange(len(x.supervised)), t['y']].clamp_min(1e-30).log() * t['weight']).sum() + reg * model.penalty()
        assert torch.isfinite(loss); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
    return model


def uncovered_answers(dist, x):
    """规则没判出、有唯一官方答案的小区 {小区: 学校}（别区的这部分当训练题）。按本次规则结果现算；整理版设置下与 data 里导出的 train 逐个相同。"""
    return {q: a[0] for q, a in dist.answers.items() if x.y[q] < 0 and len(a) == 1 and a[0] >= 0}


def training_batches(target, X, D, items):
    """X：{区: Inputs}；D：{区: District}；items：训练题设置（target_known、other_known、other_answers、sources、exclude：不当来源的区）。"""
    bs = []; names = []
    if items.get('target_known', True): bs.append(batch(X[target], X[target].supervised)); names.append(target)
    for o in source_districts(target, items.get('sources', 'all'), items.get('pool')):
        if o in items.get('exclude', ()): continue
        ids = set(X[o].supervised.tolist()) if items.get('other_known', True) else set()
        lab = uncovered_answers(D[o], X[o]) if items.get('other_answers', True) else {}
        ids |= set(lab)
        if ids: bs.append(batch(X[o], np.array(sorted(ids), int), lab)); names.append(o)
    ng = [len(set(b['groups'])) for b in bs]; tot = sum(ng); balance = items.get('balance', 'group')
    if balance == 'group':
        for b, n_ in zip(bs, ng): b['weight'] = b['weight'] * len(bs) * n_ / tot                 # 每个楼盘组等权
    elif balance == 'city':                                                                       # 试验开关：两座城市总权重相等，城内每个楼盘组等权
        cities = [city_of(n_) for n_ in names]; cs = sorted(set(cities))
        for b, n_, c in zip(bs, ng, cities): b['weight'] = b['weight'] * len(bs) / len(cs) * n_ / sum(m for m, c2 in zip(ng, cities) if c2 == c)
    else: assert balance == 'district'                                                            # 试验开关：每个区总权重相等（各批权重本来就各自和为 1）
    for b, n_ in zip(bs, names): assert (b['y'] >= 0).all(); b['district'] = n_
    return bs


def fit(target, X, D, cfg, seeds):
    """训练一个目标区：训练题和线性起点只算一次，每个种子各训一个模型。返回 {种子: (模型参数, 全区每个小区的学校概率 [n, 学校数])}。"""
    x = X[target]; tc = cfg['train']; reg = tc['reg'].get(target, tc['reg']['default'])
    off = cfg['model'].get('paths', ['pair', 'school']) == ['pair']                               # 消融：去掉学校一路
    bs = training_batches(target, X, D, tc['items']); start = warm_start(x, school_off=off); out = {}
    cond = cfg['model'].get('condition', False); drop = tc.get('reference_dropout', 0)            # 试验开关（zoning/adapt.py），默认都不启用
    if cond or drop:
        from . import adapt
        sup = [adapt.support(X[b['district']]) for b in bs]; zs = [s_.descriptor().to(DEVICE) for s_ in sup]
    for seed in seeds:
        model = RetrievalScorer(x.np_, x.na, start, seed, cfg['model']['width']); model.school_off = off
        if cond: model.enable_condition(x.pair_blocks, x.anchor_blocks, adapt.ZDIM)
        model = model.to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=tc['lr']); gen = torch.Generator().manual_seed(int(seed))
        for _ in range(tc['steps']):
            opt.zero_grad(); loss = model.regularization(reg)
            for i, b in enumerate(bs):
                if drop:
                    keep = sup[i].sample_keep(gen, drop); b = adapt.masked(b, keep)
                    if cond: b['z'] = sup[i].descriptor(keep).to(DEVICE)
                elif cond: b = dict(b, z=zs[i])
                p = model(b); loss = loss - (p[torch.arange(len(b['ids']), device=DEVICE), b['y']].clamp_min(1e-30).log() * b['weight']).sum()
            assert torch.isfinite(loss); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
        out[seed] = (model.state_dict(), predict_proba(model, x))
    return out


def predict_proba(model, x, chunk=128):
    ids = np.arange(x.n); out = []; z = None
    if getattr(model, 'cond', False):
        from . import adapt
        z = adapt.support(x).descriptor().to(DEVICE)                                              # 预测时不藏参照，区概况用全部已知小区算
    with torch.no_grad():
        for st in range(0, len(ids), chunk):
            t = batch(x, ids[st:st + chunk])
            if z is not None: t['z'] = z
            out.append(model(t).cpu().numpy())
    P = np.concatenate(out); assert np.isfinite(P).all() and np.allclose(P.sum(1), 1, atol=1e-5)
    return P
