"""计分：概率最高的学校是否等于官方答案（杭州只看有地图的学校）。区间：按区分层、按空间块重抽 3000 次。

两种范围：
  模型部分（correct）：规则没判出、有官方答案的小区，用模型的第一名；
  整体（overall）：全部有官方答案的小区，规则判出的用规则答案，其余用模型的第一名。
计分口径（与 88.21% 相同）：
  模型给出的学校 ∩ 计分范围 = 官方答案集合 才算对；杭州的计分范围是有地图的学校。
  规则给出多所学校时，任一在官方答案里算对。
  "地图＋名单"口径下的名单答案（source='list'）：计分范围加上名单点名的学校；名单点了多所（双学区）时，模型选中其中任一所算对，
  另记严格口径（strict：必须等于整个集合）。
空间块：1 公里网格。"""
import numpy as np


def block(dist, q):
    x, y = dist.coords[q]; return str((int(x // 1000), int(y // 1000)))


def eval_rows(dist, known):
    """规则没判出、有官方答案的小区。整理版设置下与 data 里导出的 eval 逐个相同（同一批小区、同一空间块分组）。"""
    src = dist.answer_source or {}
    return [dict(q=q, truth=t, block=block(dist, q), source=src.get(q, 'map')) for q, t in sorted(dist.answers.items()) if q not in known]


def _scope(e, dist):
    return dist.scope | set(e['truth']) if (dist.scope is not None and e.get('source') == 'list') else dist.scope


def _strict(pred, e, dist):
    sc = _scope(e, dist); return (set(pred) & sc if sc is not None else set(pred)) == set(e['truth'])


def _model_ok(p, e, dist):
    strict = _strict({int(p)}, e, dist)
    ok = (int(p) in set(e['truth'])) if e.get('source') == 'list' else strict
    return ok, strict


def correct(P, dist, rows=None):
    """模型部分每个评价小区判对没有：[dict(区, 城市, 空间块, 对否, 严格对否, 来源)]。P：[n, 学校数] 概率（多个种子已平均）。rows 缺省用 data 里导出的 eval。"""
    out = []
    for e in (rows if rows is not None else dist.eval_rows):
        ok, strict = _model_ok(P[e['q']].argmax(), e, dist)
        out.append(dict(d=dist.name, city=dist.city, block=str(e['block']), ok=ok, strict=strict, source=e.get('source', 'map')))
    return out


def overall(P, dist, rules_out):
    """整体：全部有官方答案的小区。"""
    src = dist.answer_source or {}; out = []
    for q, t in sorted(dist.answers.items()):
        e = dict(q=q, truth=t, source=src.get(q, 'map'))
        if q in rules_out.known:
            ok = any(s in t for s in rules_out.known[q]); strict = _strict(rules_out.known[q], e, dist); by = 'rule'
        else:
            ok, strict = _model_ok(P[q].argmax(), e, dist); by = 'model'
        out.append(dict(d=dist.name, city=dist.city, block=block(dist, q), ok=ok, strict=strict, source=e['source'], by=by))
    return out


def bootstrap(rows, delta, seed=20261104, draws=3000):
    rng = np.random.default_rng(seed); num = np.zeros(draws); den = np.zeros(draws)
    for d in sorted({r['d'] for r in rows}):
        ix = [i for i, r in enumerate(rows) if r['d'] == d]; keys = sorted({rows[i]['block'] for i in ix}); lk = {k: j for j, k in enumerate(keys)}
        b = np.array([lk[rows[i]['block']] for i in ix]); n = np.bincount(b); s = np.bincount(b, weights=np.asarray(delta, float)[ix])
        draw = rng.integers(0, len(keys), (draws, len(keys))); num += s[draw].sum(1); den += n[draw].sum(1)
    return (np.quantile(num / den, [.025, .975]) * 100).tolist()


def _groups(rows):
    names = ['两城', '长沙', '杭州'] + sorted({r['d'] for r in rows}, key=[r['d'] for r in rows].index)
    sel = {n: (lambda r, n=n: n == '两城' or r['city'] == n or r['d'] == n) for n in names}
    if any(r.get('source') == 'list' for r in rows):
        sel['杭州｜地图答案'] = lambda r: r['city'] == '杭州' and r.get('source') == 'map'
        sel['杭州｜名单答案'] = lambda r: r['city'] == '杭州' and r.get('source') == 'list'
        sel['两城｜只算地图答案'] = lambda r: r.get('source', 'map') == 'map'
    return sel


def summary(rows):
    res = {}
    for name, f in _groups(rows).items():
        rr = [r for r in rows if f(r)]
        if rr: res[name] = dict(n=len(rr), accuracy=float(np.mean([r['ok'] for r in rr]) * 100), strict=float(np.mean([r.get('strict', r['ok']) for r in rr]) * 100))
    return res


def compare(rows_a, rows_b, cfg):
    """A − B，两城、长沙、杭州给区间。两组 rows 必须是同一批小区、同一顺序。"""
    res = {}
    for name in ['两城', '长沙', '杭州']:
        ia = [i for i, r in enumerate(rows_a) if name == '两城' or r['city'] == name]
        x = np.array([rows_a[i]['ok'] for i in ia], float) - np.array([rows_b[i]['ok'] for i in ia], float)
        res[name] = dict(delta=float(x.mean() * 100), ci=bootstrap([rows_a[i] for i in ia], x, cfg['bootstrap_seed'], cfg['draws']),
                         fixed=int((x > 0).sum()), broken=int((x < 0).sum()))
    return res
