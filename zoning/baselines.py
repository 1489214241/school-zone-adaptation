"""主表的对照方法。每个函数返回一个区每个小区对各学校的概率 [n, S]，和模型用同一套计分（evaluate.correct）。
所有对照用同一个支持集：rules_out.known（规则判出的小区 ＋ 每校 K 个核验参照，见 references.py）。

不学习：
    nearest_school     规则没判出的就给直线最近的学校（业界兜底做法，如专利 CN118939674A）
    copy_nearest       抄最近一个已知小区（不同楼盘组）的学校（1-NN）
    label_propagation  标签传播（Zhou 等 NIPS 2004，局部与全局一致性）：坐标 10 近邻图，F=(I−αS)⁻¹Y；
                       α∈{0.1,0.3,0.5,0.8,0.9,0.99}、是否做类别质量归一化（Zhu 等 ICML 2003）在支持集留出上选
学习（只在本区支持集上训练；单行特征的列跟着本区学校走，各区列数不同，原版不能跨区共享）：
    protonet           原型网络（Snell 等 NIPS 2017）：支持集上按 N-way K-shot 出题训练嵌入，学校原型＝该校支持样本嵌入的均值
    modernnca          ModernNCA 官方核心（third_party/TALENT，提交 39b5402，原样调用）：本区支持集上训练，SNS 抽样比例 0.5
单行特征（node_table，照 科研推进/ModernNCA长沙验证_20261108 的做法）：
    geometry 视图：坐标、到每所学校的距离方向（每校 5 列）；
    policy 视图：再加每所学校的政策状态与排除（13）、2 公里内（1）、整社区点名（4）、名字对上（4），以及社区独热。
    数值列只用训练用的支持样本标准化。每个楼盘组只取一个点进训练，避免同楼盘既当题又当参照。
constrain：把政策明写排除的学校概率清零（全被排除时退回原概率）。原版（raw）和加排除（policy）两种都返回。
"""
import hashlib, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .evidence import geometry, policy as policy_cols

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'third_party'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MNCA_GRID = [dict(view=v, encoding=e, lr=lr) for v in ('geometry', 'policy') for e in ('none', 'plr_0.01', 'plr_0.1') for lr in (1e-3, 3e-3)]   # 与 11-08 Codex 那轮相同的 12 组
MNCA_FIXED = dict(weight_decay=1e-4, dim=64, blocks=1, block_width=128, dropout=0.1, sample_rate=0.5, batch_size=64, max_epochs=160, val_every=5, min_epochs=40, patience=30)
PROTO_GRID = [dict(view=v, lr=lr) for v in ('geometry', 'policy') for lr in (1e-3, 3e-3)]
PROTO_FIXED = dict(dim=64, hidden=128, dropout=0.1, weight_decay=1e-4, max_way=30, max_shot=5, max_query=5, episodes=3000, val_every=100, min_episodes=500, patience=800)


# ---------------- 支持集 ----------------
def support(dist, rules_out):
    """单校、可当训练题的已知小区 → {小区: 学校}。"""
    return {q: ss[0] for q, ss in rules_out.known.items() if len(ss) == 1 and q not in dist.not_training}


def one_per_group(dist, ids):
    seen, out = set(), []
    for q in sorted(ids):
        g = dist.groups[q]
        if g not in seen: seen.add(g); out.append(q)
    return np.array(out, int)


def constrain(P, rules_out):
    Q = np.where(rules_out.negative, 0, P); m = Q.sum(1, keepdims=True); bad = m[:, 0] < 1e-12
    return np.where(bad[:, None], P, Q / np.maximum(m, 1e-12))


# ---------------- 不学习的对照 ----------------
def nearest_school(dist, rules_out):
    d = np.linalg.norm(dist.school_xy[None] - dist.coords[:, None], axis=-1); P = np.zeros_like(d); P[np.arange(dist.n), d.argmin(1)] = 1
    return P


def copy_nearest(dist, rules_out):
    refs = np.array(sorted(q for q in rules_out.known if q not in dist.not_reference), int); P = nearest_school(dist, rules_out).copy()
    if not len(refs): return P
    D = np.linalg.norm(dist.coords[refs][None] - dist.coords[:, None], axis=-1); D[dist.groups[:, None] == dist.groups[refs][None]] = np.inf
    j = D.argmin(1)
    for q in range(dist.n):
        if np.isfinite(D[q, j[q]]):
            P[q] = 0
            for s in rules_out.known[int(refs[j[q]])]: P[q, s] += 1 / len(rules_out.known[int(refs[j[q]])])
    return P


LP_GRID = [dict(alpha=a, cmn=c) for a in (0.1, 0.3, 0.5, 0.8, 0.9, 0.99) for c in (False, True)]


def _propagate(Sm, Y, alpha, cmn, prior):
    F = np.maximum(np.linalg.solve(np.eye(len(Sm)) - alpha * Sm, Y), 0)
    if cmn: F = F / np.maximum(F.sum(0, keepdims=True), 1e-12) * prior[None]          # 类别质量归一化（Zhu 等 2003）
    return F


def label_propagation(dist, rules_out, k=10, chosen=None):
    """α 和是否做类别质量归一化在支持集上留出约 20% 按准确率选（与学习型对照同一种留出），再用全部支持集传播。"""
    from scipy.spatial import cKDTree
    n, S = dist.n, dist.school_n; xy = dist.coords + 1e-3 * np.arange(n)[:, None]   # 同坐标的点微移，避免零距离
    dd, ii = cKDTree(xy).query(xy, k + 1); sigma = max(float(np.mean(dd[:, -1])), 1.0)
    W = np.zeros((n, n)); r = np.repeat(np.arange(n), k); c = ii[:, 1:].ravel(); W[r, c] = np.exp(-(dd[:, 1:].ravel() / sigma) ** 2); W = np.maximum(W, W.T)
    dg = W.sum(1); Dm = 1 / np.sqrt(np.maximum(dg, 1e-12)); Sm = Dm[:, None] * W * Dm[None]

    def labels(qs):
        Y = np.zeros((n, S))
        for q in qs:
            for s in rules_out.known[q]: Y[q, s] += 1 / len(rules_out.known[q])
        return Y
    refs = [q for q in rules_out.known if q not in dist.not_reference]
    if chosen is None:
        sup = support(dist, rules_out); tr, va = split_validation(dist, sup, one_per_group(dist, list(sup)))
        hide = set(dist.groups[va].tolist()); keep = [q for q in refs if dist.groups[q] not in hide]
        Y = labels(keep); prior = Y.sum(0) / max(Y.sum(), 1e-12); best = None
        for g in LP_GRID:
            F = _propagate(Sm, Y, g['alpha'], g['cmn'], prior); acc = float(np.mean([F[q].argmax() == sup[q] for q in va])) if len(va) else 0.
            if best is None or acc > best[0] + 1e-12: best = (acc, g)
        chosen = dict(best[1], validation_accuracy=best[0])
    Y = labels(refs); prior = Y.sum(0) / max(Y.sum(), 1e-12); F = _propagate(Sm, Y, chosen['alpha'], chosen['cmn'], prior); m = F.sum(1, keepdims=True)
    P = np.where(m > 1e-12, F / np.maximum(m, 1e-12), nearest_school(dist, rules_out)); label_propagation.last_chosen = chosen
    return P


# ---------------- 单行特征 ----------------
def node_table(dist, rules_out, view, exclusion_to_model='all'):
    n, S = dist.n, dist.school_n
    geo = geometry(dist.school_xy[None] - dist.coords[:, None], 1000)                  # [n, S, 5]
    num = np.concatenate([dist.coords / 1000, geo.reshape(n, -1)], 1).astype(np.float32)
    if view == 'geometry': return num, num.shape[1]
    neg = rules_out.negative if (dist.city == '长沙' or exclusion_to_model == 'all') else np.zeros_like(rules_out.negative)
    within = (np.linalg.norm(dist.school_xy[None] - dist.coords[:, None], axis=-1) <= 2000)[..., None].astype(np.float32)
    known_rows = np.zeros(n, bool); known_rows[list(rules_out.known)] = True
    name = np.where(known_rows[:, None, None], 0, dist.name_match).astype(np.float32)
    cat = np.concatenate([policy_cols(rules_out.statuses, neg), within, dist.community_school.astype(np.float32), name], -1).reshape(n, -1)
    cats = np.unique(dist.community_name); com = (dist.community_name[:, None] == cats[None]).astype(np.float32); com[:, cats == ''] = 0
    return np.concatenate([num, cat, com], 1).astype(np.float32), num.shape[1]


def _standardize(x, nnum, fit_ids):
    x = x.copy(); mu = x[fit_ids, :nnum].mean(0); sd = x[fit_ids, :nnum].std(0); sd[sd < 1e-6] = 1; x[:, :nnum] = (x[:, :nnum] - mu) / sd
    assert np.isfinite(x).all(); return x


def split_validation(dist, sup, ids):
    """每校按楼盘名哈希留约 20% 当验证（该校 ≥3 个时，至少留 2 个训练）。"""
    tr, va = [], []
    for s in sorted({sup[q] for q in ids}):
        ii = sorted([q for q in ids if sup[q] == s], key=lambda q: hashlib.sha256(f'BL|{dist.name}|{dist.groups[q]}'.encode()).hexdigest())
        nv = min(max(1, int(round(len(ii) * .2))), len(ii) - 2) if len(ii) >= 3 else 0
        va += ii[:nv]; tr += ii[nv:]
    return np.array(sorted(tr), int), np.array(sorted(va), int)


def _nll(P, ids, y):
    return float(-np.log(np.clip(P[ids, y[ids]], 1e-7, None)).mean())


# ---------------- ModernNCA（官方核心） ----------------
def _mnca_model(cfg, d_in, nnum, S):
    from TALENT.model.models.modernNCA import ModernNCA
    emb = None if cfg['encoding'] == 'none' else dict(type='PLREmbeddings', n_frequencies=4, frequency_scale=float(cfg['encoding'].split('_')[1]), d_embedding=4, lite=True)
    F = MNCA_FIXED
    return ModernNCA(d_in=d_in, d_num=nnum, d_out=S, dim=F['dim'], dropout=F['dropout'], d_block=F['block_width'], n_blocks=F['blocks'], num_embeddings=emb, sample_rate=F['sample_rate']).to(DEV)


def _mnca_probs(m, X, y, cand, ids=None):
    ids = np.arange(len(X)) if ids is None else ids; m.eval(); out = []
    with torch.no_grad():
        xc = torch.as_tensor(X[cand], device=DEV); yc = torch.as_tensor(y[cand], device=DEV)
        for q in np.array_split(ids, max(1, int(np.ceil(len(ids) / 256)))):
            z = m(torch.as_tensor(X[q], device=DEV), None, xc, yc, False); p = (z.exp() - 1e-7).clamp_min(0); out.append((p / p.sum(1, keepdim=True).clamp_min(1e-12)).cpu().numpy())
    return np.concatenate(out)


def _mnca_fit(X, nnum, y, S, tr, va, cfg, seed, epochs=None):
    F = MNCA_FIXED; torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = _mnca_model(cfg, X.shape[1], nnum, S); opt = torch.optim.AdamW(m.parameters(), lr=cfg['lr'], weight_decay=F['weight_decay'])
    Xt = torch.as_tensor(X, device=DEV); yt = torch.as_tensor(y, device=DEV); best, best_ep, last = np.inf, 0, 0
    for ep in range(1, (epochs or F['max_epochs']) + 1):
        m.train(); perm = rng.permutation(tr)
        for q in np.array_split(perm, max(1, int(np.ceil(len(perm) / F['batch_size'])))):
            if len(q) < 2: continue
            other = np.setdiff1d(tr, q); lp = m(Xt[q], yt[q], Xt[other], yt[other], True)
            loss = nn.functional.nll_loss(lp, yt[q]); opt.zero_grad(); loss.backward(); opt.step()
        if epochs is None and len(va) and (ep % F['val_every'] == 0 or ep == 1):
            vl = _nll(_mnca_probs(m, X, y, tr, va), np.arange(len(va)), y[va])
            if vl < best - 1e-6: best, best_ep, last = vl, ep, ep
            if ep >= F['min_epochs'] and ep - last >= F['patience']: break
    return m, (best, best_ep if epochs is None else epochs)


# ---------------- 原型网络 ----------------
class _Encoder(nn.Module):
    def __init__(self, d_in):
        super().__init__(); F = PROTO_FIXED
        self.net = nn.Sequential(nn.Linear(d_in, F['hidden']), nn.ReLU(), nn.Dropout(F['dropout']), nn.Linear(F['hidden'], F['dim']))

    def forward(self, x): return self.net(x)


def _proto_probs(enc, X, y, S, cand, ids=None):
    ids = np.arange(len(X)) if ids is None else ids; enc.eval()
    with torch.no_grad():
        z = enc(torch.as_tensor(X, device=DEV)); zc = z[torch.as_tensor(cand, device=DEV)]; yc = torch.as_tensor(y[cand], device=DEV)
        proto = torch.zeros(S, z.shape[1], device=DEV).index_add_(0, yc, zc); cnt = torch.bincount(yc, minlength=S).float(); has = cnt > 0
        proto = proto / cnt.clamp_min(1)[:, None]; logit = -torch.cdist(z[torch.as_tensor(ids, device=DEV)], proto).square(); logit[:, ~has] = -torch.inf
        return torch.softmax(logit, 1).cpu().numpy()


def _proto_fit(X, y, S, tr, va, cfg, seed, episodes=None):
    F = PROTO_FIXED; torch.manual_seed(seed); rng = np.random.default_rng(seed)
    enc = _Encoder(X.shape[1]).to(DEV); opt = torch.optim.AdamW(enc.parameters(), lr=cfg['lr'], weight_decay=F['weight_decay']); Xt = torch.as_tensor(X, device=DEV)
    by = {}
    for q in tr: by.setdefault(int(y[q]), []).append(q)
    classes = [s for s, v in by.items() if len(v) >= 2]; best, best_ep, last = np.inf, 0, 0
    for ep in range(1, (episodes or F['episodes']) + 1):
        enc.train(); way = rng.choice(classes, min(F['max_way'], len(classes)), replace=False); si, qi, qy = [], [], []
        for c, s in enumerate(way):
            v = rng.permutation(by[s]); ns = int(rng.integers(1, min(F['max_shot'], len(v) - 1) + 1)); nq = min(F['max_query'], len(v) - ns)
            si.append(v[:ns]); qi += list(v[ns:ns + nq]); qy += [c] * nq
        sizes = [len(s_) for s_ in si]; allidx = np.concatenate(si + [np.array(qi, int)]); z = enc(Xt[torch.as_tensor(allidx, device=DEV)])   # 只算这一轮抽到的
        zs = torch.split(z[:sum(sizes)], sizes); proto = torch.stack([v.mean(0) for v in zs]); zq = z[sum(sizes):]
        logit = -torch.cdist(zq, proto).square(); loss = nn.functional.cross_entropy(logit, torch.as_tensor(qy, device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        if episodes is None and len(va) and (ep % F['val_every'] == 0):
            vl = _nll(_proto_probs(enc, X, y, S, tr, va), np.arange(len(va)), y[va])
            if vl < best - 1e-6: best, best_ep, last = vl, ep, ep
            if ep >= F['min_episodes'] and ep - last >= F['patience']: break
    return enc, (best, best_ep if episodes is None else episodes)


def _by_group(predict, n, cand, groups):
    """预测全部小区；与某个候选同楼盘的小区，去掉那个候选再算（和模型"同楼盘的参照不能用"一致）。predict(候选, 小区) → 概率。"""
    P = predict(cand, np.arange(n)); cset = set(cand.tolist()); cg = set(groups[cand].tolist())
    clash = np.array([q for q in range(n) if groups[q] in cg and q not in cset], int)
    for g in sorted(set(groups[clash].tolist())):
        qs = clash[groups[clash] == g]; P[qs] = predict(cand[groups[cand] != g], qs)
    return P


# ---------------- 学习型对照的入口 ----------------
def learned(method, dist, rules_out, seeds, chosen=None):
    """method：'modernnca' 或 'protonet'。chosen：已选好的设置（不给就在本区支持集上按验证 NLL 选）。
    返回 dict(raw=[n,S] 种子平均概率, policy=加排除, chosen=设置与轮数, n_support=支持楼盘组数)。"""
    sup = support(dist, rules_out); y = np.zeros(dist.n, np.int64)
    for q, s in sup.items(): y[q] = s
    ids = one_per_group(dist, list(sup)); S = dist.school_n; fit, probs = (_mnca_fit, _mnca_probs) if method == 'modernnca' else (_proto_fit, _proto_probs)
    if chosen is None:
        tr, va = split_validation(dist, sup, ids); trials = []
        for cfg in (MNCA_GRID if method == 'modernnca' else PROTO_GRID):
            X, nnum = node_table(dist, rules_out, cfg['view']); X = _standardize(X, nnum, tr)
            args = (X, nnum, y, S, tr, va, cfg, seeds[0]) if method == 'modernnca' else (X, y, S, tr, va, cfg, seeds[0])
            _, (vl, ep) = fit(*args); trials.append((vl, ep, cfg))
        vl, ep, cfg = min(trials, key=lambda t: t[0]); chosen = dict(cfg=cfg, steps=max(int(ep), 1), validation_nll=vl)
    cfg = chosen['cfg']; X, nnum = node_table(dist, rules_out, cfg['view']); X = _standardize(X, nnum, ids); P = []
    for seed in seeds:
        if method == 'modernnca':
            m, _ = _mnca_fit(X, nnum, y, S, ids, np.array([], int), cfg, seed, epochs=chosen['steps']); P.append(_by_group(lambda c, q: _mnca_probs(m, X, y, c, q), dist.n, ids, dist.groups))
        else:
            m, _ = _proto_fit(X, y, S, ids, np.array([], int), cfg, seed, episodes=chosen['steps']); P.append(_by_group(lambda c, q: _proto_probs(m, X, y, S, c, q), dist.n, ids, dist.groups))
    raw = np.mean(P, 0)
    return dict(raw=raw, policy=constrain(raw, rules_out), chosen=chosen, n_support=len(ids))
