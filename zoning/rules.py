"""第一、二步：政策文字 → 每个（小区，学校）判入 / 排除 / 不知道。

输入是政策文字的原子观察（data/<区>/observations/，只记"看到了什么"）：
    名单命中      哪个小区被哪所学校的名单点名
    边判断        四至每条边，每个小区在线的哪一侧：在内 1 / 在外 0 / 不知道 -1
    行政范围      "某区范围内"这类限定
    "不含"命中    条款里排除的楼盘名对上了哪个小区
合成时只用五条原则（括号里是 统一证据执行器_20261105/rules.py 的编号，便于对照）：
    原则一  一条条款要各边同时满足（K1、K2）
    原则二  明写的例外优先：被"不含"点名的小区在这条条款外（K4）
    原则三  拿不准就不判（K3、K5、K6、K7、K8、X2）
    原则四  学校范围是该校各条款的"或"；按设置把范围外记为排除（X1、X3、X4）
    原则五  小区下结论有先后：名单点名 → 四至恰好判入一所且没被排除 → 多所或冲突不判 → 其余交给模型（P1–P5、P7）
三值"且""或"用 Kleene 强三值逻辑。
"""
import json
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

DIRS = '东南西北'

# ---------------- 设置 ----------------
BASE = dict(keep_out_unadmitted=True,   # 原则三：没接纳的条款，"在外"是否保留
            sides_to_model='admitted',  # 交给模型的方向状态来自哪些条款：all / admitted / pure
            blocked_drops=True,         # 原则三：学校四至被扣下时，方向状态是否也清掉
            negative='none',            # 原则四：排除规则 none / any_side / all_sides
            school_negative={},         # 已登记的单校修补
            nonarea_as_unknown=False,   # 原则四：学校还有名单等非四至条款时，是否当"不知道"并入"或"
            off=())                     # 关掉的规则编号（只用于复现旧设置）
_HZ_OLD = dict(BASE, keep_out_unadmitted=False, sides_to_model='pure', nonarea_as_unknown=True, off=('K8', 'X2'))
PRESETS = {
    # 两城最终版（11-03）实际在用的设置：各区旧程序沿用下来的
    'final_20261103': {'芙蓉区': dict(BASE, negative='any_side'), '雨花区': dict(BASE), '开福区': dict(BASE),
                       '天心区': dict(BASE, sides_to_model='all', blocked_drops=False, negative='all_sides', school_negative={8: 'any_side'}),
                       '拱墅区': _HZ_OLD, '萧山区': _HZ_OLD, '富阳区': _HZ_OLD, '临安区': _HZ_OLD},
    # 八区统一（11-06 验证非劣后采用，见 科研推进/规则设置统一_20261106/结论.md）；天心曙光小学是已登记的修补
    'unified_20261106': {d: dict(BASE, keep_out_unadmitted=False, sides_to_model='admitted', blocked_drops=True, negative='all_sides',
                                 school_negative={8: 'any_side'} if d == '天心区' else {}, nonarea_as_unknown=False, off=())
                         for d in ['天心区', '芙蓉区', '雨花区', '开福区', '拱墅区', '萧山区', '富阳区', '临安区', '上城区', '西湖区', '滨江区', '钱塘区']},
}


def AND(arrs, n):
    if not arrs: return np.full(n, -1, np.int8)
    a = np.stack(arrs); return np.where((a == 0).any(0), 0, np.where((a == 1).all(0), 1, -1)).astype(np.int8)


def OR(arrs, n):
    if not arrs: return np.full(n, -1, np.int8)
    a = np.stack(arrs); return np.where((a == 1).any(0), 1, np.where((a == 0).all(0), 0, -1)).astype(np.int8)


def on(S, rid): return rid not in S['off']


def load_observations(obs_dir):
    O = json.loads((obs_dir / 'observations.json').read_text(encoding='utf-8'))
    O['name_known'] = {int(q): ss for q, ss in O['name_known'].items()}
    for c in O['clauses']:
        for e in c['exclusions']:
            if 'resembles' in e: e['resembles'] = {int(q): v for q, v in e['resembles'].items()}
    with np.load(obs_dir / 'arrays.npz', allow_pickle=False) as z: A = {k: z[k] for k in z.files}
    if O['city'] == '杭州': A['admin'] = {k: A['admin__' + k] for k in O.get('admin_labels', [])}
    return O, A


# ---------------- 条款：原则一、二、三 ----------------
def clause(cl, O, A, S):
    """返回 (条款范围 v[n], 是否接纳, 方向状态 [n,4])。"""
    n = O['n']; s = cl['school']; sv = A['side_values']
    v = AND([sv[:, x['column']] for x in cl['sides']], n)                                   # 原则一：各边"且"
    if cl.get('admin') and on(S, 'K2'):                                                       # 原则一：再"且"上行政范围
        v = AND([v, A['admin_values'] if O['city'] == '长沙' else A['admin'][cl['admin']]], n)
    admitted = cl['admitted_by_parse'] and not any(e.get('broad') for e in cl['exclusions'])
    hit = np.zeros(n, bool); amb = np.zeros(n, bool); res = set()
    for e in cl['exclusions']:
        hit[e.get('hit', [])] = True; amb[e.get('ambiguous', [])] = True; res |= {int(q) for q in e.get('resembles', {})}
    if not admitted and on(S, 'K3'):                                                          # 原则三：没接纳
        v[v == 1] = -1
        if not S['keep_out_unadmitted']: v[v == 0] = -1
    if on(S, 'K4'): v[hit] = 0                                                                # 原则二："不含"点名 → 在外
    if on(S, 'K5'): v[amb & (v == 1)] = -1                                                    # 原则三："不含"拿不准
    if on(S, 'K6'): v[[q for q in sorted(res) if v[q] == 1 and not hit[q]]] = -1              # 原则三：名字像"不含"对象
    if not cl['parse_complete'] and on(S, 'K7'): v[v == 1] = -1                               # 原则三：条款没解析完整
    if on(S, 'K8'):                                                                           # 原则三：范围里落进别校点名的楼盘
        other = [q for q, ss in O['name_known'].items() if s not in ss and v[q] == 1]
        if len([q for q in other if not O['generic'][q]]) >= 2: v[v == 1] = -1; admitted = False   # 两个以上独特楼盘：整条暂停
        elif other: v[other] = -1
    give = {'all': True, 'admitted': admitted, 'pure': bool(cl.get('pure'))}[S['sides_to_model']]
    ds = []
    for d in DIRS:
        cols = [sv[:, x['column']] for x in cl['sides'] if x['direction'] == d]
        ds.append(AND(cols, n) if cols and give else np.full(n, -1, np.int8))
    return v, admitted, np.stack(ds, -1)


# ---------------- 学校：原则三、四 ----------------
def school(s, cls, O, A, S):
    """返回 (判入 hits[n], 方向状态 [n,4], 排除 neg[n])。"""
    n = O['n']; area = [c for c in cls if c['kind'] == 'area']
    res = [(c,) + clause(c, O, A, S) for c in area]
    unk = [np.full(n, -1, np.int8)] if S['nonarea_as_unknown'] and len(area) < len(cls) else []
    union = OR([r[1] for r in res] + unk, n)                                                  # 原则四：各条款"或"
    own = [q for q, ss in O['name_known'].items() if s in ss]
    explicit_alt = any(c['kind'] != 'area' for c in cls)
    blocked = bool([q for q in own if union[q] == 0 and not O['generic'][q]] and not explicit_alt) and on(S, 'X2')   # 原则三：本校点名的楼盘落在范围外 → 扣下
    hits = (union == 1) if not blocked else np.zeros(n, bool)
    st = np.full((n, 4), -1, np.int8)
    if not blocked or not S['blocked_drops']:
        for di in range(4): st[:, di] = OR([r[3][:, di] for r in res] + unk, n)
    rule = S['school_negative'].get(s, S['negative'])
    all_ready = all(all(x['status'] == 'ready' for x in c['sides']) for c in area)
    neg = np.zeros(n, bool)
    if on(S, 'X4') and not blocked and res and not explicit_alt and (rule == 'any_side' or (rule == 'all_sides' and all_ready)):
        neg = union == 0                                                                      # 原则四：范围外记排除
    return hits, st, neg


# ---------------- 小区：原则五 ----------------
def sites(O, hits, negative, S):
    known, kind, flagged = {}, {}, set()
    if on(S, 'P1'): flagged |= set(O.get('name_conflict', []))                               # 名单与"不含"冲突：不判
    for q, ss in O['name_known'].items():
        if q not in flagged and on(S, 'P2'): known[q] = list(ss); kind[q] = 'named'            # 名单点名（可多校）
    skip = set(O['review']) | (set(O['geo_excluded']) if on(S, 'P3') else set())
    for q in range(O['n']):
        if q in known or q in flagged or q in skip or q in O['name_known']: continue
        cand = np.flatnonzero(hits[q]).tolist(); forb = [s for s in cand if negative[q, s]]
        if len(cand) == 1 and not forb and on(S, 'P4'): known[q] = cand; kind[q] = 'boundary'   # 四至恰好判入一所
        elif len(cand) > 1 or forb: flagged.add(q)                                              # 多所或与排除冲突：不判
    return known, kind


@dataclass
class RuleOutput:
    statuses: np.ndarray   # [n, S, 4] 交给模型的方向状态：在内 1 / 在外 0 / 不知道 -1
    negative: np.ndarray   # [n, S]    排除
    known: dict            # {小区: [学校]} 规则判出（直接输出）
    kind: dict             # {小区: 'named' | 'boundary'}


def run(dist, preset, hangzhou_names='all', hangzhou_boundary='v1', hangzhou_community_rule=False):
    """对一个区执行规则，结果换到本区的小区编号（杭州的观察用全市编号）。
    hangzhou_boundary：杭州四至观察，v1（原执行器）或 v2（补路：连写的边、拐弯路线围成范围、找不到的路、街道限定）。
    hangzhou_names：杭州名单点名的来源。all＝范围文字＋入学平台的逐校户籍生点名名单（默认）；policy_text＝只用范围文字（scripts/export_text_only_names.py 导出）。"""
    obs = dist.obs_dir.parent / 'observations_bv2' if (dist.city == '杭州' and hangzhou_boundary == 'v2') else dist.obs_dir   # 杭州四至补路（科研推进/杭州四至补路_20261107）
    S = PRESETS[preset][dist.name]; O, A = load_observations(obs); n, m = O['n'], O['school_n']
    if O['city'] == '杭州' and hangzhou_names in ('policy_text', 'policy_text_aliases'):
        T = json.loads((dist.obs_dir / 'names_policy_text.json').read_text(encoding='utf-8'))
        O['name_known'] = {int(q): ss for q, ss in T['name_known'].items()}; O['name_conflict'] = T['name_conflict']
        A['name_negative'] = np.zeros((n, m), bool)
        for q, s in T['name_negative']: A['name_negative'][q, s] = True
        f = dist.obs_dir / 'names_policy_text_aliases.json'                                        # 核实过的别名（目前只有萧山）
        if hangzhou_names == 'policy_text_aliases' and f.exists():
            for q, ss in json.loads(f.read_text(encoding='utf-8'))['name_known'].items():
                O['name_known'][int(q)] = sorted(set(O['name_known'].get(int(q), [])) | set(ss))
    else: assert hangzhou_names in ('all', 'policy_text', 'policy_text_aliases'), hangzhou_names      # 长沙不受这项影响
    by = defaultdict(list)
    for c in O['clauses']: by[c['school']].append(c)
    hits = np.zeros((n, m), bool); st = np.full((n, m, 4), -1, np.int8); neg = np.zeros((n, m), bool)
    for s in range(m): hits[:, s], st[:, s], neg[:, s] = school(s, by.get(s, []), O, A, S)
    for q, ss in O['name_known'].items():
        for s in ss: neg[q, s] = False                                                        # 名单点名的不排除
    negative = A['name_negative'].astype(bool) | neg
    known, kind = sites(O, hits, negative, S)
    ix = dist.orig if dist.orig is not None else np.arange(n); pos = {int(o): i for i, o in enumerate(ix)}
    known_m = {pos[q]: sorted(int(s) for s in v) for q, v in known.items() if q in pos}; kind_m = {pos[q]: k for q, k in kind.items() if q in pos}
    if hangzhou_community_rule and dist.city == '杭州':                                      # 社区规则：归属把握大、所在社区只被一所学校整体点名、该校四至没排除
        neg_m = negative[ix]; cs = dist.community_school
        for q in range(len(ix)):
            w = np.flatnonzero(cs[q, :, 0] > 0)
            if q not in known_m and dist.community_strict[q] and len(w) == 1 and not neg_m[q, w[0]]: known_m[q] = [int(w[0])]; kind_m[q] = 'community'
    return RuleOutput(statuses=st[ix], negative=negative[ix], known=known_m, kind=kind_m)
