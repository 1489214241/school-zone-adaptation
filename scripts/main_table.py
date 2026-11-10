"""主表与台阶表（科研推进/问题定义与参照数_20261109.md；记忆 model-and-baselines-decision-20261109）。
  python scripts/main_table.py configs/default.json --tag pre_v1            # 跑全部（已跑完的跳过）
  python scripts/main_table.py configs/default.json --tag pre_v1 --report   # 只汇总
每校额外核验 K 个参照，K ∈ {0,1,3,5}；K>0 时换 3 批核验对象（seed 1、2、3），K=0 只跑一次。
每个 (K, seed)：完整模型（run.py）＋ 对照（run_baselines.py；学习型对照和 K=0 时在支持集上选好的设置一致，只重训）。
台阶表（K=0、K=1 的 seed 1）：
    L0 ModernNCA 原版（对照 modernnca_raw）
    L1 ＋① 成对证据：只留参照一路（距离方向、跨路、学出的分界、同社区），只用本区规则已知训练，不给政策
    L2 ＋② 学校当候选：加上学校一路（到校距离方向、2 公里内）
    L3 ＋③ 政策证据：四至状态、排除、名单点名、名字、整社区点名
    L4 ＋④ 用别区答案训练 ＝ 完整模型
输出：outputs/main_<tag>/k<K>_s<seed>/（模型）、…__baselines/（对照）、…__L1|L2|L3/（台阶）、summary.json、summary.md。
"""
import sys, json, copy, argparse, subprocess
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
import numpy as np
PY = sys.executable
KS = [0, 1, 3, 5]; SEEDS = [1, 2, 3]; LADDER_K = [0, 1]
BASELINES = ['nearest_school', 'copy_nearest', 'label_propagation', 'protonet_raw', 'protonet_policy', 'modernnca_raw', 'modernnca_policy']
NONPOLICY_PAIR = ['distance', 'crossing', 'within_2km', 'learned_boundary', 'same_community']; NONPOLICY_SCHOOL = ['distance', 'within_2km']
T0 = dict(target_known=True, other_known=False, other_answers=False)


def ladder_cfg(base, rung):
    c = copy.deepcopy(base); ev = c['evidence']; tr = c['train']
    if rung in ('L1', 'L2'):
        ev['pair_blocks'] = NONPOLICY_PAIR; ev['school_blocks'] = NONPOLICY_SCHOOL; ev['exclusion_to_model'] = 'none'
    if rung == 'L1': c['model']['paths'] = ['pair']
    tr['items'] = dict(T0)
    return c


def runs(base, tag):
    out = []
    for k in KS:
        for s in (SEEDS[:1] if k == 0 else SEEDS):
            c = copy.deepcopy(base); c['references'] = dict(k=k, seed=s); name = f'k{k}_s{s}'; c['output_dir'] = f'outputs/main_{tag}/{name}'
            c['说明'] = f'主表（{tag}）：每校额外核验 {k} 个参照，核验对象第 {s} 批'; out.append((name, k, s, c))
    return out


def sh(args, log):
    with open(log, 'w', encoding='utf-8') as f:
        r = subprocess.run([PY, '-X', 'utf8', *map(str, args)], cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    assert r.returncode == 0, f'失败：{args}（见 {log}）'


def execute(base, tag):
    """同一个 (K, seed) 里对照与模型并行跑；K=0 的对照选好设置后，K>0 的学习型对照沿用。"""
    root = REPO / f'outputs/main_{tag}'; (root / 'configs').mkdir(parents=True, exist_ok=True); sel = None
    for name, k, s, c in runs(base, tag):
        jobs = [(name, c)] + ([(f'{name}__{r}', dict(ladder_cfg(c, r), output_dir=c['output_dir'] + f'__{r}')) for r in ('L1', 'L2', 'L3')] if (k in LADDER_K and s == SEEDS[0]) else [])
        for jn, jc in jobs: (root / 'configs' / f'{jn}.json').write_text(json.dumps(jc, ensure_ascii=False, indent=2), encoding='utf-8')
        p = root / 'configs' / f'{name}.json'; bdir = REPO / (c['output_dir'] + '__baselines'); bp = None
        if not all((bdir / b / 'eval.json').exists() for b in BASELINES):
            print('对照', name, flush=True); f = open(root / f'{name}__baselines.log', 'w', encoding='utf-8')
            bp = (subprocess.Popen([PY, '-X', 'utf8', 'scripts/run_baselines.py', str(p)] + (['--select-from', sel] if sel else []), cwd=REPO, stdout=f, stderr=subprocess.STDOUT), f)
        for jn, jc in jobs:
            if not (REPO / jc['output_dir'] / 'eval.json').exists(): print('模型', jn, flush=True); sh(['scripts/run.py', root / 'configs' / f'{jn}.json'], root / f'{jn}.log')
        if bp:
            rc = bp[0].wait(); bp[1].close(); assert rc == 0, f'对照失败：{name}'
        if k == 0: sel = str(bdir)


def report(base, tag):
    from zoning import rules, evaluate, references
    from run import load_districts
    D = load_districts(base); R0 = {d: rules.run(D[d], base['rules']['preset'], base['rules'].get('hangzhou_names', 'all'), base['rules'].get('hangzhou_boundary', 'v1'), base['rules'].get('hangzhou_community_rule', False)) for d in D}
    ev = base['evaluate']; res = {}
    for name, k, s, c in runs(base, tag):
        R, picked = references.apply(c['references'], D, R0)
        def rows_of(odir):
            rr = []
            for d in D:
                f = REPO / odir / 'probs' / f'{d}.npz'
                if not f.exists(): return None
                rr += evaluate.correct(np.load(f)['probs'].mean(0), D[d], evaluate.eval_rows(D[d], R[d].known))
            return rr
        model = rows_of(c['output_dir'])
        if model is None: continue
        meth = {'模型': model}
        for b in BASELINES: meth[b] = rows_of(c['output_dir'] + f'__baselines/{b}')
        for r in ('L1', 'L2', 'L3'): meth[r] = rows_of(c['output_dir'] + f'__{r}')
        meth = {m: v for m, v in meth.items() if v is not None}
        res[name] = dict(k=k, seed=s, verified_points=sum(v['points'] for v in picked.values()) if picked else 0, n_eval=len(model),
                         accuracy={m: evaluate.summary(v) for m, v in meth.items()},
                         model_minus={m: evaluate.compare(model, v, ev) for m, v in meth.items() if m != '模型'},
                         ladder={f'{a}->{b}': evaluate.compare(meth[b], meth[a], ev) for a, b in [('modernnca_raw', 'L1'), ('L1', 'L2'), ('L2', 'L3'), ('L3', '模型')] if a in meth and b in meth})
    root = REPO / f'outputs/main_{tag}'; (root / 'summary.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    L = [f'# 主表（{tag}）', '', '准确率：规则和核验都没判出、有官方答案的小区（模型部分）。K>0 为三批核验对象的平均。', '']
    names = ['模型'] + BASELINES
    L.append('| 方法 | ' + ' | '.join(f'K={k}' for k in KS) + ' |'); L.append('|' + '---|' * (len(KS) + 1))
    for m in names:
        cells = []
        for k in KS:
            v = [r['accuracy'][m]['两城']['accuracy'] for r in res.values() if r['k'] == k and m in r['accuracy']]
            cells.append(f'{np.mean(v):.2f}' if v else '—')
        L.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    L += ['', '## 模型 − 对照（两城，每批一行：差值 [95% 区间]）', '']
    for name, r in res.items():
        L.append(f"- {name}（评价 {r['n_eval']}，核验 {r['verified_points']} 点）：" + '；'.join(f"{m} {x['两城']['delta']:+.2f} [{x['两城']['ci'][0]:+.2f}, {x['两城']['ci'][1]:+.2f}]" for m, x in r['model_minus'].items()))
    L += ['', '## 台阶表（两城，后一级 − 前一级）', '']
    for name, r in res.items():
        if r['ladder']: L.append(f'- {name}：' + '；'.join(f"{s_} {x['两城']['delta']:+.2f} [{x['两城']['ci'][0]:+.2f}, {x['两城']['ci'][1]:+.2f}]" for s_, x in r['ladder'].items()))
    (root / 'summary.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--tag', required=True); ap.add_argument('--report', action='store_true')
    a = ap.parse_args(); base = json.loads(Path(a.config).read_text(encoding='utf-8'))
    if not a.report: execute(base, a.tag)
    report(base, a.tag)


if __name__ == '__main__':
    main()
