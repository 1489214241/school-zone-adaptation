"""从完整模型逐个去掉一处改动（补 main_table.py 的台阶表：台阶表依次加，结果和顺序有关）。
  python scripts/ablation_table.py configs/default.json --tag pre_v1 [--jobs 2] [--report]
在 K=0（seed 1）和 K=1（seed 1、2、3）上，各跑：
    D1 去掉①成对证据里的非距离部分：参照一路不再用跨路、学出的分界、同社区
    D2 去掉②学校当候选（model.paths = ["pair"]；一个可用参照都没有的小区仍保留学校一路）
    D3 去掉③政策证据：两路证据只留非政策部分，排除不交给模型（同台阶表 L1/L2 的证据设置）
    D4 去掉④别区答案训练：只用本区规则已知训练（＝台阶表 L3；已有 __L3 时直接沿用）
完整模型与评价集沿用 outputs/main_<tag>/k<K>_s<seed>/。输出 …__D1..D4/、ablation.json、ablation.md。
"""
import sys, json, copy, argparse, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
import numpy as np
from main_table import runs, sh, NONPOLICY_PAIR, NONPOLICY_SCHOOL, T0

TARGET = [(0, 1), (1, 1), (1, 2), (1, 3)]
DROPS = {'D1': '去掉①成对证据（跨路、学出的分界、同社区）', 'D2': '去掉②学校当候选', 'D3': '去掉③政策证据', 'D4': '去掉④别区答案训练'}


def drop_cfg(base, d):
    c = copy.deepcopy(base); ev = c['evidence']
    if d == 'D1': ev['pair_blocks'] = [b for b in ev['pair_blocks'] if b not in ('crossing', 'learned_boundary', 'same_community')]
    if d == 'D2': c['model']['paths'] = ['pair']
    if d == 'D3': ev['pair_blocks'] = NONPOLICY_PAIR; ev['school_blocks'] = NONPOLICY_SCHOOL; ev['exclusion_to_model'] = 'none'
    if d == 'D4': c['train']['items'] = dict(T0)
    c['说明'] = c.get('说明', '') + f'；从完整模型{DROPS[d]}'
    return c


def jobs(base, tag):
    out = []
    for name, k, s, c in runs(base, tag):
        if (k, s) not in TARGET: continue
        for d in DROPS:
            odir = c['output_dir'] + f'__{d}'
            if d == 'D4' and (REPO / (c['output_dir'] + '__L3') / 'eval.json').exists(): odir = c['output_dir'] + '__L3'
            out.append((name, k, s, d, dict(drop_cfg(c, d), output_dir=odir)))
    return out


def execute(base, tag, n):
    root = REPO / f'outputs/main_{tag}'; todo = []
    for name, k, s, d, c in jobs(base, tag):
        if (REPO / c['output_dir'] / 'eval.json').exists(): continue
        p = root / 'configs' / f'{name}__{d}.json'; p.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding='utf-8'); todo.append((f'{name}__{d}', p))
    def one(j):
        print('开始', j[0], flush=True); sh(['scripts/run.py', j[1]], root / f'{j[0]}.log'); print('完成', j[0], flush=True)
    with ThreadPoolExecutor(n) as ex: list(ex.map(one, todo))


def report(base, tag):
    from zoning import rules, evaluate, references
    from run import load_districts
    D = load_districts(base); R0 = {d: rules.run(D[d], base['rules']['preset'], base['rules'].get('hangzhou_names', 'all'), base['rules'].get('hangzhou_boundary', 'v1'), base['rules'].get('hangzhou_community_rule', False)) for d in D}
    ev = base['evaluate']; res = {}; cache = {}

    def rows_of(odir, R):
        rr = []
        for d in D:
            f = REPO / odir / 'probs' / f'{d}.npz'
            if not f.exists(): return None
            rr += evaluate.correct(np.load(f)['probs'].mean(0), D[d], evaluate.eval_rows(D[d], R[d].known))
        return rr
    for name, k, s, d, c in jobs(base, tag):
        if name not in cache:
            R, _ = references.apply(c['references'], D, R0); cache[name] = (R, rows_of(c['output_dir'].rsplit('__', 1)[0], R))
        R, full = cache[name]; rows = rows_of(c['output_dir'], R)
        if full is None or rows is None: continue
        res.setdefault(name, dict(k=k, seed=s, drops={}))['drops'][d] = evaluate.compare(full, rows, ev)
    root = REPO / f'outputs/main_{tag}'; (root / 'ablation.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    L = [f'# 从完整模型逐个去掉（{tag}）', '', '完整模型 − 去掉后（正数＝这处改动有用）：差值 [95% 区间]（改对/改坏）', '']
    for name, r in res.items():
        L.append(f'## {name}'); L.append('')
        L.append('| 去掉 | 两城 | 长沙 | 杭州 |'); L.append('|---|---|---|---|')
        for d, x in r['drops'].items():
            L.append(f'| {DROPS[d]} | ' + ' | '.join(f"{x[c_]['delta']:+.2f} [{x[c_]['ci'][0]:+.2f}, {x[c_]['ci'][1]:+.2f}]（{x[c_]['fixed']}/{x[c_]['broken']}）" for c_ in ['两城', '长沙', '杭州']) + ' |')
        L.append('')
    (root / 'ablation.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--tag', required=True); ap.add_argument('--jobs', type=int, default=2); ap.add_argument('--report', action='store_true')
    a = ap.parse_args(); base = json.loads(Path(a.config).read_text(encoding='utf-8'))
    if not a.report: execute(base, a.tag, a.jobs)
    report(base, a.tag)


if __name__ == '__main__':
    main()
