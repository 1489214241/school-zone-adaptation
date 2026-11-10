"""主表对照：和模型同一份设置（同一批规则结果、同一批核验参照、同一批评价小区），跑 zoning/baselines.py 里的对照方法。
  python scripts/run_baselines.py configs/default.json                         # 全部对照
  python scripts/run_baselines.py <设置> --methods copy_nearest protonet       # 只跑几个
  python scripts/run_baselines.py <设置> --select-from <另一次对照输出目录>     # 学习型对照沿用那次在支持集上选好的设置（如 K=0 时选的）
输出到 <output_dir>__baselines/<方法>/：probs/<区>.npz（与 run.py 同格式，可用 run.py --eval-only --compare 和模型成对比较）、eval.json；
学习型对照另存 chosen.json（选中的设置与训练轮数）。学习型对照：3 个种子（20261181–83），raw＝原版，policy＝加政策排除。
"""
import sys, json, time, argparse
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
import numpy as np
import torch
from zoning import rules, evaluate, references, baselines as B
from run import load_districts

SIMPLE = {'nearest_school': B.nearest_school, 'copy_nearest': B.copy_nearest, 'label_propagation': B.label_propagation}
LEARNED = ['protonet', 'modernnca']
SEEDS = [20261181, 20261182, 20261183]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--methods', nargs='*', default=list(SIMPLE) + LEARNED); ap.add_argument('--select-from')
    a = ap.parse_args(); torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = json.loads(Path(a.config).read_text(encoding='utf-8')); base = REPO / (cfg['output_dir'] + '__baselines'); t0 = time.time()
    D = load_districts(cfg)
    R = {d: rules.run(D[d], cfg['rules']['preset'], cfg['rules'].get('hangzhou_names', 'all'), cfg['rules'].get('hangzhou_boundary', 'v1'), cfg['rules'].get('hangzhou_community_rule', False)) for d in D}
    R, picked = references.apply(cfg.get('references'), D, R)
    for m in a.methods:
        variants = ['raw', 'policy'] if m in LEARNED else ['']
        outs = {v: base / (m + (f'_{v}' if v else '')) for v in variants}; rows = {v: [] for v in variants}; chosen = {}
        prev = json.loads((Path(a.select_from) / f'{m}_raw' / 'chosen.json').read_text(encoding='utf-8')) if (a.select_from and m in LEARNED) else {}
        for d in D:
            if m in SIMPLE:
                Ps = {'': SIMPLE[m](D[d], R[d])}
                if m == 'label_propagation': chosen[d] = B.label_propagation.last_chosen
            else:
                res = B.learned(m, D[d], R[d], SEEDS, prev.get(d)); Ps = {'raw': res['raw'], 'policy': res['policy']}; chosen[d] = dict(res['chosen'], n_support=res['n_support'])
            for v, P in Ps.items():
                (outs[v] / 'probs').mkdir(parents=True, exist_ok=True); np.savez_compressed(outs[v] / 'probs' / f'{d}.npz', probs=P[None].astype(np.float32))
                rows[v] += evaluate.correct(P, D[d], evaluate.eval_rows(D[d], R[d].known))
            print(m, d, f'{time.time() - t0:.0f}s', flush=True)
        for v in variants:
            (outs[v] / 'eval.json').write_text(json.dumps(dict(summary=evaluate.summary(rows[v]), config=a.config, references=cfg.get('references')), ensure_ascii=False, indent=1), encoding='utf-8')
            if chosen: (outs[v] / 'chosen.json').write_text(json.dumps(chosen, ensure_ascii=False, indent=1), encoding='utf-8')
            s = evaluate.summary(rows[v]); print(m, v, {k: round(s[k]['accuracy'], 2) for k in ['两城', '长沙', '杭州'] if k in s}, flush=True)


if __name__ == '__main__':
    main()
