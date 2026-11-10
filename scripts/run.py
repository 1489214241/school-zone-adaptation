"""一个入口跑完全流程：规则 → 证据 → 训练 → 评价 → 输出。
  python scripts/run.py                               # 用 configs/default.json
  python scripts/run.py configs/final_20261103.json   # 两城最终版设置
  python scripts/run.py configs/default.json --compare outputs/final_20261103   # 另与一次已跑完的结果比较
  python scripts/run.py configs/default.json --eval-only                        # 不训练，用 output_dir 里已有的概率重新评价
输出到 config 的 output_dir：probs/<区>.npz（各种子概率）、eval.json（模型部分 summary、整体 overall）、answers/<区>.csv（每个小区的答案）、config.json（本次设置）。
"""
import sys, json, time, argparse
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
import numpy as np
from zoning import data, rules, evidence, train, evaluate, predict, references, changsha_sets


def load_districts(cfg):
    D = {d: data.load(d) for d in cfg['districts']}
    if cfg.get('data', {}).get('hangzhou_gold', 'map') == 'map_list': D = {d: data.with_list_gold(x) for d, x in D.items()}
    if cfg.get('data', {}).get('hangzhou_community') == 'mca': D = {d: data.with_mca_community(x) for d, x in D.items()}
    return D


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config', nargs='?', default=str(REPO / 'configs/default.json')); ap.add_argument('--compare'); ap.add_argument('--eval-only', action='store_true')
    a = ap.parse_args(); cfg = json.loads(Path(a.config).read_text(encoding='utf-8')); out = REPO / cfg['output_dir']; out.mkdir(parents=True, exist_ok=True)
    if not a.eval_only: (out / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    t0 = time.time(); D = load_districts(cfg)
    R = {d: rules.run(D[d], cfg['rules']['preset'], cfg['rules'].get('hangzhou_names', 'all'), cfg['rules'].get('hangzhou_boundary', 'v1'), cfg['rules'].get('hangzhou_community_rule', False)) for d in D}                                   # 第一、二步
    print('规则', {d: len(r.known) for d, r in R.items()}, f'{time.time() - t0:.0f}s', flush=True)
    R, picked = references.apply(cfg.get('references'), D, R)                                     # 每校额外核验 K 个参照（zoning/references.py）；不设则与原来逐位相同
    if picked:
        (out / 'references.json').write_text(json.dumps(picked, ensure_ascii=False, indent=1), encoding='utf-8')
        print('核验参照（学校、楼盘组、点）', {d: (v['schools'], v['groups'], v['points']) for d, v in picked.items()}, flush=True)
    if not a.eval_only:
        X = {d: evidence.build(D[d], R[d], cfg['evidence']) for d in D}                           # 第三步
        print('证据', {d: (x.pair.shape[1], x.np_, x.na) for d, x in X.items()}, f'{time.time() - t0:.0f}s', flush=True)
    rows, whole, changsha_rows = [], [], []
    for d in D:                                                                                    # 第四、五步
        if a.eval_only: P = np.load(out / 'probs' / f'{d}.npz')['probs']
        else:
            res = train.fit(d, X, D, cfg, cfg['train']['seeds'])
            P = np.stack([res[s][1] for s in cfg['train']['seeds']]); (out / 'probs').mkdir(exist_ok=True); np.savez_compressed(out / 'probs' / f'{d}.npz', probs=P, seeds=cfg['train']['seeds'])
            predict.write_csv(predict.answers(D[d], R[d], P.mean(0)), out / 'answers' / f'{d}.csv')   # 第六步
            print('训练', d, f'{time.time() - t0:.0f}s', flush=True)
        rows += evaluate.correct(P.mean(0), D[d], evaluate.eval_rows(D[d], R[d].known))            # 评价：模型部分
        whole += evaluate.overall(P.mean(0), D[d], R[d])                                           # 评价：整体
        if D[d].city == '长沙':
            frozen = predict.answers(D[d], R[d], P.mean(0))
            changsha_rows += changsha_sets.score_rows(D[d], frozen)
            folder = out / 'policy_input'; folder.mkdir(exist_ok=True)
            (folder / f'{d}.json').write_text(json.dumps(changsha_sets.input_markers(D[d].obs_dir), ensure_ascii=False, indent=2), encoding='utf-8')
    res = dict(summary=evaluate.summary(rows), overall=evaluate.summary(whole))
    if changsha_rows:
        res['changsha_sets'] = changsha_sets.report(changsha_rows)
        res['metric_notes'] = dict(changsha_primary='changsha_sets：最终输出完整集合评分',
                                   summary_and_overall='历史模型/规则口径，不含长沙逐小区核验覆盖，不能代表修正后的输出')
    if a.compare:
        ref = REPO / a.compare; rr, ww = [], []
        for d in D:
            Pr = np.load(ref / 'probs' / f'{d}.npz')['probs'].mean(0)
            rr += evaluate.correct(Pr, D[d], evaluate.eval_rows(D[d], R[d].known)); ww += evaluate.overall(Pr, D[d], R[d])
        res['compare'] = dict(reference=a.compare, delta=evaluate.compare(rows, rr, cfg['evaluate']), overall_delta=evaluate.compare(whole, ww, cfg['evaluate']))
    (out / ('eval_recheck.json' if a.eval_only else 'eval.json')).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
