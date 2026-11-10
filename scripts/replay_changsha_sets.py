"""仅重放长沙已存概率，导出多校输入标记、带依据输出和集合主评分；不训练。

从代码库目录运行：python scripts/replay_changsha_sets.py
输出到单独目录，不覆盖原输出。Gold和规则输入只读。
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
from zoning import data, rules, predict, changsha_sets


def read(path):
    return json.loads(path.read_text('utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-output', default=None)
    ap.add_argument('--output', default='outputs/changsha_verified_sets_20261109')
    a = ap.parse_args(); cfg = read(REPO/'configs/default.json')
    source = (REPO/(a.source_output or cfg['output_dir'])).resolve()
    out = (REPO/a.output).resolve()
    if source == out or out in source.parents or source in out.parents:
        raise ValueError('新输出目录必须与原输出目录分开，不能相互包含')
    if REPO not in out.parents:
        raise ValueError('输出目录必须位于当前代码库内')
    saved = read(source/'config.json')
    for key in ('rules', 'evidence', 'model', 'train', 'data'):
        if saved.get(key) != cfg.get(key):
            raise ValueError(f'冻结输出与当前配置不一致：{key}')
    districts = [d for d in cfg['districts'] if d in data.CITIES['长沙']]
    files = [REPO/'configs/default.json', source/'config.json']
    for d in cfg['districts']:
        files.extend([REPO/'data'/d/'district.json', REPO/'data'/d/'answers.json',
                      REPO/'data'/d/'observations/observations.json',
                      source/'probs'/f'{d}.npz', source/'answers'/f'{d}.csv'])
        relation = REPO/'data'/d/'observations/verified_multischool.json'
        if relation.exists(): files.append(relation)
    hashes = {str(p.relative_to(REPO)): sha(p) for p in files}
    # 标记先从政策观察生成，不读取答案文件或答案审计报告。
    markers = {d: changsha_sets.input_markers(REPO/'data'/d/'observations') for d in districts}
    rows, baseline_rows, changes, output_counts = [], [], [], {}
    out.mkdir(parents=True, exist_ok=True)
    (out/'policy_input').mkdir(exist_ok=True)
    for d in districts:
        dist = data.load(d)
        r = rules.run(dist, cfg['rules']['preset'], cfg['rules'].get('hangzhou_names', 'all'),
                      cfg['rules'].get('hangzhou_boundary', 'v1'), cfg['rules'].get('hangzhou_community_rule', False))
        with np.load(source/'probs'/f'{d}.npz', allow_pickle=False) as z:
            probability = z['probs'].mean(0)
        pred = predict.answers(dist, r, probability)
        with (source/'answers'/f'{d}.csv').open(encoding='utf-8-sig', newline='') as f:
            baseline_pred = [dict(q=int(x['小区编号']), name=x['小区'], schools=x['学校'].split('、'),
                                  source=x['来源'], confidence=x['把握']) for x in csv.DictReader(f)]
        baseline = {x['q']: set(x['schools']) for x in baseline_pred}
        approved = {x['q']: x for x in markers[d]['records'] if x.get('output_action') == 'replace_with_policy_set'}
        if set(baseline) != {x['q'] for x in pred}:
            raise AssertionError(f'{d}预测行集合变化')
        for x in pred:
            if baseline[x['q']] != set(x['schools']):
                marker = approved.get(x['q'])
                if marker is None or set(x['schools']) != set(marker['school_names']):
                    raise AssertionError('出现未经来源核验允许的输出变化')
                changes.append(dict(district=d, q=x['q'], name=x['name'],
                                    before=sorted(baseline[x['q']]), after=x['schools']))
        current_scores = changsha_sets.score_rows(dist, pred)
        previous_scores = changsha_sets.score_rows(dist, baseline_pred)
        rows.extend(current_scores); baseline_rows.extend(previous_scores)
        before = {x['q']: x for x in previous_scores}; after = {x['q']: x for x in current_scores}
        for c in [c for c in changes if c['district'] == d]:
            q = c['q']; valid = q in after and after[q]['valid']
            c.update(evaluated=valid, fixed=valid and not before[q]['exact'] and after[q]['exact'],
                     harmed=valid and before[q]['exact'] and not after[q]['exact'])
        predict.write_csv(pred, out/'answers'/f'{d}.csv')
        (out/'policy_input'/f'{d}.json').write_text(json.dumps(markers[d], ensure_ascii=False, indent=2), 'utf-8')
        output_counts[d] = dict(predictions=len(pred), policy_multischool=len(markers[d]['records']),
                                verified_relations=len(approved), pending=len(markers[d]['pending']),
                                multischool_outputs=sum(len(x['schools']) > 1 for x in pred))
    result = changsha_sets.report(rows)
    baseline_report = changsha_sets.report(baseline_rows)
    result['baseline'] = {k: baseline_report[k] for k in ('overall', 'by_district', 'singleton_gold', 'multischool_gold')}
    result['comparison'] = dict(changed=len(changes), fixed=sum(x['fixed'] for x in changes),
                                harmed=sum(x['harmed'] for x in changes),
                                delta_exact_pp=result['overall']['exact_set_accuracy']-baseline_report['overall']['exact_set_accuracy'])
    # 杭州只核对全部输出字段，不写入杭州结果目录。
    hz_check = {}
    for d in [d for d in cfg['districts'] if d in data.CITIES['杭州']]:
        dist = data.load(d)
        if cfg.get('data', {}).get('hangzhou_gold') == 'map_list': dist = data.with_list_gold(dist)
        if cfg.get('data', {}).get('hangzhou_community') == 'mca': dist = data.with_mca_community(dist)
        r = rules.run(dist, cfg['rules']['preset'], cfg['rules'].get('hangzhou_names', 'all'),
                      cfg['rules'].get('hangzhou_boundary', 'v1'), cfg['rules'].get('hangzhou_community_rule', False))
        with np.load(source/'probs'/f'{d}.npz', allow_pickle=False) as z:
            pred = predict.answers(dist, r, z['probs'].mean(0))
        with (source/'answers'/f'{d}.csv').open(encoding='utf-8-sig', newline='') as f:
            old = list(csv.DictReader(f))
        actual = [dict(小区编号=str(x['q']), 小区=x['name'], 来源=x['source'], 学校='、'.join(x['schools']), 把握=x['confidence']) for x in pred]
        if actual != old: raise AssertionError(f'杭州输出发生变化：{d}')
        hz_check[d] = dict(rows=len(pred), all_fields_unchanged=True)
    audit = dict(source_output=str(source.relative_to(REPO)), output_counts=output_counts,
                 school_set_changes=changes, input_and_baseline_hashes=hashes,
                 input_and_baseline_unchanged=all(sha(REPO/p) == h for p, h in hashes.items()),
                 trained=False, gold_modified=False, hangzhou_written=False, hangzhou_checks=hz_check,
                 selection='已查看同批答案的错误驱动实体核验；无独立留出验证')
    if not audit['input_and_baseline_unchanged']:
        raise AssertionError('不应修改原始输入、Gold或冻结输出')
    (out/'eval.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
    (out/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), 'utf-8')
    m = result['overall']; dual = result['multischool_gold']
    lines = ['长沙多校标记与完整集合评分', '',
             '本次只调整长沙，使用现存冻结概率，不重训，不修改Gold；实体核验通过的小区使用政策学校集合。',
             f"政策多校输入和多校输出：{sum(x['policy_multischool'] for x in output_counts.values())}条。重复点名标为政策多校记录，未确认家庭任选资格。",
             f"原始Gold {m['total_gold']}条，学校编号无效单列{m['excluded_invalid_truth']}条，有效计分{m['evaluated']}条。",
             f"完整集合正确{m['exact_set_correct']}条，比例{m['exact_set_accuracy']:.4f}%；学校微平均查准{m['school_micro_precision']:.4f}%，查全{m['school_micro_recall']:.4f}%。",
             f"修改前完整集合正确率{baseline_report['overall']['exact_set_accuracy']:.4f}%；修改后{m['exact_set_accuracy']:.4f}%。",
             f"双校Gold {dual['evaluated']}条，其中完整预测{dual['exact_set_correct']}条。",
             f"预测学校集合变化{len(changes)}条，改对{result['comparison']['fixed']}，改坏{result['comparison']['harmed']}；全部变化均须对应核验输入。",
             '候选已在同批Gold中定位；这是输入/输出修正的开发结果，不是模型训练提升或独立验证。', '',
             '文件：policy_input/为仅由政策观察生成的标记；answers/为带多校依据的输出；eval.json包含主评分、逐学校指标和全部多校例外；audit.json记录输入/Gold/冻结输出的哈希核验。', '',
             '多校Gold或多校输出的全部记录（用于评价核对，不回填输入）：']
    for row in result['multischool_cases']:
        dist = data.load(row['district'])
        truth = '、'.join(dist.school_names[s] for s in row['truth'])
        pred_names = '、'.join(dist.school_names[s] for s in row['pred'])
        lines.append(f"{row['district']} q={row['q']} {row['name']} | Gold：{truth} | 输出：{pred_names} | 完整一致：{row['exact']}")
    (out/'调整说明.txt').write_text('\n'.join(lines)+'\n', 'utf-8')
    print(json.dumps(dict(overall=result['overall'], multischool_gold=result['multischool_gold'],
                         output_counts=output_counts, comparison=result['comparison'], hangzhou_checks=hz_check), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
