"""长沙：显式保留政策多校点名，并按完整学校集合计分。

输入标记只读取 observations 和经来源核验的逐小区关系，不读答案。
重复点名不等于已确认家庭任选。不扩展学校对，不修改规则、参照或训练标签。长沙 scope=None，沿用当前
地图归属任务的学校全集；这不是对实际家庭招生资格的判定。
"""
import hashlib
import json
from collections import Counter


def verified_relations(obs_dir, obs, source_sha256):
    """加载人工来源核验记录；输入漂移时拒绝应用，未决记录不参与预测。"""
    path = obs_dir / 'verified_multischool.json'
    if not path.exists():
        return [], [], None
    raw = path.read_bytes(); payload = json.loads(raw)
    if (payload['schema'] != 'changsha_verified_relations_v1'
            or payload['district'] != obs['district']
            or payload['observations_sha256'] != source_sha256
            or payload['policy_year'] != 2026):
        raise ValueError('长沙多校关系的版本、区域或政策观察哈希不匹配，须重新核验')
    approved, pending, seen = [], [], set()
    clauses = {(c['school'], c['clause'], c['text']) for c in obs['clauses']}
    for r in payload['records']:
        q = r['q']
        if q in seen or not 0 <= q < obs['n'] or obs['names'][q] != r['name']:
            raise ValueError('多校关系小区身份不匹配或重复')
        seen.add(q)
        if r['status'] == 'pending':
            pending.append(r)
            continue
        if (r['status'] != 'approved' or r['relation_type'] != 'policy_named_overlap'
                or r['choice_status'] != 'unconfirmed'
                or r['output_action'] != 'replace_with_policy_set'
                or not r['identity_basis'] or not r['evidence'] or not r['applicability']):
            raise ValueError('多校关系缺少核验或适用范围')
        ss = r['schools']
        if len(ss) < 2 or len(set(ss)) != len(ss) or any(s < 0 or s >= obs['school_n'] for s in ss):
            raise ValueError('无效的多校关系学校集合')
        if r['school_names'] != [obs['school_names'][s] for s in ss]:
            raise ValueError('多校关系学校名称不匹配')
        if set(ss) != {c['school'] for c in r['clauses']} or any(
                (c['school'], c['clause'], c['text']) not in clauses for c in r['clauses']):
            raise ValueError('多校关系缺少对应政策原文')
        approved.append(r)
    return approved, pending, hashlib.sha256(raw).hexdigest()


def input_markers(obs_dir):
    """只读既有政策观察，将多校 name_known 显式导出为逐小区记录。"""
    path = obs_dir / 'observations.json'
    raw = path.read_bytes(); obs = json.loads(raw)
    if obs['city'] != '长沙':
        raise ValueError('长沙多校标记不适用于其他城市')
    records = []
    for q0, schools in sorted(obs['name_known'].items(), key=lambda x: int(x[0])):
        q = int(q0); ss = sorted(set(schools))
        if len(ss) < 2:
            continue
        if not 0 <= q < obs['n'] or any(s < 0 or s >= obs['school_n'] for s in ss):
            raise ValueError('政策观察中的小区或学校编号越界')
        records.append(dict(
            q=q, name=obs['names'][q], schools=ss,
            school_names=[obs['school_names'][s] for s in ss],
            relation_type='policy_named_overlap', choice_status='unconfirmed',
            applicability='仅当前政策点名的小区；不扩展到两校其他小区',
            identity_basis='沿用当前输入的名称映射；本标记不新增别名认定',
            clauses=[dict(school=c['school'], clause=c['clause'], text=c['text'])
                     for c in obs['clauses'] if c['school'] in ss],
        ))
    source_sha256 = hashlib.sha256(raw).hexdigest()
    approved, pending, relation_sha = verified_relations(obs_dir, obs, source_sha256)
    by_q = {r['q']: r for r in records}
    for r in approved:
        if r['q'] in by_q and set(by_q[r['q']]['schools']) != set(r['schools']):
            raise ValueError('核验关系与已有多校点名冲突')
        by_q[r['q']] = r
    return dict(schema='changsha_policy_multischool_v2', district=obs['district'],
                source='observations/observations.json',
                source_sha256=source_sha256, verified_relations_sha256=relation_sha,
                records=[by_q[q] for q in sorted(by_q)], pending=pending)


def annotate_output(dist, rules, rows):
    """已有规则补注释；仅对身份和范围核验通过的指定小区输出政策集合。"""
    if dist.city != '长沙':
        return rows
    markers = {r['q']: r for r in input_markers(dist.obs_dir)['records']}
    for r in rows:
        q = r['q']; known = rules.known.get(q, [])
        marker = markers.get(q)
        if marker and marker.get('output_action') == 'replace_with_policy_set':
            if (r['name'] != marker['name'] or dist.site_names[q] != marker['name']
                    or [dist.school_names[s] for s in marker['schools']] != marker['school_names']):
                raise ValueError('输出与核验关系的小区或学校身份不一致')
            if known and not set(known).issubset(marker['schools']):
                raise ValueError('核验关系与现有规则冲突，须人工复核')
            r.update(schools=list(marker['school_names']), source='政策核验（逐小区多校）',
                     confidence='', set_basis='政策重复点名＋实体核验',
                     choice_status='未确认任选资格', policy_school_ids=list(marker['schools']),
                     policy_applicability=marker['applicability'], policy_year=2026)
            continue
        if len(known) < 2:
            continue
        matched = marker is not None and set(marker['schools']) == set(known)
        r['set_basis'] = '政策重复点名' if matched else '既有规则多校集合'
        r['choice_status'] = '未确认任选资格'
        r['policy_school_ids'] = list(known)
    return rows


def score_rows(dist, predictions):
    """预测已冻结后才读取Gold；非法Gold单列，不能将-1当作学校。"""
    if dist.city != '长沙' or dist.scope is not None:
        raise ValueError('该完整集合口径仅适用于当前长沙学校全集')
    names = {n: i for i, n in enumerate(dist.school_names)}
    by_q = {int(r['q']): r for r in predictions}
    if len(by_q) != len(predictions):
        raise ValueError('重复预测小区编号')
    out = []
    for q, truth in sorted(dist.answers.items()):
        r = by_q[q]; pred = {names[n] for n in r['schools']}
        if not pred:
            raise ValueError('预测学校集合不能为空')
        base = dict(district=dist.name, q=q, name=dist.site_names[q],
                    truth=list(truth), pred=sorted(pred), source=r['source'])
        if not truth or any(s < 0 or s >= dist.school_n for s in truth):
            out.append(dict(**base, valid=False, exclusion='invalid_truth_school_id'))
            continue
        target = set(truth)
        if len(target) != len(truth):
            raise ValueError('Gold集合含重复学校编号')
        hit = bool(pred & target); exact = pred == target
        out.append(dict(**base, valid=True, exact=exact, any_hit=hit,
                        legacy_ok=hit if r['source'].startswith('规则') else exact,
                        tp=len(pred & target), fp=len(pred - target), fn=len(target - pred)))
    return out


def _ratio(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else None


def metrics(rows):
    valid = [r for r in rows if r['valid']]
    tp = sum(r['tp'] for r in valid); fp = sum(r['fp'] for r in valid); fn = sum(r['fn'] for r in valid)
    exact = sum(r['exact'] for r in valid)
    return dict(total_gold=len(rows), evaluated=len(valid), excluded_invalid_truth=len(rows)-len(valid),
                exact_set_correct=exact, exact_set_accuracy=_ratio(exact, len(valid)),
                any_hit_accuracy=_ratio(sum(r['any_hit'] for r in valid), len(valid)),
                legacy_accuracy_same_valid_rows=_ratio(sum(r['legacy_ok'] for r in valid), len(valid)),
                school_tp=tp, school_fp=fp, school_fn=fn,
                school_micro_precision=_ratio(tp, tp+fp), school_micro_recall=_ratio(tp, tp+fn),
                school_micro_f1=_ratio(2*tp, 2*tp+fp+fn))


def report(rows):
    """严格集合为主指标；同时给旧口径对照和学校级查准/查全。"""
    valid = [r for r in rows if r['valid']]
    by_school = {}
    for r in valid:
        pred, truth = set(r['pred']), set(r['truth'])
        for s in pred | truth:
            c = by_school.setdefault((r['district'], s), Counter())
            c['tp'] += s in pred and s in truth
            c['fp'] += s in pred and s not in truth
            c['fn'] += s not in pred and s in truth
    school_rows = [dict(district=d, school=s, **c,
                        precision=_ratio(c['tp'], c['tp']+c['fp']),
                        recall=_ratio(c['tp'], c['tp']+c['fn']))
                   for (d, s), c in sorted(by_school.items())]
    return dict(
        primary_metric='exact_set_accuracy',
        task='当前长沙地图归属集合；不是家庭实际任选资格',
        overall=metrics(rows),
        by_district={d: metrics([r for r in rows if r['district'] == d])
                     for d in sorted({r['district'] for r in rows})},
        singleton_gold=metrics([r for r in valid if len(r['truth']) == 1]),
        multischool_gold=metrics([r for r in valid if len(r['truth']) > 1]),
        school_metrics=school_rows,
        excluded=[r for r in rows if not r['valid']],
        multischool_cases=[r for r in valid if len(r['truth']) > 1 or len(r['pred']) > 1],
    )
