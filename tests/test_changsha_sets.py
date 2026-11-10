import json
import copy
import hashlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from zoning import changsha_sets


class ChangshaSetTests(unittest.TestCase):
    def source_fixture(self):
        obs = dict(city='长沙', district='测试区', n=2, school_n=3,
                   names=['甲住宅', '甲商业'], school_names=['A', 'B', 'C'], name_known={},
                   clauses=[dict(school=0, clause=1, text='甲住宅'), dict(school=1, clause=2, text='甲住宅')])
        raw = json.dumps(obs).encode()
        relation = dict(q=0, name='甲住宅', status='approved', schools=[0, 1], school_names=['A', 'B'],
                        relation_type='policy_named_overlap', choice_status='unconfirmed',
                        output_action='replace_with_policy_set', identity_basis='核验', evidence=[{'url': 'test'}],
                        applicability='仅住宅', clauses=obs['clauses'])
        payload = dict(schema='changsha_verified_relations_v1', district='测试区', policy_year=2026,
                       observations_sha256=hashlib.sha256(raw).hexdigest(),
                       records=[relation, dict(q=1, name='甲商业', status='pending')])
        return obs, raw, payload

    def read_sources(self, raw, payload):
        allowed = {'observations.json': raw, 'verified_multischool.json': json.dumps(payload).encode()}
        def reader(path):
            self.assertIn(path.name, allowed, '不允许读取Gold或模型结果')
            return allowed[path.name]
        return reader

    def test_verified_output_replaces_wrong_top1_without_propagating(self):
        obs, raw, payload = self.source_fixture()
        dist = SimpleNamespace(city='长沙', obs_dir=Path('source_only'), n=2,
                               site_names=obs['names'], school_names=obs['school_names'])
        rules = SimpleNamespace(known={}, kind={})
        from zoning import predict
        import numpy as np
        probabilities = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
        original = probabilities.copy()
        with patch.object(Path, 'exists', return_value=True), patch.object(
                Path, 'read_bytes', autospec=True, side_effect=self.read_sources(raw, payload)):
            rows = predict.answers(dist, rules, probabilities)
        self.assertEqual(rows[0]['schools'], ['A', 'B'])
        self.assertEqual(rows[0]['confidence'], '')
        self.assertEqual(rows[0]['choice_status'], '未确认任选资格')
        self.assertEqual(rows[1]['schools'], ['A'])  # 同一项目未决商业记录不扩校
        self.assertEqual(rows[1]['source'], '模型')
        self.assertEqual(rules.known, {})
        np.testing.assert_array_equal(probabilities, original)

    def test_source_drift_and_unsupported_links_are_rejected(self):
        obs, raw, original = self.source_fixture()
        mutations = [lambda p: p.update(observations_sha256='wrong'),
                     lambda p: p.update(policy_year=2027),
                     lambda p: p['records'][0].update(name='另一小区'),
                     lambda p: p['records'][0].update(school_names=['A', 'C']),
                     lambda p: p['records'][0].update(clauses=[dict(school=0, clause=1, text='编造条款')]),
                     lambda p: p['records'][0].update(status='unchecked'),
                     lambda p: p['records'].append(copy.deepcopy(p['records'][0]))]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                payload = copy.deepcopy(original); mutate(payload)
                with patch.object(Path, 'exists', return_value=True), patch.object(
                        Path, 'read_bytes', autospec=True, side_effect=self.read_sources(raw, payload)):
                    with self.assertRaises(ValueError):
                        changsha_sets.input_markers(Path('source_only'))

    def test_existing_rule_conflict_is_not_silently_overwritten(self):
        obs, raw, payload = self.source_fixture()
        dist = SimpleNamespace(city='长沙', obs_dir=Path('source_only'),
                               site_names=obs['names'], school_names=obs['school_names'])
        with patch.object(Path, 'exists', return_value=True), patch.object(
                Path, 'read_bytes', autospec=True, side_effect=self.read_sources(raw, payload)):
            with self.assertRaises(ValueError):
                changsha_sets.annotate_output(dist, SimpleNamespace(known={0: [2]}),
                                               [dict(q=0, name='甲住宅', schools=['C'])])

    def dist(self, truth):
        return SimpleNamespace(city='长沙', name='测试区', scope=None,
                               school_names=['A', 'B', 'C'], school_n=3,
                               site_names=['小区'], answers={0: truth})

    def score(self, truth, pred, source='规则判出（名单）'):
        rows = changsha_sets.score_rows(self.dist(truth),
                                       [dict(q=0, schools=pred, source=source)])
        return rows[0], changsha_sets.report(rows)

    def test_partial_hit_is_not_complete_dual_school(self):
        r, report = self.score([0, 1], ['A'])
        self.assertTrue(r['any_hit'])
        self.assertTrue(r['legacy_ok'])
        self.assertFalse(r['exact'])
        self.assertEqual((r['tp'], r['fp'], r['fn']), (1, 0, 1))
        self.assertEqual(report['overall']['school_micro_recall'], 50)

    def test_extra_school_counts_as_false_positive(self):
        r, report = self.score([0], ['A', 'B'])
        self.assertTrue(r['legacy_ok'])
        self.assertFalse(r['exact'])
        self.assertEqual((r['tp'], r['fp'], r['fn']), (1, 1, 0))
        self.assertEqual(report['overall']['school_micro_precision'], 50)

    def test_complete_set_is_order_independent(self):
        r, report = self.score([0, 1], ['B', 'A'])
        self.assertTrue(r['exact'])
        self.assertEqual(report['overall']['exact_set_accuracy'], 100)
        self.assertEqual(report['overall']['school_micro_f1'], 100)

    def test_invalid_gold_is_excluded_not_a_negative_index(self):
        r, report = self.score([-1], ['C'])
        self.assertFalse(r['valid'])
        self.assertEqual(report['overall']['evaluated'], 0)
        self.assertEqual(report['overall']['excluded_invalid_truth'], 1)
        self.assertIsNone(report['overall']['exact_set_accuracy'])
        self.assertEqual(report['school_metrics'], [])

    def test_hangzhou_is_not_annotated_or_scored(self):
        dist = SimpleNamespace(city='杭州', scope={0})
        rows = [dict(q=0, schools=['A', 'B'])]
        self.assertIs(changsha_sets.annotate_output(dist, None, rows), rows)
        with self.assertRaises(ValueError):
            changsha_sets.score_rows(dist, rows)

    def test_markers_require_no_gold_and_do_not_invent_choice_right(self):
        folder = Path('policy_input_only')
        with patch.object(Path, 'read_bytes', autospec=True) as read_bytes:
            obs = dict(city='长沙', district='测试区', n=1, school_n=2,
                       names=['甲小区'], school_names=['A', 'B'], name_known={'0': [0, 1]},
                       clauses=[dict(school=0, clause=1, text='甲小区'),
                                dict(school=1, clause=1, text='甲小区')])
            def source_only(path):
                self.assertEqual(path, folder/'observations.json')
                return json.dumps(obs, ensure_ascii=False).encode('utf-8')
            read_bytes.side_effect = source_only
            marker = changsha_sets.input_markers(folder)['records'][0]
            self.assertEqual(marker['schools'], [0, 1])
            self.assertEqual(marker['choice_status'], 'unconfirmed')
            self.assertEqual(marker['relation_type'], 'policy_named_overlap')
            dist = SimpleNamespace(city='长沙', obs_dir=folder)
            rows = [dict(q=0, schools=['A'])]
            # 多校输入未被规则接纳时，不将模型单校输出扩成双校。
            changsha_sets.annotate_output(dist, SimpleNamespace(known={}), rows)
            self.assertEqual(rows, [dict(q=0, schools=['A'])])


if __name__ == '__main__':
    unittest.main()
