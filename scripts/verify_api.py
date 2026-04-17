# -*- coding: utf-8 -*-
"""端到端验证：后端三条关键路径"""
import requests, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = 'http://127.0.0.1:8901'

print('=== 1) pipeline_status ===')
s = requests.get(f'{BASE}/api/pipeline_status').json()
print(f"  ready={s['ready']}  backend={s['backend']}  ft_available={s['fine_tuned_available']}")

print('\n=== 2) pipeline_match 单条走 L3 LLM ===')
r = requests.post(f'{BASE}/api/pipeline_match',
                  json={'input_title': '做用户研究和竞品分析的人',
                        'context': '在产品部门，用Axure画原型'}).json()
print(f"  path={r['path']}  level={r['level']}  "
      f"standard_title={r['standard_title']}  conf={r['confidence']}")

print('\n=== 3) pipeline_report 离线报表 ===')
d = requests.get(f'{BASE}/api/pipeline_report').json()
print(f"  total={d['total']}")
print(f"  纯 KG 准确率: {d['kg_only']['accuracy']*100:.2f}%  "
      f"({d['kg_only']['correct']}/{d['total']})")
print(f"  混合链路准确率: {d['hybrid']['accuracy']*100:.2f}%  "
      f"({d['hybrid']['correct']}/{d['total']})")
print(f"  提升: +{d['uplift_points']*100:.2f} pp")
print(f"  LLM 救回: {d['llm_rescue_count']}  LLM 误判: {d['llm_wrong_count']}")
print(f"  来源分布: {d['sources']}")
print('  按目标层级:')
for k, v in d['by_target_level'].items():
    print(f"    {k:16s} 样本={v['total']:3d} 纯KG={v['kg_acc']*100:5.1f}% "
          f"混合={v['final_acc']*100:5.1f}%")

print('\n=== 4) kg_expand_candidates 词表扩充候选 ===')
c = requests.get(f'{BASE}/api/kg_expand_candidates').json()
print(f"  候选条数: {c['total']}")
for row in c['rows'][:3]:
    print(f"    - {row['candidate_variant']} → {row['suggested_standard']}")

print('\n=== 5) llm_failures 失败聚合 ===')
f = requests.get(f'{BASE}/api/llm_failures').json()
print(f"  失败分组数: {f['total_groups']}")
for row in f['rows'][:3]:
    print(f"    - {row['expected_standard']}: {row['fail_count']} 失败 "
          f"(错答={row['top_llm_mistakes'][:40]})")

print('\n✓ 所有接口通路均正常')
