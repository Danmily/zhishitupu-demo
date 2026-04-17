# -*- coding: utf-8 -*-
import json, sys, io
from urllib.request import urlopen, Request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = "http://127.0.0.1:8901"

# 1. Stats
r = json.loads(urlopen(f"{BASE}/api/stats").read().decode())
kg = r["kg"]
print(f"=== Stats ===")
print(f"KG: {kg['standard_titles']} titles, {kg['total_edges']} edges, {kg['cross_references']} x-refs")
print(f"Tests: {r['tests_total']}")

# 2. Single match
def match(title, ctx=""):
    data = json.dumps({"input_title": title, "context": ctx}).encode("utf-8")
    req = Request(f"{BASE}/api/match", data=data, headers={"Content-Type": "application/json"})
    return json.loads(urlopen(req).read().decode())

for title, ctx in [
    ("AI产品负责人", "字节跳动，带10人团队"),
    ("CDO", "Chief Data Officer，管数据部门"),
    ("用户增长", "找增长方向的人"),
    ("CTO", "创业公司联合创始人"),
    ("前端工程师", "3年React经验"),
]:
    r = match(title, ctx)
    print(f"\n'{title}' => {r['standard_title']} | {r['level']} | conf={r.get('confidence')}")
    for step in r.get("trace", []):
        print(f"  trace: {step}")

# 3. Batch eval
print("\n=== Batch Eval (all cases, instant) ===")
r = json.loads(urlopen(f"{BASE}/api/batch_eval").read().decode())
print(f"Overall: {r['correct']}/{r['total']} = {r['accuracy']*100:.1f}%")
print("By Level:")
for lv, info in r["by_level"].items():
    print(f"  {lv}: {info['correct']}/{info['total']} = {info['accuracy']*100:.1f}%")
print("By Difficulty:")
for d in ["easy", "medium", "hard"]:
    info = r["by_difficulty"].get(d, {})
    print(f"  {d}: {info.get('correct',0)}/{info.get('total',0)} = {info.get('accuracy',0)*100:.1f}%")

failures = [d for d in r["details"] if not d["ok"]]
if failures:
    print(f"\nFailed ({len(failures)}):")
    for f in failures:
        print(f"  {f['test_id']} '{f['input_title']}' => got '{f['got']}' (expected '{f['expected']}') [{f['level']}]")
