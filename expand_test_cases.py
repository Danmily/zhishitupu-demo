import copy
import json
import random
from pathlib import Path


def main() -> None:
    path = Path(__file__).parent / "test_cases_v2.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    target = 1500
    base = list(data)
    n = len(base)

    seniority_tags = ["高级", "资深", "首席", "初级"]
    context_suffixes = [
        "",
        "，有跨部门协作经验",
        "，熟悉数据分析与复盘",
        "，带过小团队",
        "，负责过从0到1项目",
    ]
    title_suffixes = ["", "（偏策略）", "（偏执行）", "（偏平台）"]

    random.seed(42)
    out = []
    i = 0

    while len(out) < target:
        item = copy.deepcopy(base[i % n])
        item["test_id"] = f"T_{len(out) + 1:04d}"

        if i >= n:
            tag = seniority_tags[(i // n) % len(seniority_tags)]
            title_suffix = title_suffixes[(i // (2 * n)) % len(title_suffixes)]
            context_suffix = context_suffixes[(i // (3 * n)) % len(context_suffixes)]

            if random.random() < 0.75:
                item["input_title"] = f"{tag}{item['input_title']}{title_suffix}"
            else:
                item["input_title"] = f"{item['input_title']}{title_suffix}"

            context = item.get("context") or ""
            item["context"] = (context + context_suffix) if context else context_suffix.lstrip("，")

            seniority = item.get("expected_output", {}).get("seniority")
            if seniority in (None, ""):
                if tag in ("高级", "资深"):
                    item["expected_output"]["seniority"] = "高级/资深"
                elif tag == "初级":
                    item["expected_output"]["seniority"] = "初级/助理"
                elif tag == "首席":
                    item["expected_output"]["seniority"] = "C-level"

            notes = item.get("notes", "")
            item["notes"] = (notes + " | auto_aug").strip(" |")

        out.append(item)
        i += 1

    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(out)} test cases: {path}")


if __name__ == "__main__":
    main()
