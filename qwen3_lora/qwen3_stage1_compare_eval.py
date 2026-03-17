#!/usr/bin/env python3
"""Compare base and fine-tuned eval summaries."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-result", required=True, help="Path to base eval result json.")
    parser.add_argument("--tuned-result", required=True, help="Path to tuned eval result json.")
    args = parser.parse_args()

    with open(args.base_result, "r", encoding="utf-8") as f:
        base = json.load(f)["summary"]
    with open(args.tuned_result, "r", encoding="utf-8") as f:
        tuned = json.load(f)["summary"]

    keys = ["exact_match_avg", "char_f1_avg", "rouge_l_f1_avg", "accuracy_at_threshold"]
    diff = {}
    for key in keys:
        diff[key] = round(tuned.get(key, 0.0) - base.get(key, 0.0), 6)

    report = {
        "base": base,
        "tuned": tuned,
        "delta": diff,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
