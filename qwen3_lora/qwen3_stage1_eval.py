#!/usr/bin/env python3
"""Run inference on eval set and compute proxy accuracy metrics."""

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    return text


def lcs_length(a, b):
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for ca in a:
        prev = 0
        for j, cb in enumerate(b, start=1):
            cur = dp[j]
            if ca == cb:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def char_f1(pred, ref):
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    pred_chars = list(pred)
    ref_chars = list(ref)
    ref_used = [False] * len(ref_chars)
    common = 0
    for ch in pred_chars:
        for idx, ref_ch in enumerate(ref_chars):
            if not ref_used[idx] and ch == ref_ch:
                ref_used[idx] = True
                common += 1
                break
    if common == 0:
        return 0.0
    precision = common / len(pred_chars)
    recall = common / len(ref_chars)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(pred, ref):
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    lcs = lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def extract_text_generation(result_path):
    content = result_path.read_text(encoding="utf-8").strip()
    parsed = ast.literal_eval(content)
    if not parsed:
        return ""
    first = parsed[0]
    if isinstance(first, dict):
        text_list = first.get("text_generation_text") or []
        if text_list:
            return str(text_list[0])
    return str(first)


def run_predict(repo_root, config_path, prompt, pretrained_model_dir, load_checkpoint, python_bin):
    cmd = [
        python_bin,
        "run_mindformer.py",
        "--config",
        str(config_path),
        "--run_mode",
        "predict",
        "--use_parallel",
        "False",
        "--predict_data",
        prompt,
        "--options",
        f"pretrained_model_dir={pretrained_model_dir}",
    ]
    if load_checkpoint:
        cmd.extend(["--load_checkpoint", str(load_checkpoint)])
    subprocess.run(cmd, cwd=repo_root, check=True)
    return extract_text_generation(Path(repo_root) / "text_generation_result.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", required=True, help="Path to stage1_eval.json.")
    parser.add_argument("--config", required=True, help="Predict yaml path.")
    parser.add_argument("--pretrained-model-dir", required=True, help="HF model dir.")
    parser.add_argument("--load-checkpoint", default="", help="Merged fine-tuned ckpt path.")
    parser.add_argument("--output-file", required=True, help="Path to save eval result json.")
    parser.add_argument("--python-bin", default="python", help="Python executable for run_mindformer.")
    parser.add_argument("--accuracy-threshold", type=float, default=0.7, help="Threshold for hit rate.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    eval_file = Path(args.eval_file)
    config_path = Path(args.config)
    output_file = Path(args.output_file)

    with eval_file.open("r", encoding="utf-8") as f:
        eval_records = json.load(f)

    results = []
    exact_total = 0.0
    char_f1_total = 0.0
    rouge_total = 0.0
    hit_total = 0

    for record in eval_records:
        prediction = run_predict(
            repo_root=repo_root,
            config_path=config_path,
            prompt=record["query"],
            pretrained_model_dir=args.pretrained_model_dir,
            load_checkpoint=args.load_checkpoint,
            python_bin=args.python_bin,
        )
        pred_norm = normalize_text(prediction)
        ref_norm = normalize_text(record["reference"])
        exact = 1.0 if pred_norm == ref_norm else 0.0
        f1 = char_f1(pred_norm, ref_norm)
        rouge = rouge_l_f1(pred_norm, ref_norm)
        score = max(exact, f1, rouge)
        hit = 1 if score >= args.accuracy_threshold else 0

        exact_total += exact
        char_f1_total += f1
        rouge_total += rouge
        hit_total += hit

        results.append(
            {
                "id": record["id"],
                "query": record["query"],
                "reference": record["reference"],
                "prediction": prediction,
                "exact_match": exact,
                "char_f1": round(f1, 6),
                "rouge_l_f1": round(rouge, 6),
                "hit": hit,
            }
        )

    total = max(len(results), 1)
    summary = {
        "total": len(results),
        "exact_match_avg": round(exact_total / total, 6),
        "char_f1_avg": round(char_f1_total / total, 6),
        "rouge_l_f1_avg": round(rouge_total / total, 6),
        "accuracy_at_threshold": round(hit_total / total, 6),
        "accuracy_threshold": args.accuracy_threshold,
        "load_checkpoint": args.load_checkpoint,
    }

    output = {"summary": summary, "results": results}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
