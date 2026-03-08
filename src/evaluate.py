"""
Scoring script for fine-tuned SLM outputs.
Computes F1, Precision, Recall, Exact Match. Applies noise filter before scoring.

Usage:
  python src/evaluate.py --predictions predictions.jsonl --test data/test.jsonl --model mistral
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.metrics import f1_score, precision_score, recall_score
from filter import filter_sympathetic_noise, extract_root_cause_from_text

FAILURE_TYPES = [
    "core_network_failure", "authentication_failure", "normal",
    "handover_failure", "congestion", "qos_violation",
    "transport_jitter", "radio_failure",
]

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def primary_label(codes):
    f = filter_sympathetic_noise(codes)
    return f[0] if f else "normal"

def score(preds, gts):
    p = [primary_label(x) for x in preds]
    g = [primary_label(x) for x in gts]
    return {
        "f1":          round(f1_score(g, p, average="micro", labels=FAILURE_TYPES, zero_division=0), 4),
        "precision":   round(precision_score(g, p, average="micro", labels=FAILURE_TYPES, zero_division=0), 4),
        "recall":      round(recall_score(g, p, average="micro", labels=FAILURE_TYPES, zero_division=0), 4),
        "exact_match": round(sum(a == b for a, b in zip(p, g)) / len(g), 4),
        "n": len(g),
    }

def score_per_class(preds, gts):
    p = [primary_label(x) for x in preds]
    g = [primary_label(x) for x in gts]
    out = {}
    for ft in FAILURE_TYPES:
        mt = [1 if x == ft else 0 for x in g]
        mp = [1 if x == ft else 0 for x in p]
        if sum(mt) == 0:
            continue
        out[ft] = {
            "f1":        round(f1_score(mt, mp, zero_division=0), 4),
            "precision": round(precision_score(mt, mp, zero_division=0), 4),
            "recall":    round(recall_score(mt, mp, zero_division=0), 4),
            "n": sum(mt),
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--model", default="model")
    ap.add_argument("--strategy", default="zero_shot")
    ap.add_argument("--out", default="results/results.json")
    args = ap.parse_args()

    test_data = load_jsonl(args.test)
    pred_data = load_jsonl(args.predictions)
    # Align: if predictions < test examples, skip the first N test examples
    # (they were used as few-shot examples and not evaluated)
    offset = len(test_data) - len(pred_data)
    if offset > 0:
        print(f"  Aligning: skipping first {offset} test examples (used as few-shot)")
        test_data = test_data[offset:]
    gts   = [ex["root_cause"] for ex in test_data]
    preds = [ex.get("root_cause") or extract_root_cause_from_text(ex.get("output", "")) for ex in pred_data]

    metrics   = score(preds, gts)
    per_class = score_per_class(preds, gts)
    entry = {"model": args.model, "strategy": args.strategy, "metrics": metrics, "per_class": per_class}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results = json.load(open(args.out)) if os.path.exists(args.out) else []
    results = [r for r in results if (r.get("model"), r.get("strategy")) != (args.model, args.strategy)]
    results.append(entry)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"[{args.model}/{args.strategy}] F1={metrics['f1']} EM={metrics['exact_match']} n={metrics['n']}")

if __name__ == "__main__":
    main()
