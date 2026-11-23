"""Optional offline SetFit intent trainer on intfloat/multilingual-e5-small.

Usage:
  python train_setfit.py --kb ../workflows/kb_export.json --queue ../workflows/queue_export.json --out /data/setfit_model
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from normalizer import normalize_text


def build_training_pairs(kb_rows: List[dict], queue_rows: List[dict]) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []

    for r in kb_rows:
        if not isinstance(r, dict) or r.get("active") is False:
            continue
        if str(r.get("source") or "").startswith("copilot_"):
            continue
        rk = str(r.get("rule_key") or "").strip()
        if not rk:
            continue
        raw_kws = r.get("keywords") or []
        if isinstance(raw_kws, str):
            try:
                raw_kws = json.loads(raw_kws)
            except json.JSONDecodeError:
                raw_kws = [s.strip() for s in raw_kws.split(",") if s.strip()]
        for kw in raw_kws[:12]:
            clean = normalize_text(str(kw))
            if clean:
                texts.append(clean)
                labels.append(rk)

    for q in queue_rows:
        if str(q.get("status") or "") != "approved":
            continue
        ans = str(q.get("answer") or "").strip()
        label = ans[6:] if ans.startswith("ALIAS:") else f"learned_{q.get('id')}"
        ques = normalize_text(str(q.get("question") or ""))
        if ques and label:
            texts.append(ques)
            labels.append(label)

    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SetFit intent classifier on multilingual-e5-small")
    parser.add_argument("--kb", required=True, help="Path to exported chatwoot_kb_rules JSON")
    parser.add_argument("--queue", default="", help="Optional path to exported chatwoot_learning_queue JSON")
    parser.add_argument("--out", default="/data/setfit_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    kb_raw = json.loads(Path(args.kb).read_text(encoding="utf-8"))
    kb_rows = kb_raw if isinstance(kb_raw, list) else kb_raw.get("rules", kb_raw.get("rows", []))
    queue_rows = []
    if args.queue and Path(args.queue).exists():
        q_raw = json.loads(Path(args.queue).read_text(encoding="utf-8"))
        queue_rows = q_raw if isinstance(q_raw, list) else q_raw.get("queue", q_raw.get("rows", []))

    texts, labels = build_training_pairs(kb_rows, queue_rows)
    if len(set(labels)) < 2:
        raise SystemExit("Need at least 2 distinct rule labels to train SetFit.")

    ds = Dataset.from_dict({"text": texts, "label": labels})
    model = SetFitModel.from_pretrained("intfloat/multilingual-e5-small", labels=sorted(set(labels)))
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=args.out, num_epochs=args.epochs, batch_size=16),
        train_dataset=ds,
    )
    trainer.train()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    print(f"Saved SetFit model ({len(texts)} samples, {len(set(labels))} labels) -> {args.out}")


if __name__ == "__main__":
    main()
