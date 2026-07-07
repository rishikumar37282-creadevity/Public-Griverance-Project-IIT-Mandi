"""CLI entry point.

  python -m src.main train [--epochs N] [--lr F] [--run-name NAME] ...
  python -m src.main predict "complaint text"
  python -m src.main evaluate
"""
import argparse
import json
import sys


def cmd_train(args):
    from .train import train_run
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("command", "run_name") and v is not None}
    summary = train_run(overrides, run_name=args.run_name or "")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, default=str))


def cmd_predict(args):
    from .inference import get_classifier
    from .escalation import recommend
    clf = get_classifier()
    res = clf.classify(args.text)
    res["action_plan"] = recommend(res["urgency"], res["department"], res["org_code"])
    print(json.dumps(res, indent=2, default=str))


def cmd_evaluate(_args):
    from .inference import get_classifier
    clf = get_classifier()
    print(json.dumps(clf.model_card()["test_metrics"], indent=2))


def build_parser():
    p = argparse.ArgumentParser(description="Public Grievance Urgency Classification System")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a model and save production artifacts")
    t.add_argument("--epochs", type=int)
    t.add_argument("--lr", type=float)
    t.add_argument("--batch_size", type=int)
    t.add_argument("--hidden_dim", type=int)
    t.add_argument("--emb_dim", type=int)
    t.add_argument("--num_layers", type=int)
    t.add_argument("--dropout", type=float)
    t.add_argument("--critical_boost", type=float)
    t.add_argument("--patience", type=int)
    t.add_argument("--optimizer", choices=["adamw", "adam", "sgd"])
    t.add_argument("--scheduler", choices=["plateau", "cosine", "none"])
    t.add_argument("--seed", type=int)
    t.add_argument("--run-name", dest="run_name")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="classify one complaint")
    pr.add_argument("text")
    pr.set_defaults(func=cmd_predict)

    ev = sub.add_parser("evaluate", help="print saved test metrics")
    ev.set_defaults(func=cmd_evaluate)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
