"""
Distractor threshold sweep.

Replaces incremental_threshold.py. Runs the same model at increasing
distractor counts and reports how outcome accuracy and TESR degrade.

Usage:
    # Run one level
    python -m harness.threshold_sweep --dataset jefferson --model qwen2.5:7b --distractors 0

    # Run the full sweep (0 → 5 → 10 → 20 → 40)
    python -m harness.threshold_sweep --dataset jefferson --model qwen2.5:7b --sweep

    # Show summary from already-saved results
    python -m harness.threshold_sweep --dataset jefferson --model qwen2.5:7b --summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.metrics import print_report
from harness.model_client import resolve_model_config
from harness.runner import DATASETS, run_evaluation, save_results

DEFAULT_SWEEP = [0, 5, 10, 20, 40]
RESULTS_DIR   = Path("results")


async def _run_one(dataset, model_cfg, distractors, levels):
    output = await run_evaluation(
        dataset=dataset,
        model_cfg=model_cfg,
        levels=levels,
        limit=None,
        num_distractors=distractors,
        allow_fallback=False,
    )
    path = save_results(output)
    return output, path


def _sweep(dataset, model_cfg, levels, distractor_levels):
    results = []
    for n in distractor_levels:
        mode = "oracle" if n == 0 else f"{n} distractors"
        print(f"\n{'='*62}")
        print(f"  {mode}")
        print(f"{'='*62}")
        output, path = asyncio.run(_run_one(dataset, model_cfg, n, levels))
        results.append((n, output["metrics"], path))
        print_report(output["metrics"], model_cfg.name, dataset)

    _print_sweep_summary(results, model_cfg.name)


def _print_sweep_summary(results: list, model: str):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Threshold sweep summary — {model}")
    print(sep)
    header = f"  {'Distractors':<14} {'WOS%':<8} {'WOS-L1':<9} {'WOS-L2':<9} {'WOS-L3':<9} {'No-call'}"
    print(header)
    print("  " + "-" * 62)

    baseline_wos = None

    for n, metrics, _ in results:
        wos = metrics["wos"]
        if baseline_wos is None:
            baseline_wos = wos

        delta_wos = f"({wos - baseline_wos:+.1f}%)" if n != results[0][0] else "(baseline)"
        label = "oracle" if n == 0 else str(n)

        print(
            f"  {label:<14} "
            f"{wos:<5.1f}% {delta_wos:<10} "
            f"{metrics['wos_l1']:<9.1f} "
            f"{metrics['wos_l2']:<9.1f} "
            f"{metrics['wos_l3']:<9.1f} "
            f"{metrics['no_tool_call']}"
        )

    print(sep + "\n")


def _load_sweep_from_disk(dataset: str, model: str, distractor_levels: list) -> list:
    """Try to load previously saved results matching this sweep config."""
    model_safe = model.replace(":", "_").replace("/", "_")
    found = []
    for n in distractor_levels:
        # glob for any result file matching dataset + model
        pattern = f"{dataset}_{model_safe}_*.json"
        matches = sorted(RESULTS_DIR.glob(pattern), reverse=True)
        # pick the most recent one that has matching num_distractors
        for f in matches:
            try:
                data = json.loads(f.read_text())
                if data.get("num_distractors") == n:
                    found.append((n, data.get("metrics", {}), f))
                    break
            except Exception:
                continue
    return found


def main():
    p = argparse.ArgumentParser(description="Distractor threshold sweep")
    p.add_argument("--dataset",    required=True, choices=list(DATASETS))
    p.add_argument("--model",      required=True)
    p.add_argument("--backend",    default="ollama", choices=["ollama", "openai"])
    p.add_argument("--base-url",   default=None)
    p.add_argument("--api-key",    default=None)
    p.add_argument("--level",      nargs="+", default=["L1", "L2", "L3"])
    p.add_argument("--distractors",type=int, default=None,
                   help="Run a single sweep point at N distractors")
    p.add_argument("--sweep",      action="store_true",
                   help=f"Run full sweep at {DEFAULT_SWEEP}")
    p.add_argument("--sweep-points", nargs="+", type=int, default=None,
                   help="Custom sweep points, e.g. --sweep-points 0 10 30")
    p.add_argument("--summary",    action="store_true",
                   help="Print summary table from saved results (no new runs)")
    args = p.parse_args()

    model_cfg = resolve_model_config(
        args.model,
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    if args.summary:
        points = args.sweep_points or DEFAULT_SWEEP
        data   = _load_sweep_from_disk(args.dataset, args.model, points)
        if data:
            _print_sweep_summary(data, args.model)
        else:
            print("No saved results found for this model/dataset. Run --sweep first.")
        return

    if args.distractors is not None:
        output, path = asyncio.run(_run_one(args.dataset, model_cfg, args.distractors, args.level))
        print_report(output["metrics"], args.model, args.dataset)
        print(f"Saved → {path}")
        return

    if args.sweep:
        points = args.sweep_points or DEFAULT_SWEEP
        _sweep(args.dataset, model_cfg, args.level, points)
        return

    p.print_help()


if __name__ == "__main__":
    main()
