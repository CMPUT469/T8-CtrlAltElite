"""
Supabase logging for evaluation runs.

Usage:
    from harness.db_logger import log_run
    log_run(output, model="qwen2.5:7b", suite="jefferson")   
Requires .env:
    SUPABASE_URL=https://xxxx.supabase.co
    SUPABASE_KEY=your-anon-or-service-key

If credentials are missing, logging is silently skipped — local JSON still saves.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        return None
    try:
        from supabase import create_client
        _client = create_client(url, key)
        return _client
    except ImportError:
        print("[db_logger] supabase not installed — skipping. Run: pip install supabase")
        return None
    except Exception as exc:
        print(f"[db_logger] Could not connect to Supabase: {exc}")
        return None


def log_run(
    output: dict,
    suite: str,
    num_distractors: Optional[int] = None,
) -> Optional[str]:
    """
    Upload one completed run to Supabase.

    Args:
        output:          dict returned by harness.runner.run_evaluation()
        suite:           dataset name, e.g. "jefferson" or "bfcl"
        num_distractors: distractor count used (None = standard mode)

    Returns:
        run_id (uuid string) on success, None if skipped or failed.
    """
    client = _get_client()
    if client is None:
        return None

    model   = output.get("model", "unknown")
    metrics = output.get("metrics", {})
    details = output.get("details", [])
    ts      = output.get("timestamp") or datetime.now(timezone.utc).isoformat()

    try:
        run_row = {
            "model":            model,
            "timestamp":        ts,
            "test_suite":       suite,
            "num_distractors":  num_distractors,

            # Primary metric
            "wos":              metrics.get("wos"),
            "wos_l1":           metrics.get("wos_l1"),
            "wos_l2":           metrics.get("wos_l2"),
            "wos_l3":           metrics.get("wos_l3"),

            # Diagnosis
            "total_tasks":      metrics.get("total_tasks"),
            "no_tool_call":     metrics.get("no_tool_call"),

            "raw_metrics":      metrics,
        }

        resp   = client.table("test_runs").insert(run_row).execute()
        run_id = resp.data[0]["id"]

        if details:
            detail_rows = []
            for d in details:
                actual_result = d.get("actual_result")
                if actual_result is not None:
                    try:
                        json.dumps(actual_result)
                    except (TypeError, ValueError):
                        actual_result = str(actual_result)

                detail_rows.append({
                    "run_id":           run_id,
                    "test_id":          d.get("task_id"),
                    "level":            d.get("level"),
                    "query":            d.get("query"),
                    # Reference path (transparency only — not used for scoring)
                    "ref_functions":    d.get("ref_functions"),
                    # What the model actually did
                    "actual_functions": d.get("actual_functions"),
                    "actual_params":    d.get("actual_params"),
                    "actual_result":    actual_result,
                    "correct_result":   d.get("correct_result"),
                    "optimal_steps":    d.get("optimal_steps"),
                    "actual_steps":     d.get("actual_steps"),
                    "error":            d.get("error"),
                    "call_source":      d.get("call_source"),
                    "raw_model_output": d.get("raw_model_output"),
                    "tool_result":      d.get("tool_result"),
                    "expected_outcome": d.get("expected_outcome"),
                })

            client.table("test_details").insert(detail_rows).execute()

        print(f"[db_logger] Logged run {run_id} ({len(details)} tasks) → Supabase")
        return run_id

    except Exception as exc:
        print(f"[db_logger] Failed to log run: {exc}")
        return None