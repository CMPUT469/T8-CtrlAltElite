"""
Supabase logging module for T8-CtrlAltElite test results.

Usage:
    from db_logger import log_run
    log_run(results, metrics, model="qwen2.5", suite="bfcl", num_distractors=None)

Requires .env with:
    SUPABASE_URL=https://xxxx.supabase.co
    SUPABASE_KEY=your-anon-or-service-key

If credentials are missing, logging is silently skipped and local JSON still works.
"""

import json
import os
from datetime import datetime
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
        print("[db_logger] supabase package not installed — skipping cloud logging.")
        print("[db_logger] Run: pip install supabase")
        return None
    except Exception as e:
        print(f"[db_logger] Failed to connect to Supabase: {e}")
        return None


def log_run(
    results: dict,
    metrics: dict,
    model: str,
    suite: str,
    num_distractors: Optional[int] = None,
) -> Optional[str]:
    """
    Upload one test run and its per-test details to Supabase.

    Returns the run_id (uuid) on success, None if skipped or failed.
    """
    client = _get_client()
    if client is None:
        return None

    try:
        # --- Insert run row ---
        timestamp = results.get("timestamp") or datetime.utcnow().isoformat()

        run_row = {
            "model": model,
            "timestamp": timestamp,
            "test_suite": suite,
            "num_distractors": num_distractors,
            "f1_score": metrics.get("f1_score"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "tsr_function_selection": metrics.get("tsr_function_selection"),
            "tsr_result_accuracy": metrics.get("tsr_result_accuracy"),
            "total_tests": metrics.get("total_tests"),
            "correct_function": metrics.get("correct_function"),
            "correct_result": metrics.get("correct_result"),
            "no_tool_call": metrics.get("no_tool_call"),
            "wrong_tool": metrics.get("wrong_tool"),
            "raw_metrics": metrics,
        }

        response = client.table("test_runs").insert(run_row).execute()
        run_id = response.data[0]["id"]

        # --- Insert per-test detail rows ---
        details = results.get("details", [])
        if details:
            detail_rows = []
            for d in details:
                # Keep actual_result as-is (jsonb) — no float coercion
                actual_result = d.get("actual_result")
                if actual_result is not None:
                    try:
                        json.dumps(actual_result)
                    except (TypeError, ValueError):
                        actual_result = str(actual_result)

                # Wrap raw string incorrect_output in a dict for jsonb
                incorrect_output = d.get("incorrect_output")
                if isinstance(incorrect_output, str):
                    incorrect_output = {"raw_text": incorrect_output}

                detail_rows.append({
                    "run_id": run_id,
                    "test_id": d.get("test_id"),
                    "query": d.get("query"),
                    "expected_function": d.get("expected_function"),
                    "actual_function": d.get("actual_function"),
                    "expected_params": d.get("expected_params"),
                    "actual_params": d.get("actual_params"),
                    "actual_result": actual_result,
                    "expected_result": d.get("expected_result"),
                    "correct_function": d.get("correct_function"),
                    "correct_params": d.get("correct_params"),
                    "correct_result": d.get("correct_result"),
                    "error": d.get("error"),
                    "incorrect_output": incorrect_output,
                    "raw_model_output": d.get("raw_model_output"),
                    "tool_result": d.get("tool_result"),
                })

            client.table("test_details").insert(detail_rows).execute()

        print(f"[db_logger] Logged run {run_id} ({len(details)} tests) to Supabase.")
        return run_id

    except Exception as e:
        print(f"[db_logger] Failed to log run: {e}")
        return None
