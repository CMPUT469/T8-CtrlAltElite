"""
Generate additional L1 tasks per tool to tighten the anchor threshold sweep
([harness/random_sweep.py](../harness/random_sweep.py)).

Emits one `tasks_l1_extra.jsonl` file per dataset alongside the existing
`tasks_l1.jsonl`. The sweep loader auto-picks these up so existing files are
left untouched. See [scalable-gathering-tide.md](../../../.claude/plans/scalable-gathering-tide.md)
for the rationale.

Strategies:
  - bfcl       deterministic Python builtins (mirrors mcp-server/tools/bfcl_math_tools.py)
  - jefferson  deterministic SciPy + Python (mirrors jefferson_stats_tools.py)
  - finance    parameter-only (live API at eval time → no expected_outcome)
  - postgres   live psycopg2 query against the bookings demo DB at generation time

Run:
    .venv/bin/python scripts/gen_l1_tasks.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TARGET_PER_TOOL = 15

DATASET_DIRS = {
    "bfcl": "datasets/bfcl_math",
    "jefferson": "datasets/jefferson_stats",
    "finance": "datasets/finance",
    "postgres": "datasets/postgres",
}


# ──────────────────────────────────────────────────────────────────────────────
# Common helpers
# ──────────────────────────────────────────────────────────────────────────────

def _round_floats(value, ndigits: int = 4):
    """Recursively round floats so JSON output stays stable + readable."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return value
        return round(value, ndigits)
    if isinstance(value, list):
        return [_round_floats(v, ndigits) for v in value]
    if isinstance(value, dict):
        return {k: _round_floats(v, ndigits) for k, v in value.items()}
    return value


def _write_jsonl(path: Path, tasks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# bfcl (math) — mirrors mcp-server/tools/bfcl_math_tools.py exactly
# ──────────────────────────────────────────────────────────────────────────────

def _math_id(n: int) -> str:
    return f"math_l1_{n:03d}"


def _make_task(task_id: str, function: str, query: str, params: dict, outcome: dict | None) -> dict:
    t = {
        "id": task_id,
        "level": "L1",
        "function": function,
        "query": query,
        "expected_params": params,
        "optimal_steps": 1,
    }
    if outcome is not None:
        t["expected_outcome"] = outcome
    return t


def gen_bfcl(rng: random.Random) -> list[dict]:
    tasks: list[dict] = []
    next_id = 16  # existing file ends at math_l1_015

    add_phrasings = [
        "Add {a} and {b}.", "What is {a} + {b}?", "Sum {a} and {b}.",
        "Compute {a} plus {b}.", "Find the total of {a} and {b}.",
    ]
    sub_phrasings = [
        "Subtract {b} from {a}.", "What is {a} minus {b}?",
        "Compute {a} - {b}.", "Find the difference {a} - {b}.",
    ]
    mul_phrasings = [
        "Multiply {a} by {b}.", "What is {a} times {b}?",
        "Compute the product of {a} and {b}.", "Find {a} * {b}.",
    ]
    div_phrasings = [
        "Divide {a} by {b}.", "What is {a} divided by {b}?",
        "Compute {a} / {b}.", "Find the quotient of {a} divided by {b}.",
    ]

    def emit(fn, query, params, outcome):
        nonlocal next_id
        tasks.append(_make_task(_math_id(next_id), fn, query, params, outcome))
        next_id += 1

    # add (15) — integer + float values
    for _ in range(TARGET_PER_TOOL):
        a = rng.randint(-500, 500)
        b = rng.randint(-500, 500)
        q = rng.choice(add_phrasings).format(a=a, b=b)
        emit("add", q, {"a": a, "b": b}, {"result": a + b})

    # subtract (15)
    for _ in range(TARGET_PER_TOOL):
        a = rng.randint(-500, 500)
        b = rng.randint(-500, 500)
        q = rng.choice(sub_phrasings).format(a=a, b=b)
        emit("subtract", q, {"a": a, "b": b}, {"result": a - b})

    # multiply (15)
    for _ in range(TARGET_PER_TOOL):
        a = rng.randint(-50, 50)
        b = rng.randint(-50, 50)
        q = rng.choice(mul_phrasings).format(a=a, b=b)
        emit("multiply", q, {"a": a, "b": b}, {"result": a * b})

    # divide (15) — pick clean integer ratios so result is exact float
    for _ in range(TARGET_PER_TOOL):
        b = rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
        q_factor = rng.randint(1, 50)
        a = b * q_factor * rng.choice([-1, 1])
        q = rng.choice(div_phrasings).format(a=a, b=b)
        emit("divide", q, {"a": a, "b": b}, {"result": a / b})

    # power (15)
    for _ in range(TARGET_PER_TOOL):
        base = rng.randint(2, 12)
        exponent = rng.randint(2, 6)
        q = rng.choice([
            "What is {base} raised to the power of {exp}?",
            "Compute {base}^{exp}.",
            "Calculate {base} to the {exp}.",
        ]).format(base=base, exp=exponent)
        emit("power", q, {"base": base, "exponent": exponent},
             {"result": base ** exponent})

    # square_root (15)
    for _ in range(TARGET_PER_TOOL):
        precision = rng.randint(2, 5)
        # mix perfect squares and arbitrary values
        if rng.random() < 0.5:
            n = rng.randint(2, 30) ** 2
        else:
            n = rng.randint(10, 9999)
        q = rng.choice([
            "Calculate the square root of {n} to {p} decimal places.",
            "What is sqrt({n}) rounded to {p} decimals?",
            "Find the square root of {n} (precision {p}).",
        ]).format(n=n, p=precision)
        emit("square_root", q, {"number": n, "precision": precision},
             {"result": round(math.sqrt(n), precision)})

    # absolute_value (15)
    for _ in range(TARGET_PER_TOOL):
        n = rng.randint(-9999, 9999)
        q = rng.choice([
            "What is the absolute value of {n}?",
            "Compute |{n}|.",
            "Find abs({n}).",
        ]).format(n=n)
        emit("absolute_value", q, {"number": n}, {"result": abs(n)})

    # round_number (15)
    for _ in range(TARGET_PER_TOOL):
        whole = rng.randint(-100, 100)
        frac = rng.randint(0, 999_999) / 1_000_000
        n = round(whole + frac, 6)
        dp = rng.randint(0, 4)
        q = rng.choice([
            "Round {n} to {dp} decimal places.",
            "What is {n} rounded to {dp} decimals?",
        ]).format(n=n, dp=dp)
        emit("round_number", q, {"number": n, "decimal_places": dp},
             {"result": round(n, dp)})

    # percentage (15)
    for _ in range(TARGET_PER_TOOL):
        whole = rng.randint(20, 1000)
        part = rng.randint(1, whole)
        q = rng.choice([
            "What percentage is {part} of {whole}?",
            "Express {part} as a percentage of {whole}.",
            "Compute {part}/{whole} as a percentage.",
        ]).format(part=part, whole=whole)
        emit("percentage", q, {"part": part, "whole": whole},
             {"result": (part / whole) * 100})

    # sum_values (15)
    for _ in range(TARGET_PER_TOOL):
        size = rng.randint(3, 10)
        nums = [rng.randint(-100, 200) for _ in range(size)]
        q = f"Find the sum of {nums}."
        emit("sum_values", q, {"numbers": nums}, {"result": sum(nums)})

    # mean (15)
    for _ in range(TARGET_PER_TOOL):
        size = rng.randint(3, 10)
        nums = [rng.randint(-50, 100) for _ in range(size)]
        q = f"Calculate the mean of {nums}."
        emit("mean", q, {"numbers": nums}, {"result": sum(nums) / len(nums)})

    # min_value (15)
    for _ in range(TARGET_PER_TOOL):
        size = rng.randint(3, 10)
        nums = [rng.randint(-200, 200) for _ in range(size)]
        q = f"Find the minimum of {nums}."
        emit("min_value", q, {"numbers": nums}, {"result": min(nums)})

    # max_value (15)
    for _ in range(TARGET_PER_TOOL):
        size = rng.randint(3, 10)
        nums = [rng.randint(-200, 200) for _ in range(size)]
        q = f"Find the maximum of {nums}."
        emit("max_value", q, {"numbers": nums}, {"result": max(nums)})

    # standard_deviation (15) — sample stddev (n-1) matching the tool
    for _ in range(TARGET_PER_TOOL):
        size = rng.randint(4, 10)
        nums = [rng.randint(0, 50) for _ in range(size)]
        mean_val = sum(nums) / len(nums)
        variance = sum((x - mean_val) ** 2 for x in nums) / (len(nums) - 1)
        result = math.sqrt(variance)
        q = f"Calculate the standard deviation of {nums}."
        emit("standard_deviation", q, {"numbers": nums},
             {"result": round(result, 5)})

    # logarithm (15)
    for _ in range(TARGET_PER_TOOL):
        base = rng.choice([2, 10, math.e, 3, 5])
        # pick number so result is reasonable
        n = rng.choice([8, 16, 32, 64, 100, 1000, 81, 125, 243, 1024, 27, 729])
        if base == math.e:
            q = f"Calculate the natural log of {n}."
            params = {"number": n}
        else:
            q = f"Calculate log base {base} of {n}."
            params = {"number": n, "base": base}
        emit("logarithm", q, params,
             {"result": round(math.log(n, base), 6)})

    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# jefferson (stats) — mirrors mcp-server/tools/jefferson_stats_tools.py
# ──────────────────────────────────────────────────────────────────────────────

def _stats_id(n: int) -> str:
    return f"l1_{n:03d}"


def gen_jefferson(rng: random.Random) -> list[dict]:
    try:
        from scipy import stats as scipy_stats
    except ImportError as exc:
        raise RuntimeError(
            f"scipy is required to generate jefferson tasks: {exc}"
        ) from exc

    tasks: list[dict] = []
    next_id = 19  # existing file ends at l1_018

    def emit(fn, query, params, outcome):
        nonlocal next_id
        tasks.append(_make_task(_stats_id(next_id), fn, query, params, outcome))
        next_id += 1

    def random_collection(min_size=6, max_size=12, low=0, high=50) -> list[int]:
        size = rng.randint(min_size, max_size)
        return [rng.randint(low, high) for _ in range(size)]

    # calculate_median (15)
    for _ in range(TARGET_PER_TOOL):
        c = random_collection()
        sorted_c = sorted(c)
        n = len(sorted_c)
        median = (sorted_c[n//2 - 1] + sorted_c[n//2]) / 2 if n % 2 == 0 else sorted_c[n//2]
        emit("calculate_median", f"Find the median of {c}.",
             {"collection": c}, {"result": float(median)})

    # calculate_mode (15) — make sure there's a unique mode
    for _ in range(TARGET_PER_TOOL):
        base = [rng.randint(1, 20) for _ in range(rng.randint(4, 8))]
        repeated = rng.randint(1, 20)
        c = base + [repeated, repeated, repeated, repeated]
        rng.shuffle(c)
        # compute mode the same way the tool does
        freq = {}
        for v in c:
            freq[v] = freq.get(v, 0) + 1
        max_freq = max(freq.values())
        modes = [k for k, v in freq.items() if v == max_freq]
        result = float(modes[0]) if len(modes) == 1 else [float(m) for m in modes]
        emit("calculate_mode", f"What is the mode of {c}?",
             {"collection": c}, {"result": result})

    # calculate_range (15)
    for _ in range(TARGET_PER_TOOL):
        c = random_collection()
        emit("calculate_range", f"What is the range of {c}?",
             {"collection": c}, {"result": float(max(c) - min(c))})

    # calculate_variance (15) — population variance (matches tool)
    for _ in range(TARGET_PER_TOOL):
        c = random_collection()
        m = sum(c) / len(c)
        var = sum((x - m) ** 2 for x in c) / len(c)
        emit("calculate_variance", f"Compute the variance of {c}.",
             {"collection": c}, {"result": round(float(var), 5)})

    # calculate_quartiles (15)
    for _ in range(TARGET_PER_TOOL):
        c = random_collection(min_size=8, max_size=14)
        sorted_c = sorted(c)
        n = len(sorted_c)
        q1 = sorted_c[n // 4]
        q2 = sorted_c[n // 2]
        q3 = sorted_c[3 * n // 4]
        emit("calculate_quartiles", f"Calculate Q1, Q2, Q3 for {c}.",
             {"collection": c},
             {"q1": float(q1), "q2": float(q2), "q3": float(q3)})

    # calculate_iqr (15)
    for _ in range(TARGET_PER_TOOL):
        c = random_collection(min_size=8, max_size=14)
        sorted_c = sorted(c)
        n = len(sorted_c)
        q1 = sorted_c[n // 4]
        q3 = sorted_c[3 * n // 4]
        emit("calculate_iqr", f"Calculate the interquartile range of {c}.",
             {"collection": c}, {"result": float(q3 - q1)})

    # calculate_skewness (15)
    for _ in range(TARGET_PER_TOOL):
        c = random_collection(min_size=8, max_size=14, low=1, high=30)
        skew_val = float(scipy_stats.skew(c))
        emit("calculate_skewness", f"What is the skewness of {c}?",
             {"collection": c}, {"result": round(skew_val, 4)})

    # calculate_kurtosis (15)
    for _ in range(TARGET_PER_TOOL):
        c = random_collection(min_size=8, max_size=14, low=1, high=30)
        kurt_val = float(scipy_stats.kurtosis(c))
        emit("calculate_kurtosis", f"Calculate the kurtosis of {c}.",
             {"collection": c}, {"result": round(kurt_val, 4)})

    # calculate_correlation (15) — non-perfect to keep p_value finite
    for _ in range(TARGET_PER_TOOL):
        size = rng.randint(6, 10)
        c1 = [rng.randint(1, 50) for _ in range(size)]
        c2 = [v * rng.choice([2, 3]) + rng.randint(-3, 3) for v in c1]
        corr, _ = scipy_stats.pearsonr(c1, c2)
        emit("calculate_correlation",
             f"Find the Pearson correlation between {c1} and {c2}.",
             {"collection1": c1, "collection2": c2},
             {"correlation": round(float(corr), 4)})

    # calculate_covariance (15) — sample covariance (n-1) matching the tool
    for _ in range(TARGET_PER_TOOL):
        size = rng.randint(6, 10)
        c1 = [rng.randint(1, 50) for _ in range(size)]
        c2 = [rng.randint(1, 50) for _ in range(size)]
        m1 = sum(c1) / len(c1)
        m2 = sum(c2) / len(c2)
        cov = sum((c1[i] - m1) * (c2[i] - m2) for i in range(size)) / (size - 1)
        emit("calculate_covariance",
             f"Calculate the covariance between {c1} and {c2}.",
             {"collection1": c1, "collection2": c2},
             {"result": round(float(cov), 4)})

    # calculate_z_scores (15)
    for _ in range(TARGET_PER_TOOL):
        c = random_collection(min_size=6, max_size=10, low=10, high=80)
        zs = [round(float(z), 2) for z in scipy_stats.zscore(c)]
        emit("calculate_z_scores", f"Calculate z-scores for {c}.",
             {"collection": c}, {"z_scores": zs})

    # perform_t_test (15)
    for _ in range(TARGET_PER_TOOL):
        c = random_collection(min_size=6, max_size=10, low=5, high=25)
        popmean = rng.randint(5, 25)
        t_stat, p_val = scipy_stats.ttest_1samp(c, popmean)
        emit("perform_t_test",
             f"Run a one-sample t-test on {c} against population mean {popmean}.",
             {"collection": c, "popmean": popmean},
             {"t_statistic": round(float(t_stat), 4),
              "p_value": round(float(p_val), 4)})

    # calculate_confidence_interval (15)
    for _ in range(TARGET_PER_TOOL):
        c = random_collection(min_size=6, max_size=12, low=20, high=80)
        confidence = 0.95
        m = sum(c) / len(c)
        sem = scipy_stats.sem(c)
        interval = sem * scipy_stats.t.ppf((1 + confidence) / 2, len(c) - 1)
        emit("calculate_confidence_interval",
             f"Calculate the {int(confidence*100)}% confidence interval for the mean of {c}.",
             {"collection": c, "confidence": confidence},
             {"lower_bound": round(float(m - interval), 2),
              "upper_bound": round(float(m + interval), 2)})

    # detect_outliers (15) — inject one extreme outlier so the test is meaningful
    for _ in range(TARGET_PER_TOOL):
        base = [rng.randint(10, 30) for _ in range(rng.randint(8, 12))]
        outlier = rng.randint(200, 500)
        c = base + [outlier]
        rng.shuffle(c)
        sorted_c = sorted(c)
        n = len(sorted_c)
        q1 = sorted_c[n // 4]
        q3 = sorted_c[3 * n // 4]
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        outliers = [x for x in c if x < lo or x > hi]
        emit("detect_outliers", f"Detect outliers in {c} using IQR.",
             {"collection": c},
             {"outliers": outliers, "count": len(outliers)})

    # perform_normality_test (15) — pick a clearly normal-ish dataset
    for _ in range(TARGET_PER_TOOL):
        # tight cluster → likely normal under shapiro
        c = [rng.randint(9, 14) for _ in range(rng.randint(10, 14))]
        stat, p = scipy_stats.shapiro(c)
        interpretation = "normally distributed" if p > 0.05 else "not normally distributed"
        emit("perform_normality_test",
             f"Test if {c} is normally distributed.",
             {"collection": c},
             {"interpretation": interpretation})

    # perform_linear_regression (15)
    for _ in range(TARGET_PER_TOOL):
        size = rng.randint(6, 10)
        x = list(range(1, size + 1))
        slope = rng.choice([1, 2, 3, -1, -2])
        intercept = rng.randint(-5, 10)
        y = [slope * xi + intercept + rng.choice([-1, 0, 1]) for xi in x]
        result = scipy_stats.linregress(x, y)
        emit("perform_linear_regression",
             f"Perform linear regression on x={x} and y={y}.",
             {"x": x, "y": y},
             {"slope": round(float(result.slope), 4),
              "intercept": round(float(result.intercept), 4)})

    # calculate_moving_average (15)
    for _ in range(TARGET_PER_TOOL):
        size = rng.randint(8, 14)
        c = [rng.randint(0, 30) for _ in range(size)]
        window = rng.randint(2, min(5, size - 1))
        result = [
            round(sum(c[i:i + window]) / window, 4)
            for i in range(len(c) - window + 1)
        ]
        emit("calculate_moving_average",
             f"Calculate the {window}-period moving average of {c}.",
             {"collection": c, "window_size": window},
             {"result": result})

    # generate_descriptive_statistics (15)
    for _ in range(TARGET_PER_TOOL):
        c = random_collection(min_size=8, max_size=14, low=1, high=50)
        n = len(c)
        m = sum(c) / n
        emit("generate_descriptive_statistics",
             f"Generate descriptive statistics for {c}.",
             {"collection": c},
             {"count": n,
              "mean": round(float(m), 4),
              "min": float(min(c)),
              "max": float(max(c))})

    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# finance — parameter-only (live API at eval time → no expected_outcome)
# ──────────────────────────────────────────────────────────────────────────────

def _finance_id(n: int) -> str:
    return f"finance_{n:03d}"


TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA"]
PERIODS = ["annual", "quarterly"]
FILING_TYPES = ["10-K", "10-Q", "8-K"]
COMPANY_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft",
    "TSLA": "Tesla", "NVDA": "Nvidia",
}


def gen_finance(rng: random.Random) -> list[dict]:
    tasks: list[dict] = []
    next_id = 30  # existing file ends at finance_029

    def emit(fn, query, params):
        nonlocal next_id
        tasks.append(_make_task(_finance_id(next_id), fn, query, params, outcome=None))
        next_id += 1

    def pick_ticker():
        return rng.choice(TICKERS)

    # get_income_statements / get_balance_sheets / get_cash_flow_statements
    statement_tools = [
        ("get_income_statements", "income statements"),
        ("get_balance_sheets", "balance sheets"),
        ("get_cash_flow_statements", "cash flow statements"),
    ]
    for fn, label in statement_tools:
        for _ in range(TARGET_PER_TOOL):
            ticker = pick_ticker()
            period = rng.choice(PERIODS)
            limit = rng.randint(1, 6)
            emit(fn,
                 f"Retrieve the latest {limit} {period} {label} for {COMPANY_NAMES[ticker]}.",
                 {"ticker": ticker, "period": period, "limit": limit})

    # get_current_stock_price (15)
    for _ in range(TARGET_PER_TOOL):
        ticker = pick_ticker()
        emit("get_current_stock_price",
             f"What is the current stock price of {COMPANY_NAMES[ticker]}?",
             {"ticker": ticker})

    # get_historical_stock_prices (15)
    for _ in range(TARGET_PER_TOOL):
        ticker = pick_ticker()
        year = rng.randint(2022, 2025)
        start_month = rng.randint(1, 9)
        end_month = start_month + rng.randint(1, 3)
        start_date = f"{year}-{start_month:02d}-01"
        end_date = f"{year}-{end_month:02d}-28"
        interval = rng.choice(["day", "week"])
        emit("get_historical_stock_prices",
             f"Retrieve {interval}ly stock prices for {COMPANY_NAMES[ticker]} from {start_date} to {end_date}.",
             {"ticker": ticker,
              "start_date": start_date,
              "end_date": end_date,
              "interval": interval,
              "interval_multiplier": 1})

    # get_company_news (15)
    for _ in range(TARGET_PER_TOOL):
        ticker = pick_ticker()
        emit("get_company_news",
             f"Show recent company news for {COMPANY_NAMES[ticker]}.",
             {"ticker": ticker})

    # get_sec_filings (15) — vary filing_type and limit
    for _ in range(TARGET_PER_TOOL):
        ticker = pick_ticker()
        filing_type = rng.choice(FILING_TYPES)
        limit = rng.randint(2, 10)
        emit("get_sec_filings",
             f"List the latest {limit} {filing_type} filings for {COMPANY_NAMES[ticker]}.",
             {"ticker": ticker, "limit": limit, "filing_type": filing_type})

    # getAnalystEstimates (15)
    for _ in range(TARGET_PER_TOOL):
        ticker = pick_ticker()
        period = rng.choice(PERIODS)
        limit = rng.randint(1, 6)
        emit("getAnalystEstimates",
             f"Retrieve the latest {limit} {period} analyst estimates for {COMPANY_NAMES[ticker]}.",
             {"ticker": ticker, "period": period, "limit": limit})

    # getFinancialMetrics (15)
    for _ in range(TARGET_PER_TOOL):
        ticker = pick_ticker()
        period = rng.choice(PERIODS)
        limit = rng.randint(1, 6)
        emit("getFinancialMetrics",
             f"Show the latest {limit} {period} financial metrics for {COMPANY_NAMES[ticker]}.",
             {"ticker": ticker, "period": period, "limit": limit})

    # getFinancialMetricsSnapshot (15)
    for _ in range(TARGET_PER_TOOL):
        ticker = pick_ticker()
        emit("getFinancialMetricsSnapshot",
             f"What is the latest financial metrics snapshot for {COMPANY_NAMES[ticker]}?",
             {"ticker": ticker})

    # getSegmentedRevenues (15)
    for _ in range(TARGET_PER_TOOL):
        ticker = pick_ticker()
        period = rng.choice(PERIODS)
        limit = rng.randint(1, 6)
        emit("getSegmentedRevenues",
             f"Retrieve the latest {limit} {period} segmented revenues for {COMPANY_NAMES[ticker]}.",
             {"ticker": ticker, "period": period, "limit": limit})

    # getAvailableFilingItems (15)
    for _ in range(TARGET_PER_TOOL):
        filing_type = rng.choice(FILING_TYPES)
        emit("getAvailableFilingItems",
             f"List the available filing items for {filing_type} filings.",
             {"filing_type": filing_type})

    # getFilingItems (15)
    items = ["Risk Factors", "Management Discussion", "Financial Statements", "Legal Proceedings"]
    for _ in range(TARGET_PER_TOOL):
        ticker = pick_ticker()
        filing_type = rng.choice(FILING_TYPES)
        year = rng.randint(2022, 2024)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        filing_date = f"{year}-{month:02d}-{day:02d}"
        item = rng.choice(items)
        emit("getFilingItems",
             f"Extract the {item} item from {COMPANY_NAMES[ticker]}'s {filing_type} filed on {filing_date}.",
             {"ticker": ticker, "filing_type": filing_type, "filing_date": filing_date, "item": item})

    # getCompanyFacts (15)
    for _ in range(TARGET_PER_TOOL):
        ticker = pick_ticker()
        emit("getCompanyFacts",
             f"Retrieve company facts for {COMPANY_NAMES[ticker]}.",
             {"ticker": ticker})

    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# postgres — live psycopg2 query against bookings demo DB
# ──────────────────────────────────────────────────────────────────────────────

def _pg_id(n: int) -> str:
    return f"pg_test_{n}"


def _open_pg():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Postgres generator requires a live "
            "connection to the bookings demo DB."
        )
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError(f"psycopg2 is required for postgres generator: {exc}") from exc
    return psycopg2.connect(db_url)


def _pg_list_tables(conn, schema: str = "bookings") -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s ORDER BY table_name",
            (schema,),
        )
        return [r[0] for r in cur.fetchall()]


def _pg_describe_table(conn, table: str, schema: str = "bookings") -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (schema, table),
        )
        return [{"column": r[0], "type": r[1]} for r in cur.fetchall()]


def _pg_row_count(conn, table: str, schema: str = "bookings") -> int:
    from psycopg2 import sql as psql
    with conn.cursor() as cur:
        cur.execute(
            psql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                psql.Identifier(schema), psql.Identifier(table)
            )
        )
        return cur.fetchone()[0]


def _pg_foreign_keys(conn, table: str, schema: str = "bookings") -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT kcu.column_name, ccu.table_name AS foreign_table,
                   ccu.column_name AS foreign_column
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
             AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema = %s AND tc.table_name = %s
            ORDER BY kcu.ordinal_position, ccu.column_name DESC
            """,
            (schema, table),
        )
        return [
            {"column": r[0], "references_table": r[1], "references_column": r[2]}
            for r in cur.fetchall()
        ]


def _pg_find_relationships(conn, table: str, schema: str = "bookings") -> dict:
    """Mirror the find_relationships SQL from sql_tools.py exactly."""
    explicit_sql = """
        SELECT kcu.column_name,
               ccu.table_name AS foreign_table,
               ccu.column_name AS foreign_column,
               'explicit_fk' AS relationship_type
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = %s AND tc.table_name = %s
    """
    implied_sql = """
        WITH source_cols AS (
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
                AND (column_name LIKE '%%_id' OR column_name LIKE '%%_fk')
        )
        SELECT sc.column_name,
               t.table_name AS foreign_table,
               'id' AS foreign_column,
               CASE
                   WHEN sc.column_name = t.table_name || '_id' THEN 'strong_implied'
                   ELSE 'possible_implied'
               END AS relationship_type
        FROM source_cols sc
        CROSS JOIN information_schema.tables t
        JOIN information_schema.columns c
            ON c.table_schema = t.table_schema
            AND c.table_name = t.table_name
            AND c.column_name = 'id'
        WHERE t.table_schema = %s AND t.table_name != %s
            AND sc.data_type = c.data_type
    """
    with conn.cursor() as cur:
        cur.execute(explicit_sql, (schema, table))
        cols = [d[0] for d in cur.description]
        explicit = [dict(zip(cols, row)) for row in cur.fetchall()]

        cur.execute(implied_sql, (schema, table, schema, table))
        cols = [d[0] for d in cur.description]
        implied = [dict(zip(cols, row)) for row in cur.fetchall()]
    return {"explicit": explicit, "implied": implied}


def _pg_list_schemas(conn, include_system: bool = False) -> list[dict]:
    conditions = []
    params: list = []
    if not include_system:
        conditions.append("n.nspname NOT LIKE %s AND n.nspname != %s")
        params.extend(["pg_%", "information_schema"])
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"""
        SELECT n.nspname AS schema_name,
               pg_get_userbyid(n.nspowner) AS owner,
               has_schema_privilege(n.nspname, 'USAGE') AS has_usage
        FROM pg_namespace n
        {where}
        ORDER BY n.nspname
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _pg_execute_query(conn, sql_query: str, limit: int = 50) -> dict:
    """Mirror execute_query: wraps SELECT in subq, adds limit, returns rows."""
    import psycopg2
    normalized = sql_query.strip().rstrip(";").strip()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM (%s) subq LIMIT %s",
            (psycopg2.extensions.AsIs(normalized), limit),
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    return {"result": rows, "row_count": len(rows)}


def gen_postgres(rng: random.Random) -> list[dict]:
    conn = _open_pg()
    tasks: list[dict] = []
    next_id = 100  # safely past existing pg_test_21

    def emit(fn, query, params, outcome):
        nonlocal next_id
        tasks.append(_make_task(_pg_id(next_id), fn, query, params, outcome))
        next_id += 1

    try:
        bookings_tables = _pg_list_tables(conn, "bookings")
        # Snapshot the alphabetised list before any shuffling below
        list_tables_outcome = {"result": list(bookings_tables)}

        # list_tables (15) — all use bookings schema, vary phrasing
        list_tables_phrasings = [
            "What tables are available in the bookings schema?",
            "List all tables in the bookings schema.",
            "Show me every table in the bookings schema.",
            "Which tables exist in the bookings schema?",
            "Enumerate the tables under the bookings schema.",
            "Give me the full list of bookings schema tables.",
            "Display all tables defined in the bookings schema.",
            "Return the tables that live in the bookings schema.",
            "Tell me what tables the bookings schema contains.",
            "Print out all bookings tables.",
            "What are all the tables in the bookings schema?",
            "Show every table from the bookings schema.",
            "List bookings schema tables.",
            "Get a listing of bookings schema tables.",
            "Fetch the table names from the bookings schema.",
        ]
        for phrasing in list_tables_phrasings[:TARGET_PER_TOOL]:
            emit("list_tables", phrasing, {"schema": "bookings"}, list_tables_outcome)

        # describe_table (15) — pick distinct tables, cycle if needed
        rng.shuffle(bookings_tables)
        describe_phrasings = [
            "Show me the columns in the {t} table",
            "Describe the structure of the {t} table",
            "What columns does the {t} table have?",
            "Show the schema of the {t} table",
            "What are the column definitions for {t}?",
        ]
        for i in range(TARGET_PER_TOOL):
            t = bookings_tables[i % len(bookings_tables)]
            cols = _pg_describe_table(conn, t)
            phrasing = describe_phrasings[i % len(describe_phrasings)].format(t=t)
            emit("describe_table", phrasing, {"table_name": t}, {"result": cols})

        # get_row_count (15)
        rng.shuffle(bookings_tables)
        rowcount_phrasings = [
            "How many rows are in the {t} table?",
            "Count the total number of rows in {t}",
            "What is the row count for the {t} table?",
            "How many records does {t} contain?",
        ]
        for i in range(TARGET_PER_TOOL):
            t = bookings_tables[i % len(bookings_tables)]
            count = _pg_row_count(conn, t)
            phrasing = rowcount_phrasings[i % len(rowcount_phrasings)].format(t=t)
            emit("get_row_count", phrasing, {"table_name": t}, {"result": count})

        # get_foreign_keys (15)
        rng.shuffle(bookings_tables)
        fk_phrasings = [
            "What are the foreign keys on the {t} table?",
            "Show the foreign key constraints on the {t} table",
            "List the foreign keys for {t}",
            "Which tables does {t} reference via foreign keys?",
        ]
        for i in range(TARGET_PER_TOOL):
            t = bookings_tables[i % len(bookings_tables)]
            fks = _pg_foreign_keys(conn, t)
            phrasing = fk_phrasings[i % len(fk_phrasings)].format(t=t)
            emit("get_foreign_keys", phrasing, {"table_name": t}, {"result": fks})

        # find_relationships (15)
        rng.shuffle(bookings_tables)
        rel_phrasings = [
            "Find both explicit and implied relationships for the {t} table.",
            "Show the relationships (explicit and implied) for {t}.",
            "Discover all relationships involving the {t} table.",
            "What relationships does the {t} table participate in?",
        ]
        for i in range(TARGET_PER_TOOL):
            t = bookings_tables[i % len(bookings_tables)]
            rels = _pg_find_relationships(conn, t)
            phrasing = rel_phrasings[i % len(rel_phrasings)].format(t=t)
            emit("find_relationships", phrasing, {"table_name": t}, rels)

        # list_schemas (15) — default args, vary phrasing
        list_schemas_outcome = {"result": _pg_list_schemas(conn, include_system=False)}
        list_schemas_phrasings = [
            "What schemas are available in the database?",
            "List all schemas in the database.",
            "Show me every database schema.",
            "Enumerate the schemas in the database.",
            "Which schemas exist in this database?",
            "Display all available schemas.",
            "Return the schemas defined in the database.",
            "Tell me which schemas this database contains.",
            "Get the list of schemas in the database.",
            "Print all database schemas.",
            "Show the schemas in the database.",
            "What database schemas are present?",
            "List every schema in the database.",
            "Fetch the schema names.",
            "Give me all the schemas.",
        ]
        for phrasing in list_schemas_phrasings[:TARGET_PER_TOOL]:
            emit("list_schemas", phrasing, {}, list_schemas_outcome)

        # execute_query (15) — diverse SELECTs
        query_specs = [
            ("Run this query: SELECT flight_no, departure_airport, arrival_airport FROM bookings.flights ORDER BY flight_no LIMIT 5",
             "SELECT flight_no, departure_airport, arrival_airport FROM bookings.flights ORDER BY flight_no LIMIT 5"),
            ("Query the database: SELECT COUNT(*) AS total_flights FROM bookings.flights",
             "SELECT COUNT(*) AS total_flights FROM bookings.flights"),
            ("Run this SQL: SELECT book_ref, total_amount FROM bookings.bookings ORDER BY total_amount DESC LIMIT 10",
             "SELECT book_ref, total_amount FROM bookings.bookings ORDER BY total_amount DESC LIMIT 10"),
            ("Execute: SELECT COUNT(*) AS total_tickets FROM bookings.tickets",
             "SELECT COUNT(*) AS total_tickets FROM bookings.tickets"),
            ("Run this query: SELECT seat_no, fare_conditions FROM bookings.seats WHERE aircraft_code = '733' ORDER BY seat_no LIMIT 5",
             "SELECT seat_no, fare_conditions FROM bookings.seats WHERE aircraft_code = '733' ORDER BY seat_no LIMIT 5"),
            ("Query: SELECT COUNT(*) AS booking_count FROM bookings.bookings",
             "SELECT COUNT(*) AS booking_count FROM bookings.bookings"),
            ("Run: SELECT aircraft_code, range FROM bookings.aircrafts_data ORDER BY range DESC LIMIT 5",
             "SELECT aircraft_code, range FROM bookings.aircrafts_data ORDER BY range DESC LIMIT 5"),
            ("Execute SQL: SELECT COUNT(*) AS pass_count FROM bookings.boarding_passes",
             "SELECT COUNT(*) AS pass_count FROM bookings.boarding_passes"),
            ("Run this: SELECT DISTINCT status FROM bookings.flights ORDER BY status",
             "SELECT DISTINCT status FROM bookings.flights ORDER BY status"),
            ("Query: SELECT COUNT(DISTINCT departure_airport) AS airport_count FROM bookings.flights",
             "SELECT COUNT(DISTINCT departure_airport) AS airport_count FROM bookings.flights"),
            ("Run this query: SELECT COUNT(*) AS aircrafts_count FROM bookings.aircrafts_data",
             "SELECT COUNT(*) AS aircrafts_count FROM bookings.aircrafts_data"),
            ("Execute: SELECT COUNT(*) AS seats_count FROM bookings.seats",
             "SELECT COUNT(*) AS seats_count FROM bookings.seats"),
            ("Run: SELECT fare_conditions, COUNT(*) AS n FROM bookings.ticket_flights GROUP BY fare_conditions ORDER BY fare_conditions",
             "SELECT fare_conditions, COUNT(*) AS n FROM bookings.ticket_flights GROUP BY fare_conditions ORDER BY fare_conditions"),
            ("Query: SELECT COUNT(*) AS ticket_flight_count FROM bookings.ticket_flights",
             "SELECT COUNT(*) AS ticket_flight_count FROM bookings.ticket_flights"),
            ("Run this SQL: SELECT MIN(total_amount) AS min_amount, MAX(total_amount) AS max_amount FROM bookings.bookings",
             "SELECT MIN(total_amount) AS min_amount, MAX(total_amount) AS max_amount FROM bookings.bookings"),
        ]
        for query, sql_text in query_specs[:TARGET_PER_TOOL]:
            outcome = _pg_execute_query(conn, sql_text)
            # JSON-serialize-clean the rows (Decimals → strings, etc)
            outcome = json.loads(json.dumps(outcome, default=str))
            emit("execute_query", query, {"sql": sql_text}, outcome)

    finally:
        conn.close()

    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

GENERATORS = {
    "bfcl": gen_bfcl,
    "jefferson": gen_jefferson,
    "finance": gen_finance,
    "postgres": gen_postgres,
}


def main():
    global TARGET_PER_TOOL  # noqa: PLW0603
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducible task generation")
    parser.add_argument("--datasets", nargs="+", default=list(GENERATORS),
                        choices=list(GENERATORS),
                        help="Datasets to generate (default: all)")
    parser.add_argument("--per-tool", type=int, default=TARGET_PER_TOOL,
                        help=f"New tasks per tool (default: {TARGET_PER_TOOL})")
    args = parser.parse_args()
    TARGET_PER_TOOL = args.per_tool

    rng = random.Random(args.seed)

    for ds in args.datasets:
        print(f"\n[{ds}] generating tasks ...")
        gen = GENERATORS[ds]
        try:
            tasks = gen(rng)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            sys.exit(1)
        out_path = ROOT / DATASET_DIRS[ds] / "tasks_l1_extra.jsonl"
        _write_jsonl(out_path, tasks)
        print(f"  wrote {len(tasks):4d} tasks → {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
