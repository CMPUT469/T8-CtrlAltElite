#!/usr/bin/env python3
"""
Generate tasks_incremental.jsonl for each domain (bfcl, jefferson, postgres, finance).

Strategy: merge existing v1 L1+L2+L3 tasks with newly-generated L1 padding tasks so
that every tool has >=15 tasks where it is required. Expected outcomes for bfcl and
jefferson are computed at generation time using scipy / native Python so there are
no transcription errors. Finance tasks use expected_params only (API data is live).
Postgres padding is targeted to the schema-inspection tools on the 'bookings' demo DB.

Run:  .venv/bin/python3 scripts/gen_incremental_tasks.py
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import yaml
from scipy import stats as scistats

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "configs/tool_order.yaml").read_text())

TARGET_PER_TOOL = 15

DOMAIN_DIR = {
    "bfcl": "datasets/bfcl_math",
    "jefferson": "datasets/jefferson_stats",
    "postgres": "datasets/postgres",
    "finance": "datasets/finance",
}


def _round(x: float, n: int = 4) -> float:
    return round(float(x), n)


def _load_existing(domain: str) -> list[dict]:
    """Load existing v1 L1+L2+L3 tasks for a domain, tagging level."""
    d = ROOT / DOMAIN_DIR[domain]
    out: list[dict] = []
    for lvl_tag, fname in (("L1", "tasks_l1.jsonl"), ("L2", "tasks_l2.jsonl"), ("L3", "tasks_l3.jsonl")):
        p = d / fname
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            t.setdefault("level", lvl_tag)
            out.append(t)
    return out


def _count_by_tool(tasks: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for t in tasks:
        fns = t.get("functions") or ([t["function"]] if "function" in t else [])
        for f in fns:
            counts[f] += 1
    return counts


# ════════════════════════════════════════════════════════════════════════════
# bfcl (math)
# ════════════════════════════════════════════════════════════════════════════

# Each entry: (query, expected_params, expected_result)
_BFCL_POOLS = {
    "add": [
        ("Add 47 and 83.", {"a": 47, "b": 83}),
        ("What is 125 plus 376?", {"a": 125, "b": 376}),
        ("Compute the sum of 89 and 211.", {"a": 89, "b": 211}),
        ("A team scored 34 in the first half and 41 in the second. Total score?", {"a": 34, "b": 41}),
        ("Add 1250 and 875.", {"a": 1250, "b": 875}),
        ("What is 17.5 plus 42.3?", {"a": 17.5, "b": 42.3}),
        ("Compute 999 + 1.", {"a": 999, "b": 1}),
        ("Sum of 3.14 and 2.86?", {"a": 3.14, "b": 2.86}),
        ("Add 620 and 380.", {"a": 620, "b": 380}),
        ("What is 55 plus 145?", {"a": 55, "b": 145}),
        ("Add -25 and 75.", {"a": -25, "b": 75}),
        ("Compute 0.5 + 0.25.", {"a": 0.5, "b": 0.25}),
        ("Add 10000 and 2500.", {"a": 10000, "b": 2500}),
        ("What is 7 plus 13?", {"a": 7, "b": 13}),
        ("Sum 150 and 250.", {"a": 150, "b": 250}),
    ],
    "subtract": [
        ("Subtract 29 from 114.", {"a": 114, "b": 29}),
        ("What is 500 minus 175?", {"a": 500, "b": 175}),
        ("Compute 1000 - 237.", {"a": 1000, "b": 237}),
        ("A jar had 84 candies; 29 were eaten. How many remain?", {"a": 84, "b": 29}),
        ("Subtract 15 from 100.", {"a": 100, "b": 15}),
        ("What is 60.5 minus 25.3?", {"a": 60.5, "b": 25.3}),
        ("Compute 7 - 12.", {"a": 7, "b": 12}),
        ("Subtract 0.75 from 2.5.", {"a": 2.5, "b": 0.75}),
        ("What is 3500 minus 1750?", {"a": 3500, "b": 1750}),
        ("Compute 88 - 88.", {"a": 88, "b": 88}),
        ("Subtract -10 from 40.", {"a": 40, "b": -10}),
        ("What is 999 minus 1?", {"a": 999, "b": 1}),
        ("Compute 250 - 125.", {"a": 250, "b": 125}),
        ("Subtract 33 from 77.", {"a": 77, "b": 33}),
        ("What is 1.5 minus 0.5?", {"a": 1.5, "b": 0.5}),
    ],
    "multiply": [
        ("Multiply 13 by 17.", {"a": 13, "b": 17}),
        ("Compute 25 times 40.", {"a": 25, "b": 40}),
        ("What is 7 * 8?", {"a": 7, "b": 8}),
        ("Multiply 125 by 4.", {"a": 125, "b": 4}),
        ("Compute 9 * 11.", {"a": 9, "b": 11}),
        ("What is 2.5 times 4?", {"a": 2.5, "b": 4}),
        ("Multiply 50 by 50.", {"a": 50, "b": 50}),
        ("Compute 3 × 33.", {"a": 3, "b": 33}),
        ("A classroom has 24 rows of 15 desks. Total desks?", {"a": 24, "b": 15}),
        ("Multiply 100 by 0.08.", {"a": 100, "b": 0.08}),
        ("What is 12 times 12?", {"a": 12, "b": 12}),
        ("Compute 6 * 250.", {"a": 6, "b": 250}),
        ("Multiply 45 by 20.", {"a": 45, "b": 20}),
        ("What is -5 times 8?", {"a": -5, "b": 8}),
        ("Compute 1.5 * 6.", {"a": 1.5, "b": 6}),
    ],
    "divide": [
        ("Divide 144 by 12.", {"a": 144, "b": 12}),
        ("What is 250 divided by 25?", {"a": 250, "b": 25}),
        ("Compute 100 / 4.", {"a": 100, "b": 4}),
        ("Divide 999 by 3.", {"a": 999, "b": 3}),
        ("A bag of 60 marbles is split evenly among 5 kids. How many each?", {"a": 60, "b": 5}),
        ("What is 88 divided by 8?", {"a": 88, "b": 8}),
        ("Compute 7.5 / 2.5.", {"a": 7.5, "b": 2.5}),
        ("Divide 1000 by 40.", {"a": 1000, "b": 40}),
        ("What is 81 divided by 9?", {"a": 81, "b": 9}),
        ("Compute 36 / 6.", {"a": 36, "b": 6}),
        ("Divide 500 by 20.", {"a": 500, "b": 20}),
        ("What is 2.4 divided by 0.6?", {"a": 2.4, "b": 0.6}),
        ("Compute 175 / 7.", {"a": 175, "b": 7}),
        ("Divide 625 by 25.", {"a": 625, "b": 25}),
        ("What is 330 divided by 11?", {"a": 330, "b": 11}),
    ],
    "power": [
        ("Compute 2 to the power of 8.", {"base": 2, "exponent": 8}),
        ("What is 3^4?", {"base": 3, "exponent": 4}),
        ("Raise 5 to the power 3.", {"base": 5, "exponent": 3}),
        ("Compute 10^6.", {"base": 10, "exponent": 6}),
        ("What is 7 squared?", {"base": 7, "exponent": 2}),
        ("Raise 2 to the 10th power.", {"base": 2, "exponent": 10}),
        ("Compute 4^5.", {"base": 4, "exponent": 5}),
        ("What is 9 cubed?", {"base": 9, "exponent": 3}),
        ("Raise 1.5 to the power 4.", {"base": 1.5, "exponent": 4}),
        ("Compute 100^2.", {"base": 100, "exponent": 2}),
        ("What is 2 to the power 12?", {"base": 2, "exponent": 12}),
        ("Raise 8 to the power 0.", {"base": 8, "exponent": 0}),
        ("Compute 6^3.", {"base": 6, "exponent": 3}),
        ("What is 11 squared?", {"base": 11, "exponent": 2}),
        ("Raise 0.5 to the power 3.", {"base": 0.5, "exponent": 3}),
    ],
    "square_root": [
        ("Find the square root of 144.", {"number": 144}),
        ("What is sqrt(256)?", {"number": 256}),
        ("Compute the square root of 81.", {"number": 81}),
        ("Find sqrt(400).", {"number": 400}),
        ("What is the square root of 2?", {"number": 2}),
        ("Compute sqrt(625).", {"number": 625}),
        ("Find the square root of 10.", {"number": 10}),
        ("What is sqrt(121)?", {"number": 121}),
        ("Compute the square root of 64.", {"number": 64}),
        ("Find sqrt(900).", {"number": 900}),
        ("What is the square root of 196?", {"number": 196}),
        ("Compute sqrt(2500).", {"number": 2500}),
        ("Find the square root of 1.", {"number": 1}),
        ("What is sqrt(49)?", {"number": 49}),
        ("Compute the square root of 0.25.", {"number": 0.25}),
    ],
    "absolute_value": [
        ("What is the absolute value of -47?", {"number": -47}),
        ("Compute |125|.", {"number": 125}),
        ("Find the absolute value of -3.14.", {"number": -3.14}),
        ("What is |-0.5|?", {"number": -0.5}),
        ("Compute the absolute value of 0.", {"number": 0}),
        ("Find |-999|.", {"number": -999}),
        ("What is the absolute value of 17?", {"number": 17}),
        ("Compute |-250|.", {"number": -250}),
        ("Find the absolute value of -1.75.", {"number": -1.75}),
        ("What is |42|?", {"number": 42}),
        ("Compute the absolute value of -88.", {"number": -88}),
        ("Find |-1000|.", {"number": -1000}),
        ("What is the absolute value of 6.28?", {"number": 6.28}),
        ("Compute |-0.001|.", {"number": -0.001}),
        ("Find the absolute value of -72.", {"number": -72}),
    ],
    "round_number": [
        ("Round 3.14159 to 2 decimal places.", {"number": 3.14159, "decimal_places": 2}),
        ("Round 2.71828 to 3 decimal places.", {"number": 2.71828, "decimal_places": 3}),
        ("Round 1.5 to the nearest whole number.", {"number": 1.5, "decimal_places": 0}),
        ("Round 99.999 to 1 decimal place.", {"number": 99.999, "decimal_places": 1}),
        ("Round 42.3456 to 2 decimal places.", {"number": 42.3456, "decimal_places": 2}),
        ("Round 7.5 to 0 decimal places.", {"number": 7.5, "decimal_places": 0}),
        ("Round 0.123456 to 4 decimal places.", {"number": 0.123456, "decimal_places": 4}),
        ("Round 1000.567 to 1 decimal place.", {"number": 1000.567, "decimal_places": 1}),
        ("Round 3.6789 to 2 decimal places.", {"number": 3.6789, "decimal_places": 2}),
        ("Round 50.55 to 1 decimal place.", {"number": 50.55, "decimal_places": 1}),
        ("Round 123.456 to 0 decimal places.", {"number": 123.456, "decimal_places": 0}),
        ("Round 0.9999 to 2 decimal places.", {"number": 0.9999, "decimal_places": 2}),
        ("Round 6.283185 to 3 decimal places.", {"number": 6.283185, "decimal_places": 3}),
        ("Round 2.5 to 0 decimal places.", {"number": 2.5, "decimal_places": 0}),
        ("Round 99.95 to 1 decimal place.", {"number": 99.95, "decimal_places": 1}),
    ],
    "percentage": [
        ("What percent is 45 of 180?", {"part": 45, "whole": 180}),
        ("Compute 25 as a percentage of 200.", {"part": 25, "whole": 200}),
        ("What is 80 out of 400 as a percent?", {"part": 80, "whole": 400}),
        ("A student scored 68 out of 85 - percentage?", {"part": 68, "whole": 85}),
        ("What percent is 12 of 60?", {"part": 12, "whole": 60}),
        ("Compute 150 as a percentage of 600.", {"part": 150, "whole": 600}),
        ("What is 7 out of 20 as a percent?", {"part": 7, "whole": 20}),
        ("3 of 50 people responded. What is the percentage?", {"part": 3, "whole": 50}),
        ("What percent is 33 of 44?", {"part": 33, "whole": 44}),
        ("Compute 90 as a percentage of 120.", {"part": 90, "whole": 120}),
        ("What is 5 out of 8 as a percent?", {"part": 5, "whole": 8}),
        ("200 of 500 customers subscribed. Percentage?", {"part": 200, "whole": 500}),
        ("What percent is 1 of 3?", {"part": 1, "whole": 3}),
        ("Compute 18 as a percentage of 24.", {"part": 18, "whole": 24}),
        ("What is 45 out of 90 as a percent?", {"part": 45, "whole": 90}),
    ],
    "sum_values": [
        ("Sum the values [10, 20, 30, 40, 50].", {"numbers": [10, 20, 30, 40, 50]}),
        ("Compute the total of [5, 15, 25, 35].", {"numbers": [5, 15, 25, 35]}),
        ("What is the sum of [100, 200, 300]?", {"numbers": [100, 200, 300]}),
        ("Sum [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].", {"numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}),
        ("Compute the total of [12.5, 7.5, 20.0].", {"numbers": [12.5, 7.5, 20.0]}),
        ("What is the sum of [99, 1, 50, 50]?", {"numbers": [99, 1, 50, 50]}),
        ("Add up [3, 7, 11, 15, 19].", {"numbers": [3, 7, 11, 15, 19]}),
        ("Sum [1000, 2000, 3000].", {"numbers": [1000, 2000, 3000]}),
        ("Compute the total of [-5, 10, -15, 20].", {"numbers": [-5, 10, -15, 20]}),
        ("What is the sum of [0.1, 0.2, 0.3, 0.4]?", {"numbers": [0.1, 0.2, 0.3, 0.4]}),
        ("Sum the list [25, 25, 25, 25].", {"numbers": [25, 25, 25, 25]}),
        ("Compute the total of [8, 16, 32, 64]?", {"numbers": [8, 16, 32, 64]}),
        ("What is the sum of [2, 4, 6, 8, 10, 12]?", {"numbers": [2, 4, 6, 8, 10, 12]}),
        ("Sum [11, 22, 33, 44, 55].", {"numbers": [11, 22, 33, 44, 55]}),
        ("Compute the total of [7, 14, 21, 28, 35].", {"numbers": [7, 14, 21, 28, 35]}),
    ],
    "mean": [
        ("Compute the mean of [10, 20, 30, 40, 50].", {"numbers": [10, 20, 30, 40, 50]}),
        ("What is the average of [5, 15, 25]?", {"numbers": [5, 15, 25]}),
        ("Find the mean of [100, 200, 300, 400].", {"numbers": [100, 200, 300, 400]}),
        ("Average of [2, 4, 6, 8, 10].", {"numbers": [2, 4, 6, 8, 10]}),
        ("Mean of [72, 75, 78, 81].", {"numbers": [72, 75, 78, 81]}),
        ("Average of [1, 1, 1, 1, 1].", {"numbers": [1, 1, 1, 1, 1]}),
        ("Mean of [3.5, 4.5, 5.5].", {"numbers": [3.5, 4.5, 5.5]}),
        ("Compute the average of [99, 100, 101].", {"numbers": [99, 100, 101]}),
        ("Mean of [-10, 0, 10, 20, 30].", {"numbers": [-10, 0, 10, 20, 30]}),
        ("Average of [12, 24, 36, 48, 60].", {"numbers": [12, 24, 36, 48, 60]}),
        ("Mean of [7, 14, 21].", {"numbers": [7, 14, 21]}),
        ("Compute the average of [0.1, 0.2, 0.3, 0.4, 0.5].", {"numbers": [0.1, 0.2, 0.3, 0.4, 0.5]}),
        ("Mean of [50, 50, 50, 50].", {"numbers": [50, 50, 50, 50]}),
        ("Find the average of [80, 90, 70, 60, 100].", {"numbers": [80, 90, 70, 60, 100]}),
        ("Mean of [11, 22, 33, 44].", {"numbers": [11, 22, 33, 44]}),
    ],
    "min_value": [
        ("Find the minimum of [5, 2, 8, 1, 9, 3].", {"numbers": [5, 2, 8, 1, 9, 3]}),
        ("What is the smallest value in [42, 17, 93, 6, 28]?", {"numbers": [42, 17, 93, 6, 28]}),
        ("Minimum of [100, 50, 200, 25].", {"numbers": [100, 50, 200, 25]}),
        ("Smallest of [3.14, 2.71, 1.41, 1.73].", {"numbers": [3.14, 2.71, 1.41, 1.73]}),
        ("Find min of [-5, -10, 0, 5, 10].", {"numbers": [-5, -10, 0, 5, 10]}),
        ("Minimum value in [88, 88, 88, 87, 88].", {"numbers": [88, 88, 88, 87, 88]}),
        ("What is the smallest in [1000, 999, 1001, 998]?", {"numbers": [1000, 999, 1001, 998]}),
        ("Min of [15, 12, 18, 9, 21].", {"numbers": [15, 12, 18, 9, 21]}),
        ("Smallest of [0.5, 0.25, 0.75, 0.1].", {"numbers": [0.5, 0.25, 0.75, 0.1]}),
        ("Find minimum of [60, 45, 80, 30, 55, 70].", {"numbers": [60, 45, 80, 30, 55, 70]}),
        ("Min in [7, 3, 11, 5, 13].", {"numbers": [7, 3, 11, 5, 13]}),
        ("What is the smallest value in [250, 175, 300, 125]?", {"numbers": [250, 175, 300, 125]}),
        ("Minimum of [4.2, 3.8, 4.5, 4.0].", {"numbers": [4.2, 3.8, 4.5, 4.0]}),
        ("Smallest of [99, 100, 98, 101, 97].", {"numbers": [99, 100, 98, 101, 97]}),
        ("Find min of [500, 400, 300, 200, 100].", {"numbers": [500, 400, 300, 200, 100]}),
    ],
    "max_value": [
        ("Find the maximum of [5, 2, 8, 1, 9, 3].", {"numbers": [5, 2, 8, 1, 9, 3]}),
        ("What is the largest value in [42, 17, 93, 6, 28]?", {"numbers": [42, 17, 93, 6, 28]}),
        ("Maximum of [100, 50, 200, 25].", {"numbers": [100, 50, 200, 25]}),
        ("Largest of [3.14, 2.71, 1.41, 1.73].", {"numbers": [3.14, 2.71, 1.41, 1.73]}),
        ("Find max of [-5, -10, 0, 5, 10].", {"numbers": [-5, -10, 0, 5, 10]}),
        ("Maximum value in [88, 88, 88, 87, 88].", {"numbers": [88, 88, 88, 87, 88]}),
        ("What is the largest in [1000, 999, 1001, 998]?", {"numbers": [1000, 999, 1001, 998]}),
        ("Max of [15, 12, 18, 9, 21].", {"numbers": [15, 12, 18, 9, 21]}),
        ("Largest of [0.5, 0.25, 0.75, 0.1].", {"numbers": [0.5, 0.25, 0.75, 0.1]}),
        ("Find maximum of [60, 45, 80, 30, 55, 70].", {"numbers": [60, 45, 80, 30, 55, 70]}),
        ("Max in [7, 3, 11, 5, 13].", {"numbers": [7, 3, 11, 5, 13]}),
        ("What is the largest value in [250, 175, 300, 125]?", {"numbers": [250, 175, 300, 125]}),
        ("Maximum of [4.2, 3.8, 4.5, 4.0].", {"numbers": [4.2, 3.8, 4.5, 4.0]}),
        ("Largest of [99, 100, 98, 101, 97].", {"numbers": [99, 100, 98, 101, 97]}),
        ("Find max of [500, 400, 300, 200, 100].", {"numbers": [500, 400, 300, 200, 100]}),
    ],
    "standard_deviation": [
        ("Compute the standard deviation of [2, 4, 4, 4, 5, 5, 7, 9].", {"numbers": [2, 4, 4, 4, 5, 5, 7, 9]}),
        ("Standard deviation of [10, 12, 14, 16, 18].", {"numbers": [10, 12, 14, 16, 18]}),
        ("Compute std dev of [5, 5, 5, 5, 5].", {"numbers": [5, 5, 5, 5, 5]}),
        ("Standard deviation of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].", {"numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}),
        ("Compute the std dev of [100, 102, 98, 101, 99].", {"numbers": [100, 102, 98, 101, 99]}),
        ("Standard deviation of [20, 30, 40, 50].", {"numbers": [20, 30, 40, 50]}),
        ("Compute std dev of [3.2, 3.5, 3.8, 4.1, 4.4].", {"numbers": [3.2, 3.5, 3.8, 4.1, 4.4]}),
        ("Standard deviation of [72, 75, 78, 81, 84].", {"numbers": [72, 75, 78, 81, 84]}),
        ("Compute the std dev of [0, 10, 20, 30, 40, 50].", {"numbers": [0, 10, 20, 30, 40, 50]}),
        ("Standard deviation of [15, 15, 20, 25, 25].", {"numbers": [15, 15, 20, 25, 25]}),
        ("Compute std dev of [7, 8, 9, 10, 11, 12].", {"numbers": [7, 8, 9, 10, 11, 12]}),
        ("Standard deviation of [50, 55, 45, 60, 40].", {"numbers": [50, 55, 45, 60, 40]}),
        ("Compute the std dev of [200, 220, 240, 260].", {"numbers": [200, 220, 240, 260]}),
        ("Standard deviation of [1.1, 1.2, 1.3, 1.4, 1.5].", {"numbers": [1.1, 1.2, 1.3, 1.4, 1.5]}),
        ("Compute std dev of [99, 101, 100, 98, 102].", {"numbers": [99, 101, 100, 98, 102]}),
    ],
    "logarithm": [
        ("Compute log base 10 of 100.", {"number": 100, "base": 10}),
        ("What is ln(e)?", {"number": math.e, "base": math.e}),
        ("Compute log base 2 of 8.", {"number": 8, "base": 2}),
        ("What is log base 10 of 1000?", {"number": 1000, "base": 10}),
        ("Compute log base 2 of 1024.", {"number": 1024, "base": 2}),
        ("What is log base 10 of 1?", {"number": 1, "base": 10}),
        ("Compute log base 5 of 25.", {"number": 25, "base": 5}),
        ("What is log base 3 of 27?", {"number": 27, "base": 3}),
        ("Compute log base 10 of 10000.", {"number": 10000, "base": 10}),
        ("What is log base 2 of 16?", {"number": 16, "base": 2}),
        ("Compute log base 10 of 0.1.", {"number": 0.1, "base": 10}),
        ("What is log base 4 of 64?", {"number": 64, "base": 4}),
        ("Compute log base 2 of 2.", {"number": 2, "base": 2}),
        ("What is log base 10 of 50?", {"number": 50, "base": 10}),
        ("Compute log base 7 of 49.", {"number": 49, "base": 7}),
    ],
}


def _bfcl_result(fn: str, params: dict):
    if fn == "add":
        return params["a"] + params["b"]
    if fn == "subtract":
        return params["a"] - params["b"]
    if fn == "multiply":
        return params["a"] * params["b"]
    if fn == "divide":
        return params["a"] / params["b"]
    if fn == "power":
        return params["base"] ** params["exponent"]
    if fn == "square_root":
        return round(math.sqrt(params["number"]), 2)
    if fn == "absolute_value":
        return abs(params["number"])
    if fn == "round_number":
        return round(params["number"], params["decimal_places"])
    if fn == "percentage":
        return (params["part"] / params["whole"]) * 100
    if fn == "sum_values":
        return sum(params["numbers"])
    if fn == "mean":
        return sum(params["numbers"]) / len(params["numbers"])
    if fn == "min_value":
        return min(params["numbers"])
    if fn == "max_value":
        return max(params["numbers"])
    if fn == "standard_deviation":
        # Tool uses sample std dev (n-1 denominator)
        return statistics.stdev(params["numbers"])
    if fn == "logarithm":
        return math.log(params["number"], params["base"])
    raise ValueError(fn)


def gen_bfcl_padding(gap_per_tool: dict[str, int]) -> list[dict]:
    tasks: list[dict] = []
    idx = 1
    for tool, pool in _BFCL_POOLS.items():
        needed = gap_per_tool.get(tool, 0)
        if needed <= 0:
            continue
        for q, params in pool[:needed]:
            result = _bfcl_result(tool, params)
            if isinstance(result, float):
                result = _round(result, 6)
            tasks.append({
                "id": f"math_inc_{idx:03d}",
                "level": "L1",
                "function": tool,
                "query": q,
                "expected_params": params,
                "expected_outcome": {"result": result},
                "optimal_steps": 1,
            })
            idx += 1
    return tasks


# ════════════════════════════════════════════════════════════════════════════
# jefferson (stats)
# ════════════════════════════════════════════════════════════════════════════

# Reusable data series
_S1 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
_S2 = [15, 17, 20, 22, 25, 19, 21, 18, 23, 16]
_S3 = [100, 102, 98, 105, 97, 103, 99, 101, 104, 96]
_S4 = [72, 75, 71, 78, 69, 74, 76, 73, 70, 77, 72, 75]
_S5 = [3.2, 3.5, 3.8, 4.1, 4.4, 4.7, 5.0, 5.3]
_S6 = [50, 52, 48, 55, 47, 53, 49, 51, 54, 46]
_S7 = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
_S8 = [25, 27, 30, 28, 29, 26, 31, 24, 32, 33]
_S9 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
_S10 = [88, 92, 85, 90, 87, 91, 86, 93, 89, 84]
_S11 = [5, 7, 9, 11, 13, 15, 17]
_S12 = [200, 210, 195, 205, 215, 190, 220, 198, 212]
_S13 = [12, 14, 16, 18, 20, 22, 24]
_S14 = [45, 48, 52, 50, 47, 51, 49, 46]
_S15 = [80, 82, 84, 86, 88, 90, 92, 94, 96, 98]
_S16 = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]  # with mode
_S17 = [1, 2, 2, 3, 3, 3, 4, 4, 5]  # with mode
_S18 = [10, 15, 10, 20, 10, 25, 10, 30]  # strong mode
_S19 = [7, 7, 7, 8, 8, 9, 10, 10]
_S20 = [100, 100, 100, 200, 200, 300]


_SERIES_POOL = [_S1, _S2, _S3, _S4, _S5, _S6, _S7, _S8, _S9, _S10,
                _S11, _S12, _S13, _S14, _S15, _S16, _S17, _S18, _S19, _S20]

# Dedicated pool for calculate_mode — every series must have at least one repeated value.
_MODE_SERIES_POOL = [
    [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],       # mode=5
    [1, 2, 2, 3, 3, 3, 4, 4, 5],              # mode=3
    [10, 15, 10, 20, 10, 25, 10, 30],          # mode=10
    [7, 7, 7, 8, 8, 9, 10, 10],               # mode=7
    [100, 100, 100, 200, 200, 300],            # mode=100
    [2, 4, 4, 4, 5, 5, 7, 9],                 # mode=4
    [8, 8, 9, 9, 9, 10, 11, 11],              # mode=9
    [1, 1, 2, 3, 3, 3, 4, 5, 5],              # mode=3
    [20, 20, 20, 30, 40, 40, 50],             # mode=20
    [6, 6, 7, 7, 7, 8, 9, 9],                # mode=7
    [50, 50, 60, 70, 70, 70, 80],             # mode=70
    [1, 2, 2, 3, 4, 4, 4, 5, 6],             # mode=4
    [15, 15, 15, 20, 25, 30, 30],             # mode=15
    [5, 5, 6, 6, 6, 7, 8, 9, 9],             # mode=6
    [100, 200, 200, 200, 300, 400, 400],      # mode=200
]

_JEFFERSON_QUERIES = {
    "calculate_median": [
        "Find the median of {data}.",
        "What is the median of {data}?",
        "Compute the median value for {data}.",
        "Given {data}, return the median.",
        "Median of {data}?",
    ],
    "calculate_mode": [
        "Find the mode of {data}.",
        "What is the most frequent value in {data}?",
        "Compute the mode of {data}.",
        "Given {data}, return the mode.",
        "Mode of {data}?",
    ],
    "calculate_range": [
        "Compute the range of {data}.",
        "What is max minus min for {data}?",
        "Find the range of values in {data}.",
        "Given {data}, report the range.",
        "Range of {data}?",
    ],
    "calculate_variance": [
        "Compute the variance of {data}.",
        "What is the variance of {data}?",
        "Find the variance for {data}.",
        "Given {data}, compute variance.",
        "Variance of {data}?",
    ],
    "calculate_quartiles": [
        "Compute Q1, median, and Q3 for {data}.",
        "Find the quartiles of {data}.",
        "What are the quartile values of {data}?",
        "Given {data}, report the three quartiles.",
        "Quartiles for {data}?",
    ],
    "calculate_iqr": [
        "Compute the IQR of {data}.",
        "What is the interquartile range of {data}?",
        "Find the IQR for {data}.",
        "Given {data}, compute Q3-Q1.",
        "Interquartile range of {data}?",
    ],
    "calculate_skewness": [
        "Compute the skewness of {data}.",
        "What is the skewness of {data}?",
        "Find the skewness measure for {data}.",
        "Given {data}, compute skewness.",
        "Skewness of {data}?",
    ],
    "calculate_kurtosis": [
        "Compute the kurtosis of {data}.",
        "What is the kurtosis of {data}?",
        "Find the kurtosis measure for {data}.",
        "Given {data}, compute kurtosis.",
        "Kurtosis of {data}?",
    ],
    "calculate_z_scores": [
        "Compute z-scores for {data}.",
        "What are the z-scores of {data}?",
        "Standardize {data} into z-scores.",
        "Given {data}, compute z-scores.",
        "Z-scores for {data}?",
    ],
    "calculate_confidence_interval": [
        "Compute a 95% confidence interval for the mean of {data}.",
        "What is the 95% CI for the mean of {data}?",
        "Find a 95% confidence interval on the mean of {data}.",
        "Given {data}, estimate a 95% CI for the mean.",
        "95% confidence interval for mean of {data}?",
    ],
    "perform_t_test": [
        "Perform a one-sample t-test on {data} against popmean 10.",
        "Test whether the mean of {data} differs from 10.",
        "Run a t-test on {data} vs population mean 10.",
        "Given {data}, test H0: mean=10.",
        "One-sample t-test of {data} against 10?",
    ],
    "detect_outliers": [
        "Detect outliers in {data}.",
        "Find any outlier values in {data}.",
        "Flag outliers in {data} using the IQR method.",
        "Given {data}, identify outliers.",
        "Outliers in {data}?",
    ],
    "perform_normality_test": [
        "Check whether {data} is normally distributed.",
        "Run a normality test on {data}.",
        "Test {data} for normality.",
        "Given {data}, assess normality.",
        "Is {data} normally distributed?",
    ],
    "calculate_moving_average": [
        "Compute the 3-period moving average of {data}.",
        "Smooth {data} with a 3-period moving average.",
        "Apply a 3-window moving average to {data}.",
        "Given {data}, compute the 3-period moving average.",
        "3-period moving average of {data}?",
    ],
    "generate_descriptive_statistics": [
        "Produce descriptive statistics for {data}.",
        "Summarize {data} with descriptive statistics.",
        "Provide a statistical summary of {data}.",
        "Given {data}, generate descriptive statistics.",
        "Descriptive statistics for {data}?",
    ],
}

# paired-data tasks
_PAIRED_QUERIES = {
    "calculate_correlation": [
        "Compute the correlation between {x} and {y}.",
        "What is the correlation of {x} and {y}?",
        "Find the Pearson correlation for {x} and {y}.",
        "Given x={x} and y={y}, compute correlation.",
        "Correlation between {x} and {y}?",
    ],
    "calculate_covariance": [
        "Compute the covariance between {x} and {y}.",
        "What is the covariance of {x} and {y}?",
        "Find cov(x, y) for x={x}, y={y}.",
        "Given x={x} and y={y}, compute covariance.",
        "Covariance between {x} and {y}?",
    ],
    "perform_linear_regression": [
        "Fit a linear regression with x={x} and y={y}.",
        "Perform linear regression on x={x}, y={y}.",
        "Run linear regression for x={x} and y={y}.",
        "Given x={x} and y={y}, estimate the regression line.",
        "Linear regression of y={y} on x={x}?",
    ],
}

_PAIRED_POOL = [
    ([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]),
    ([10, 20, 30, 40, 50], [15, 25, 35, 45, 55]),
    ([1, 2, 3, 4, 5, 6], [3, 7, 5, 11, 9, 13]),
    ([2, 4, 6, 8], [1, 5, 9, 13]),
    ([5, 10, 15, 20, 25], [12, 18, 28, 32, 40]),
    ([1, 3, 5, 7, 9], [10, 20, 30, 40, 50]),
    ([10, 15, 20, 25, 30, 35], [100, 145, 205, 245, 305, 360]),
    ([2, 3, 5, 7, 11], [4, 9, 25, 49, 121]),
    ([0, 1, 2, 3, 4, 5], [1, 3, 5, 7, 9, 11]),
    ([1, 2, 3, 4, 5, 6, 7], [2, 3, 5, 7, 11, 13, 17]),
    ([10, 20, 30], [5, 15, 25]),
    ([1, 4, 9, 16, 25], [1, 2, 3, 4, 5]),
    ([100, 200, 300, 400], [50, 100, 150, 200]),
    ([5, 6, 7, 8, 9, 10], [25, 36, 49, 64, 81, 100]),
    ([2, 4, 8, 16, 32], [1, 2, 3, 4, 5]),
]


def _jeff_outcome(tool: str, params: dict):
    data = params.get("collection")
    if tool == "calculate_median":
        return {"result": _round(statistics.median(data))}
    if tool == "calculate_mode":
        modes = statistics.multimode(data)
        if len(modes) == 1:
            return {"result": float(modes[0])}
        return {"result": [float(m) for m in modes]}
    if tool == "calculate_range":
        return {"result": _round(max(data) - min(data))}
    if tool == "calculate_variance":
        n = len(data)
        m = sum(data) / n
        var = sum((x - m) ** 2 for x in data) / n
        return {"result": _round(var)}
    if tool == "calculate_quartiles":
        sd = sorted(data)
        n = len(sd)
        q1 = sd[n // 4]
        q2 = sd[n // 2] if n % 2 else (sd[n // 2 - 1] + sd[n // 2]) / 2
        q3 = sd[3 * n // 4]
        return {"q1": _round(q1), "q2": _round(q2), "q3": _round(q3)}
    if tool == "calculate_iqr":
        sd = sorted(data)
        n = len(sd)
        q1 = sd[n // 4]
        q3 = sd[3 * n // 4]
        return {"result": _round(q3 - q1)}
    if tool == "calculate_skewness":
        return {"result": _round(scistats.skew(data))}
    if tool == "calculate_kurtosis":
        return {"result": _round(scistats.kurtosis(data))}
    if tool == "calculate_z_scores":
        zs = scistats.zscore(data)
        return {"z_scores": [_round(z) for z in zs]}
    if tool == "calculate_confidence_interval":
        m = sum(data) / len(data)
        sem = scistats.sem(data)
        conf = params.get("confidence", 0.95)
        interval = sem * scistats.t.ppf((1 + conf) / 2, len(data) - 1)
        return {"lower_bound": _round(m - interval), "upper_bound": _round(m + interval)}
    if tool == "perform_t_test":
        popmean = params.get("popmean", 0)
        t, p = scistats.ttest_1samp(data, popmean)
        return {"t_statistic": _round(t), "p_value": _round(p)}
    if tool == "detect_outliers":
        sd = sorted(data)
        n = len(sd)
        q1 = sd[n // 4]
        q3 = sd[3 * n // 4]
        iqr = q3 - q1
        lb = q1 - 1.5 * iqr
        ub = q3 + 1.5 * iqr
        outs = [x for x in data if x < lb or x > ub]
        return {"outliers": outs, "count": len(outs)}
    if tool == "perform_normality_test":
        _, p = scistats.shapiro(data)
        return {"interpretation": "normally distributed" if p > 0.05 else "not normally distributed"}
    if tool == "calculate_moving_average":
        w = params["window_size"]
        res = [sum(data[i:i + w]) / w for i in range(len(data) - w + 1)]
        return {"result": [_round(v) for v in res]}
    if tool == "generate_descriptive_statistics":
        n = len(data)
        m = sum(data) / n
        return {"count": n, "mean": _round(m)}
    raise ValueError(tool)


def _paired_outcome(tool: str, x: list[float], y: list[float]):
    if tool == "calculate_correlation":
        corr, p = scistats.pearsonr(x, y)
        return {"correlation": _round(corr)}
    if tool == "calculate_covariance":
        n = len(x)
        mx = sum(x) / n
        my = sum(y) / n
        cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
        return {"result": _round(cov)}
    if tool == "perform_linear_regression":
        r = scistats.linregress(x, y)
        return {"slope": _round(r.slope), "intercept": _round(r.intercept), "r_value": _round(r.rvalue)}
    raise ValueError(tool)


def gen_jefferson_padding(gap_per_tool: dict[str, int]) -> list[dict]:
    tasks: list[dict] = []
    idx = 1
    # Single-series tools
    for tool, queries in _JEFFERSON_QUERIES.items():
        needed = gap_per_tool.get(tool, 0)
        if needed <= 0:
            continue
        for k in range(needed):
            # Mode requires repeated values — use dedicated pool to avoid all-unique series.
            pool = _MODE_SERIES_POOL if tool == "calculate_mode" else _SERIES_POOL
            series = pool[k % len(pool)]
            qtmpl = queries[k % len(queries)]
            params = {"collection": list(series)}
            if tool == "perform_t_test":
                params["popmean"] = 10
            if tool == "calculate_confidence_interval":
                params["confidence"] = 0.95
            if tool == "calculate_moving_average":
                params["window_size"] = 3
            try:
                outcome = _jeff_outcome(tool, params)
            except Exception as e:
                print(f"  [skip] {tool} on {series}: {e}")
                continue
            tasks.append({
                "id": f"stats_inc_{idx:03d}",
                "level": "L1",
                "function": tool,
                "query": qtmpl.format(data=series),
                "expected_params": params,
                "expected_outcome": outcome,
                "optimal_steps": 1,
            })
            idx += 1
    # Paired-series tools
    for tool, queries in _PAIRED_QUERIES.items():
        needed = gap_per_tool.get(tool, 0)
        if needed <= 0:
            continue
        for k in range(needed):
            x, y = _PAIRED_POOL[k % len(_PAIRED_POOL)]
            qtmpl = queries[k % len(queries)]
            if tool == "perform_linear_regression":
                params = {"x": list(x), "y": list(y)}
            else:
                params = {"collection1": list(x), "collection2": list(y)}
            try:
                outcome = _paired_outcome(tool, x, y)
            except Exception as e:
                print(f"  [skip] {tool}: {e}")
                continue
            tasks.append({
                "id": f"stats_inc_{idx:03d}",
                "level": "L1",
                "function": tool,
                "query": qtmpl.format(x=x, y=y),
                "expected_params": params,
                "expected_outcome": outcome,
                "optimal_steps": 1,
            })
            idx += 1
    return tasks


# ════════════════════════════════════════════════════════════════════════════
# finance (no expected_outcome — API is live)
# ════════════════════════════════════════════════════════════════════════════

_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "JPM", "V",
            "WMT", "DIS", "KO", "PEP", "INTC", "AMD"]
_CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD", "XRP-USD",
           "MATIC-USD", "DOT-USD", "AVAX-USD", "LINK-USD", "LTC-USD", "BCH-USD",
           "ATOM-USD", "UNI-USD", "ALGO-USD"]

_FINANCE_STMT_PERIODS = [("annual", 2), ("annual", 3), ("annual", 4),
                         ("quarterly", 3), ("quarterly", 4), ("quarterly", 5)]

_FINANCE_DATE_RANGES = [
    ("2025-01-01", "2025-01-31"),
    ("2025-02-01", "2025-02-28"),
    ("2024-12-01", "2024-12-31"),
    ("2024-11-01", "2024-11-30"),
    ("2024-10-01", "2024-10-31"),
    ("2024-09-01", "2024-09-30"),
    ("2024-08-01", "2024-08-31"),
    ("2024-07-01", "2024-07-31"),
    ("2024-06-01", "2024-06-30"),
    ("2024-05-01", "2024-05-31"),
    ("2024-04-01", "2024-04-30"),
    ("2024-03-01", "2024-03-31"),
    ("2025-03-01", "2025-03-31"),
    ("2024-02-01", "2024-02-29"),
    ("2024-01-01", "2024-01-31"),
]

_FILING_TYPES = ["10-K", "10-Q", "8-K"]
_FILING_ITEMS_10K = ["Risk Factors", "Business", "Management Discussion and Analysis",
                     "Financial Statements", "Properties", "Legal Proceedings",
                     "Controls and Procedures"]
_FILING_DATES_10K = [("AAPL", "2024-11-01"), ("MSFT", "2024-11-01"),
                     ("NVDA", "2024-03-20"), ("TSLA", "2025-01-31"),
                     ("GOOGL", "2025-02-04")]


def _fin_task(idx: int, fn: str, query: str, params: dict) -> dict:
    return {
        "id": f"finance_inc_{idx:03d}",
        "level": "L1",
        "function": fn,
        "query": query,
        "expected_params": params,
        "optimal_steps": 1,
    }


def gen_finance_padding(gap_per_tool: dict[str, int]) -> list[dict]:
    tasks: list[dict] = []
    idx = 1

    stmt_tools = ["get_income_statements", "get_balance_sheets", "get_cash_flow_statements",
                  "getAnalystEstimates", "getFinancialMetrics", "getSegmentedRevenues"]
    for tool in stmt_tools:
        needed = gap_per_tool.get(tool, 0)
        for k in range(needed):
            ticker = _TICKERS[k % len(_TICKERS)]
            period, limit = _FINANCE_STMT_PERIODS[k % len(_FINANCE_STMT_PERIODS)]
            label = tool.replace("get_", "").replace("_", " ").replace("get", "")
            q = f"Retrieve the latest {limit} {period} records of {label} for {ticker}."
            tasks.append(_fin_task(idx, tool, q, {"ticker": ticker, "period": period, "limit": limit}))
            idx += 1

    for tool in ("get_current_stock_price", "getFinancialMetricsSnapshot", "getCompanyFacts",
                 "get_company_news"):
        needed = gap_per_tool.get(tool, 0)
        for k in range(needed):
            ticker = _TICKERS[k % len(_TICKERS)]
            if tool == "get_current_stock_price":
                q = f"What is the current stock price for {ticker}?"
            elif tool == "getFinancialMetricsSnapshot":
                q = f"Retrieve the latest financial metrics snapshot for {ticker}."
            elif tool == "getCompanyFacts":
                q = f"Retrieve company facts for {ticker}."
            else:
                q = f"Retrieve recent company news for {ticker}."
            tasks.append(_fin_task(idx, tool, q, {"ticker": ticker}))
            idx += 1

    needed = gap_per_tool.get("get_historical_stock_prices", 0)
    for k in range(needed):
        ticker = _TICKERS[k % len(_TICKERS)]
        start, end = _FINANCE_DATE_RANGES[k % len(_FINANCE_DATE_RANGES)]
        q = f"Retrieve daily stock prices for {ticker} from {start} to {end}."
        tasks.append(_fin_task(idx, "get_historical_stock_prices", q, {
            "ticker": ticker, "start_date": start, "end_date": end,
            "interval": "day", "interval_multiplier": 1,
        }))
        idx += 1

    needed = gap_per_tool.get("get_sec_filings", 0)
    for k in range(needed):
        ticker = _TICKERS[k % len(_TICKERS)]
        limit = [3, 4, 5, 6, 8, 10][k % 6]
        q = f"List the latest {limit} SEC filings for {ticker}."
        tasks.append(_fin_task(idx, "get_sec_filings", q, {"ticker": ticker, "limit": limit}))
        idx += 1

    needed = gap_per_tool.get("getAvailableFilingItems", 0)
    for k in range(needed):
        ft = _FILING_TYPES[k % len(_FILING_TYPES)]
        q = f"List the extractable filing items for {ft} filings."
        tasks.append(_fin_task(idx, "getAvailableFilingItems", q, {"filing_type": ft}))
        idx += 1

    needed = gap_per_tool.get("getFilingItems", 0)
    for k in range(needed):
        ticker, fdate = _FILING_DATES_10K[k % len(_FILING_DATES_10K)]
        item = _FILING_ITEMS_10K[k % len(_FILING_ITEMS_10K)]
        q = f"Extract the {item} section from {ticker}'s 10-K filed on {fdate}."
        tasks.append(_fin_task(idx, "getFilingItems", q, {
            "ticker": ticker, "filing_type": "10-K", "filing_date": fdate, "item": item,
        }))
        idx += 1

    needed = gap_per_tool.get("get_available_crypto_tickers", 0)
    for k in range(needed):
        qs = ["List all available cryptocurrency tickers.",
              "Enumerate every supported crypto ticker.",
              "Show the complete list of crypto tickers available.",
              "Retrieve all supported cryptocurrency symbols.",
              "What crypto tickers does the platform provide?"]
        q = qs[k % len(qs)]
        tasks.append(_fin_task(idx, "get_available_crypto_tickers", q, {}))
        idx += 1

    needed = gap_per_tool.get("get_crypto_prices", 0)
    for k in range(needed):
        c = _CRYPTO[k % len(_CRYPTO)]
        start, end = _FINANCE_DATE_RANGES[k % len(_FINANCE_DATE_RANGES)]
        q = f"Retrieve daily crypto prices for {c} from {start} to {end}."
        tasks.append(_fin_task(idx, "get_crypto_prices", q, {
            "ticker": c, "start_date": start, "end_date": end,
            "interval": "day", "interval_multiplier": 1,
        }))
        idx += 1

    needed = gap_per_tool.get("get_current_crypto_price", 0)
    for k in range(needed):
        c = _CRYPTO[k % len(_CRYPTO)]
        q = f"What is the current price of {c}?"
        tasks.append(_fin_task(idx, "get_current_crypto_price", q, {"ticker": c}))
        idx += 1

    needed = gap_per_tool.get("get_historical_crypto_prices", 0)
    for k in range(needed):
        c = _CRYPTO[k % len(_CRYPTO)]
        start, end = _FINANCE_DATE_RANGES[k % len(_FINANCE_DATE_RANGES)]
        q = f"Retrieve historical crypto prices for {c} from {start} to {end}."
        tasks.append(_fin_task(idx, "get_historical_crypto_prices", q, {
            "ticker": c, "start_date": start, "end_date": end,
            "interval": "day", "interval_multiplier": 1,
        }))
        idx += 1

    return tasks


# ════════════════════════════════════════════════════════════════════════════
# postgres (bookings schema)
# ════════════════════════════════════════════════════════════════════════════

_PG_TABLES = ["aircrafts", "airports", "boarding_passes", "bookings",
              "flights", "seats", "ticket_flights", "tickets"]


def _pg_task(idx: int, fn: str, query: str, params: dict, outcome: dict | None = None,
             optimal: int = 1) -> dict:
    t = {
        "id": f"pg_inc_{idx:03d}",
        "level": "L1",
        "query": query,
        "function": fn,
        "expected_params": params,
        "optimal_steps": optimal,
    }
    if outcome is not None:
        t["expected_outcome"] = outcome
    return t


def gen_postgres_padding(gap_per_tool: dict[str, int]) -> list[dict]:
    tasks: list[dict] = []
    idx = 1

    # list_tables — bookings is the default, omit the param (matches pg_test_0 convention).
    # For non-default schemas (public), include it explicitly.
    list_tables_tasks = [
        ("List all tables in the bookings schema.", {}),
        ("Show every table available in the bookings database.", {}),
        ("What tables exist under the bookings schema?", {}),
        ("Enumerate tables in the bookings schema.", {}),
        ("Return all bookings schema tables.", {}),
        ("Which tables are present in bookings?", {}),
        ("List the tables of the bookings schema.", {}),
        ("Show bookings schema tables.", {}),
        ("What tables does the public schema contain?", {"schema": "public"}),
        ("List tables in the public schema.", {"schema": "public"}),
    ]
    needed = gap_per_tool.get("list_tables", 0)
    for q, p in list_tables_tasks[:needed]:
        tasks.append(_pg_task(idx, "list_tables", q, p))
        idx += 1

    # describe_table
    describe_queries = [
        "Show the columns of the {t} table.",
        "Describe the {t} table structure.",
        "What columns does {t} have?",
        "List the columns in {t}.",
        "Describe {t}.",
    ]
    needed = gap_per_tool.get("describe_table", 0)
    for k in range(needed):
        table = _PG_TABLES[k % len(_PG_TABLES)]
        q = describe_queries[k % len(describe_queries)].format(t=table)
        tasks.append(_pg_task(idx, "describe_table", q, {"table_name": table}))
        idx += 1

    # execute_query
    execute_queries = [
        ("How many rows are in bookings.flights?",
         {"sql": "SELECT COUNT(*) AS count FROM bookings.flights"}),
        ("List the distinct flight statuses.",
         {"sql": "SELECT DISTINCT status FROM bookings.flights"}),
        ("What are the fare conditions in ticket_flights?",
         {"sql": "SELECT DISTINCT fare_conditions FROM bookings.ticket_flights"}),
        ("Count cancelled flights.",
         {"sql": "SELECT COUNT(*) AS count FROM bookings.flights WHERE status = 'Cancelled'"}),
        ("List the top 5 airports by code alphabetically.",
         {"sql": "SELECT airport_code FROM bookings.airports ORDER BY airport_code LIMIT 5"}),
        ("Total count of bookings.",
         {"sql": "SELECT COUNT(*) AS count FROM bookings.bookings"}),
        ("How many distinct aircrafts are there?",
         {"sql": "SELECT COUNT(DISTINCT aircraft_code) AS count FROM bookings.aircrafts"}),
        ("List flight numbers from the flights table, first 10.",
         {"sql": "SELECT flight_no FROM bookings.flights LIMIT 10"}),
        ("Average ticket amount across ticket_flights.",
         {"sql": "SELECT AVG(amount) AS avg_amount FROM bookings.ticket_flights"}),
        ("Max ticket amount in ticket_flights.",
         {"sql": "SELECT MAX(amount) AS max_amount FROM bookings.ticket_flights"}),
    ]
    needed = gap_per_tool.get("execute_query", 0)
    for q, p in execute_queries[:needed]:
        tasks.append(_pg_task(idx, "execute_query", q, p))
        idx += 1

    # get_row_count — schema defaults to bookings, omit to match existing convention.
    needed = gap_per_tool.get("get_row_count", 0)
    for k in range(needed):
        table = _PG_TABLES[k % len(_PG_TABLES)]
        q = f"How many rows are in the {table} table?"
        tasks.append(_pg_task(idx, "get_row_count", q, {"table_name": table}))
        idx += 1

    # get_foreign_keys — schema defaults to bookings, omit to match existing convention.
    needed = gap_per_tool.get("get_foreign_keys", 0)
    for k in range(needed):
        table = _PG_TABLES[k % len(_PG_TABLES)]
        q = f"List the foreign key relationships for the {table} table."
        tasks.append(_pg_task(idx, "get_foreign_keys", q, {"table_name": table}))
        idx += 1

    # list_schemas (no params)
    list_schemas_queries = [
        "List all schemas in the database.",
        "What schemas exist in this database?",
        "Show me every available schema.",
        "Enumerate all schemas.",
        "Return all schema names.",
        "List every schema in this Postgres instance.",
        "Which schemas are present in this database?",
        "Show all the schemas.",
        "List schemas available.",
        "Display the database schemas.",
        "What schemas can I query?",
        "Show the schemas in this database.",
        "List database schemas.",
        "Give me the list of schemas.",
        "Enumerate every schema available.",
    ]
    needed = gap_per_tool.get("list_schemas", 0)
    for q in list_schemas_queries[:needed]:
        tasks.append(_pg_task(idx, "list_schemas", q, {}))
        idx += 1

    # find_relationships — schema defaults to bookings, omit to match existing convention.
    needed = gap_per_tool.get("find_relationships", 0)
    for k in range(needed):
        table = _PG_TABLES[k % len(_PG_TABLES)]
        q = f"Find all table relationships involving {table}."
        tasks.append(_pg_task(idx, "find_relationships", q, {"table_name": table}))
        idx += 1

    return tasks


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════

_GENERATORS = {
    "bfcl": gen_bfcl_padding,
    "jefferson": gen_jefferson_padding,
    "postgres": gen_postgres_padding,
    "finance": gen_finance_padding,
}


def main():
    for domain, tools in [("bfcl", CFG["bfcl"]),
                          ("jefferson", CFG["jefferson"]),
                          ("postgres", CFG["postgres"]),
                          ("finance", CFG["finance"])]:
        existing = _load_existing(domain)
        counts = _count_by_tool(existing)
        gaps: dict[str, int] = {}
        for t in tools:
            have = counts.get(t, 0)
            gaps[t] = max(0, TARGET_PER_TOOL - have)

        padding = _GENERATORS[domain](gaps)
        merged = existing + padding

        # Verify post-merge per-tool coverage
        final_counts = _count_by_tool(merged)
        out_path = ROOT / DOMAIN_DIR[domain] / "tasks_incremental.jsonl"
        with open(out_path, "w") as f:
            for t in merged:
                f.write(json.dumps(t) + "\n")

        print(f"\n=== {domain} → {out_path.relative_to(ROOT)} "
              f"({len(existing)} existing + {len(padding)} padding = {len(merged)} total) ===")
        for tool in tools:
            c = final_counts.get(tool, 0)
            flag = " OK" if c >= TARGET_PER_TOOL else f" SHORT (need {TARGET_PER_TOOL - c})"
            print(f"  {tool:<35} {c:>3}{flag}")


if __name__ == "__main__":
    main()
