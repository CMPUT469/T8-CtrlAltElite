# Jefferson Stats Integration

**Date:** March 11, 2026  
**Source:** [JeffersonStatsMCP](https://github.com/sharabhshukla/JeffersonStatsMCP)

## Integrated Tools

### Basic Statistics (6 tools)
- `calculate_median` - Median value
- `calculate_mode` - Most frequent value
- `calculate_range` - Max - min
- `calculate_variance` - Variance
- `calculate_quartiles` - Q1, Q2, Q3
- `calculate_iqr` - Interquartile range

### Advanced Statistics (5 tools)
- `calculate_skewness` - Distribution asymmetry
- `calculate_kurtosis` - Distribution tailedness
- `calculate_correlation` - Pearson correlation
- `calculate_covariance` - Covariance between datasets
- `calculate_z_scores` - Standardized scores

### Hypothesis Testing (4 tools)
- `perform_t_test` - One-sample t-test
- `calculate_confidence_interval` - Confidence intervals
- `detect_outliers` - IQR-based detection
- `perform_normality_test` - Shapiro-Wilk test

### Analysis (3 tools)
- `perform_linear_regression` - Linear regression
- `calculate_moving_average` - Moving averages
- `generate_descriptive_statistics` - Summary statistics

**Total:** 18 statistical tools + 14 BFCL math tools = 32 tools

## Integration Method

Copied function definitions from JeffersonStatsMCP `src/basics.py`:
- Applied `@mcp.tool()` decorators
- Converted to Dict return types
- No server connections required

## Test Coverage

`test_jefferson_stats.py` validates 12 functions with known mathematical properties:
- Perfect correlation = 1.0
- Outlier detection accuracy
- Z-scores mean approximately 0
- Confidence interval bounds

## Setup

```bash
source mcp-client/.venv/bin/activate
pip install scipy numpy
python3 test_jefferson_stats.py
```

## Code Organization

Modularized structure for easier dataset swapping:
- `mcp-server/tools/bfcl_math_tools.py` - 14 math tools
- `mcp-server/tools/jefferson_stats_tools.py` - 18 statistical tools
- `mcp-server/main_new.py` - Entry point with modular imports

## Files Modified

1. `mcp-server/tools/jefferson_stats_tools.py` - Statistical tool implementations
2. `mcp-server/tools/bfcl_math_tools.py` - Math tool implementations
3. `test_jefferson_stats.py` - Test suite
4. `requirements-eval.txt` - Added scipy, numpy

