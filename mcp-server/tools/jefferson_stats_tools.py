"""
Jefferson Stats Tools - Advanced Statistical Analysis
Source: https://github.com/sharabhshukla/JeffersonStatsMCP

Statistical tools organized from basic to advanced for threshold testing.
"""

import math
from typing import Dict, List

try:
    from scipy import stats as scipy_stats
    _SCIPY_IMPORT_ERROR = None
except Exception as exc:
    scipy_stats = None
    _SCIPY_IMPORT_ERROR = exc


def _require_scipy_stats():
    if scipy_stats is None:
        raise RuntimeError(f"SciPy is unavailable: {_SCIPY_IMPORT_ERROR}")
    return scipy_stats


def register_tools(mcp):
    """Register all Jefferson statistical tools with the MCP server."""
    
    # ============================================================================
    # BASIC STATISTICS
    # ============================================================================

    @mcp.tool()
    def calculate_median(collection: List[float]) -> Dict:
        """Calculate the median of a collection."""
        try:
            if not collection:
                return {"error": "Collection cannot be empty"}
            sorted_data = sorted(collection)
            n = len(sorted_data)
            if n % 2 == 0:
                median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
            else:
                median = sorted_data[n//2]
            return {"result": float(median)}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def calculate_mode(collection: List[float]) -> Dict:
        """Calculate the mode (most frequent value) of a collection."""
        try:
            if not collection:
                return {"error": "Collection cannot be empty"}
            
            frequency = {}
            for value in collection:
                frequency[value] = frequency.get(value, 0) + 1
            
            max_freq = max(frequency.values())
            modes = [k for k, v in frequency.items() if v == max_freq]
            
            if len(modes) == 1:
                return {"result": float(modes[0])}
            else:
                return {"result": [float(m) for m in modes]}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def calculate_range(collection: List[float]) -> Dict:
        """Calculate the range (difference between max and min) of a collection."""
        try:
            if not collection:
                return {"error": "Collection cannot be empty"}
            return {"result": float(max(collection) - min(collection))}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def calculate_variance(collection: List[float]) -> Dict:
        """Calculate the variance of a collection."""
        try:
            if not collection:
                return {"error": "Collection cannot be empty"}
            mean = sum(collection) / len(collection)
            variance = sum((x - mean) ** 2 for x in collection) / len(collection)
            return {"result": float(variance)}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def calculate_quartiles(collection: List[float]) -> Dict:
        """Calculate the quartiles (Q1, Q2, Q3) of a collection."""
        try:
            if not collection:
                return {"error": "Collection cannot be empty"}
            sorted_data = sorted(collection)
            n = len(sorted_data)
            
            q1 = sorted_data[n // 4]
            q2 = sorted_data[n // 2]  # Median
            q3 = sorted_data[3 * n // 4]
            
            return {"q1": float(q1), "q2": float(q2), "q3": float(q3)}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def calculate_iqr(collection: List[float]) -> Dict:
        """Calculate the interquartile range (IQR) of a collection."""
        try:
            if not collection:
                return {"error": "Collection cannot be empty"}
            sorted_data = sorted(collection)
            n = len(sorted_data)
            q1 = sorted_data[n // 4]
            q3 = sorted_data[3 * n // 4]
            return {"result": float(q3 - q1)}
        except Exception as e:
            return {"error": str(e)}

    # ============================================================================
    # ADVANCED STATISTICS
    # ============================================================================

    @mcp.tool()
    def calculate_skewness(collection: List[float]) -> Dict:
        """Calculate the skewness (measure of asymmetry) of a collection."""
        try:
            stats = _require_scipy_stats()
            return {"result": float(stats.skew(collection))}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def calculate_kurtosis(collection: List[float]) -> Dict:
        """Calculate the kurtosis (measure of 'tailedness') of a collection."""
        try:
            stats = _require_scipy_stats()
            return {"result": float(stats.kurtosis(collection))}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def calculate_correlation(collection1: List[float], collection2: List[float]) -> Dict:
        """Calculate the Pearson correlation coefficient between two collections."""
        try:
            stats = _require_scipy_stats()
            if len(collection1) != len(collection2):
                return {"error": "Collections must be of the same length"}
            corr, p_value = stats.pearsonr(collection1, collection2)
            return {"correlation": float(corr), "p_value": float(p_value)}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def calculate_covariance(collection1: List[float], collection2: List[float]) -> Dict:
        """Calculate the covariance between two collections."""
        try:
            if len(collection1) != len(collection2):
                return {"error": "Collections must be of the same length"}
            mean1 = sum(collection1) / len(collection1)
            mean2 = sum(collection2) / len(collection2)
            cov = sum((collection1[i] - mean1) * (collection2[i] - mean2) 
                     for i in range(len(collection1))) / (len(collection1) - 1)
            return {"result": cov}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def calculate_z_scores(collection: List[float]) -> Dict:
        """Calculate z-scores (standard scores) for a collection."""
        try:
            stats = _require_scipy_stats()
            return {"z_scores": [float(z) for z in stats.zscore(collection)]}
        except Exception as e:
            return {"error": str(e)}

    # ============================================================================
    # HYPOTHESIS TESTING
    # ============================================================================

    @mcp.tool()
    def perform_t_test(collection: List[float], popmean: float = 0) -> Dict:
        """Perform a one-sample t-test comparing a collection to a population mean."""
        try:
            stats = _require_scipy_stats()
            t_stat, p_value = stats.ttest_1samp(collection, popmean)
            return {"t_statistic": float(t_stat), "p_value": float(p_value)}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def calculate_confidence_interval(collection: List[float], confidence: float = 0.95) -> Dict:
        """Calculate the confidence interval for the mean of a collection."""
        try:
            stats = _require_scipy_stats()
            mean = sum(collection) / len(collection)
            sem = stats.sem(collection)
            interval = sem * stats.t.ppf((1 + confidence) / 2, len(collection) - 1)
            return {"lower_bound": float(mean - interval), "upper_bound": float(mean + interval)}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def detect_outliers(collection: List[float]) -> Dict:
        """Detect outliers in a collection using the IQR method."""
        try:
            sorted_data = sorted(collection)
            q1_idx = len(sorted_data) // 4
            q3_idx = 3 * len(sorted_data) // 4
            q1 = sorted_data[q1_idx]
            q3 = sorted_data[q3_idx]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = [x for x in collection if x < lower_bound or x > upper_bound]
            return {"outliers": outliers, "count": len(outliers)}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def perform_normality_test(collection: List[float]) -> Dict:
        """Test if a collection comes from a normal distribution using Shapiro-Wilk test."""
        try:
            stats = _require_scipy_stats()
            stat, p_value = stats.shapiro(collection)
            return {
                "statistic": float(stat), 
                "p_value": float(p_value), 
                "interpretation": "normally distributed" if p_value > 0.05 else "not normally distributed"
            }
        except Exception as e:
            return {"error": str(e)}

    # ============================================================================
    # REGRESSION & ANALYSIS
    # ============================================================================

    @mcp.tool()
    def perform_linear_regression(x: List[float], y: List[float]) -> Dict:
        """Perform simple linear regression between two variables."""
        try:
            from scipy import stats
            if len(x) != len(y):
                return {"error": "Collections must be of the same length"}
            result = stats.linregress(x, y)
            return {
                "slope": float(result.slope),
                "intercept": float(result.intercept),
                "r_value": float(result.rvalue),
                "p_value": float(result.pvalue)
            }
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def calculate_moving_average(collection: List[float], window_size: int) -> Dict:
        """Calculate the moving average of a time series."""
        try:
            if window_size <= 0:
                return {"error": "Window size must be positive"}
            if window_size > len(collection):
                return {"error": "Window size cannot be larger than collection size"}
            
            result = []
            for i in range(len(collection) - window_size + 1):
                window = collection[i:i + window_size]
                result.append(sum(window) / window_size)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def generate_descriptive_statistics(collection: List[float]) -> Dict:
        """Generate a comprehensive summary of descriptive statistics for a collection."""
        try:
            from scipy import stats
            sorted_data = sorted(collection)
            n = len(collection)
            
            mean = sum(collection) / n
            median = sorted_data[n // 2] if n % 2 != 0 else (sorted_data[n//2-1] + sorted_data[n//2]) / 2
            variance = sum((x - mean) ** 2 for x in collection) / n
            std_dev = variance ** 0.5
            
            q1 = sorted_data[n // 4]
            q3 = sorted_data[3 * n // 4]
            iqr = q3 - q1
            
            return {
                "count": n,
                "mean": float(mean),
                "median": float(median),
                "min": float(min(collection)),
                "max": float(max(collection)),
                "range": float(max(collection) - min(collection)),
                "variance": float(variance),
                "std_dev": float(std_dev),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "skewness": float(stats.skew(collection)),
                "kurtosis": float(stats.kurtosis(collection))
            }
        except Exception as e:
            return {"error": str(e)}
    
    print("Registered 18 Jefferson Stats Tools (6 basic + 5 advanced + 4 hypothesis testing + 3 analysis)")
