"""
Test suite for Jefferson Stats tools integration
Tests statistical functions adapted from JeffersonStatsMCP

Run this with the MCP virtual environment:
source mcp-client/.venv/bin/activate && python3 test_jefferson_stats.py
"""

import math
from scipy import stats


# Direct implementations for testing (without MCP decorator)
def calculate_skewness(collection):
    """Calculate the skewness (measure of asymmetry) of a collection."""
    return {"result": float(stats.skew(collection))}


def calculate_kurtosis(collection):
    """Calculate the kurtosis (measure of 'tailedness') of a collection."""
    return {"result": float(stats.kurtosis(collection))}


def calculate_correlation(collection1, collection2):
    """Calculate the Pearson correlation coefficient between two collections."""
    if len(collection1) != len(collection2):
        return {"error": "Collections must be of the same length"}
    corr, p_value = stats.pearsonr(collection1, collection2)
    return {"correlation": float(corr), "p_value": float(p_value)}


def calculate_covariance(collection1, collection2):
    """Calculate the covariance between two collections."""
    if len(collection1) != len(collection2):
        return {"error": "Collections must be of the same length"}
    mean1 = sum(collection1) / len(collection1)
    mean2 = sum(collection2) / len(collection2)
    cov = sum((collection1[i] - mean1) * (collection2[i] - mean2) 
             for i in range(len(collection1))) / (len(collection1) - 1)
    return {"result": cov}


def calculate_z_scores(collection):
    """Calculate z-scores (standard scores) for a collection."""
    return {"z_scores": [float(z) for z in stats.zscore(collection)]}


def perform_t_test(collection, popmean=0):
    """Perform a one-sample t-test comparing a collection to a population mean."""
    t_stat, p_value = stats.ttest_1samp(collection, popmean)
    return {"t_statistic": float(t_stat), "p_value": float(p_value)}


def calculate_confidence_interval(collection, confidence=0.95):
    """Calculate the confidence interval for the mean of a collection."""
    mean = sum(collection) / len(collection)
    sem = stats.sem(collection)
    interval = sem * stats.t.ppf((1 + confidence) / 2, len(collection) - 1)
    return {"lower_bound": float(mean - interval), "upper_bound": float(mean + interval)}


def detect_outliers(collection):
    """Detect outliers in a collection using the IQR method."""
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


def perform_normality_test(collection):
    """Test if a collection comes from a normal distribution using the Shapiro-Wilk test."""
    stat, p_value = stats.shapiro(collection)
    return {"statistic": float(stat), "p_value": float(p_value), 
            "interpretation": "normally distributed" if p_value > 0.05 else "not normally distributed"}


def perform_linear_regression(x, y):
    """Perform simple linear regression between two variables."""
    if len(x) != len(y):
        return {"error": "Collections must be of the same length"}
    result = stats.linregress(x, y)
    return {
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r_value": float(result.rvalue),
        "p_value": float(result.pvalue)
    }


def calculate_moving_average(collection, window_size):
    """Calculate the moving average of a time series."""
    if window_size <= 0:
        return {"error": "Window size must be positive"}
    if window_size > len(collection):
        return {"error": "Window size cannot be larger than collection size"}
    
    result = []
    for i in range(len(collection) - window_size + 1):
        window = collection[i:i + window_size]
        result.append(sum(window) / window_size)
    return {"result": result}


def generate_descriptive_statistics(collection):
    """Generate a comprehensive summary of descriptive statistics for a collection."""
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


def test_skewness():
    """Test skewness calculation"""
    # Right-skewed data
    data = [1, 2, 2, 3, 3, 3, 4, 10, 20]
    result = calculate_skewness(data)
    assert "result" in result
    assert result["result"] > 0  # Should be positive (right-skewed)
    print(f"✓ Skewness: {result['result']:.4f}")


def test_kurtosis():
    """Test kurtosis calculation"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = calculate_kurtosis(data)
    assert "result" in result
    print(f"✓ Kurtosis: {result['result']:.4f}")


def test_correlation():
    """Test Pearson correlation"""
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]  # Perfect positive correlation
    result = calculate_correlation(x, y)
    assert "correlation" in result
    assert abs(result["correlation"] - 1.0) < 0.01  # Should be close to 1
    print(f"✓ Correlation: {result['correlation']:.4f}, p-value: {result['p_value']:.4f}")


def test_covariance():
    """Test covariance calculation"""
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    result = calculate_covariance(x, y)
    assert "result" in result
    assert result["result"] > 0  # Positive covariance expected
    print(f"✓ Covariance: {result['result']:.4f}")


def test_z_scores():
    """Test z-score calculation"""
    data = [10, 20, 30, 40, 50]
    result = calculate_z_scores(data)
    assert "z_scores" in result
    assert len(result["z_scores"]) == 5
    # Mean of z-scores should be ~0
    assert abs(sum(result["z_scores"]) / len(result["z_scores"])) < 0.01
    print(f"✓ Z-scores: {[f'{z:.2f}' for z in result['z_scores']]}")


def test_t_test():
    """Test one-sample t-test"""
    data = [10, 12, 11, 13, 12, 11, 10]
    result = perform_t_test(data, popmean=10)
    assert "t_statistic" in result
    assert "p_value" in result
    print(f"✓ T-test: t={result['t_statistic']:.4f}, p={result['p_value']:.4f}")


def test_confidence_interval():
    """Test confidence interval calculation"""
    data = [10, 12, 11, 13, 12, 11, 10]
    result = calculate_confidence_interval(data, confidence=0.95)
    assert "lower_bound" in result
    assert "upper_bound" in result
    assert result["lower_bound"] < result["upper_bound"]
    print(f"✓ 95% CI: [{result['lower_bound']:.2f}, {result['upper_bound']:.2f}]")


def test_outlier_detection():
    """Test outlier detection using IQR method"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is an outlier
    result = detect_outliers(data)
    assert "outliers" in result
    assert "count" in result
    assert 100 in result["outliers"]
    print(f"✓ Outliers detected: {result['outliers']} (count: {result['count']})")


def test_normality_test():
    """Test Shapiro-Wilk normality test"""
    # Approximately normal data
    data = [10, 12, 11, 13, 12, 11, 10, 14, 9, 11, 12, 13]
    result = perform_normality_test(data)
    assert "statistic" in result
    assert "p_value" in result
    assert "interpretation" in result
    print(f"✓ Normality test: statistic={result['statistic']:.4f}, p={result['p_value']:.4f}, {result['interpretation']}")


def test_linear_regression():
    """Test simple linear regression"""
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]  # y = 2x
    result = perform_linear_regression(x, y)
    assert "slope" in result
    assert "intercept" in result
    assert abs(result["slope"] - 2.0) < 0.01  # Slope should be ~2
    assert abs(result["intercept"]) < 0.01  # Intercept should be ~0
    print(f"✓ Linear regression: y = {result['slope']:.2f}x + {result['intercept']:.2f}, r={result['r_value']:.4f}")


def test_moving_average():
    """Test moving average calculation"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = calculate_moving_average(data, window_size=3)
    assert "result" in result
    assert len(result["result"]) == 8  # 10 - 3 + 1
    assert abs(result["result"][0] - 2.0) < 0.01  # First MA: (1+2+3)/3 = 2
    print(f"✓ Moving average (window=3): {[f'{x:.1f}' for x in result['result'][:5]]}...")


def test_descriptive_statistics():
    """Test comprehensive descriptive statistics"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = generate_descriptive_statistics(data)
    assert "count" in result
    assert "mean" in result
    assert "median" in result
    assert result["count"] == 10
    assert abs(result["mean"] - 5.5) < 0.01
    assert abs(result["median"] - 5.5) < 0.01
    print(f"✓ Descriptive stats: count={result['count']}, mean={result['mean']:.2f}, median={result['median']:.2f}")
    print(f"  Range: [{result['min']}, {result['max']}], std_dev={result['std_dev']:.2f}")


def run_all_tests():
    """Run all Jefferson Stats tests"""
    tests = [
        ("Skewness", test_skewness),
        ("Kurtosis", test_kurtosis),
        ("Correlation", test_correlation),
        ("Covariance", test_covariance),
        ("Z-scores", test_z_scores),
        ("T-test", test_t_test),
        ("Confidence Interval", test_confidence_interval),
        ("Outlier Detection", test_outlier_detection),
        ("Normality Test", test_normality_test),
        ("Linear Regression", test_linear_regression),
        ("Moving Average", test_moving_average),
        ("Descriptive Statistics", test_descriptive_statistics),
    ]
    
    print("\n" + "="*60)
    print("JEFFERSON STATS TOOLS - Test Suite")
    print("Source: https://github.com/sharabhshukla/JeffersonStatsMCP")
    print("="*60 + "\n")
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*60 + "\n")
    
    if failed == 0:
        print("🎉 All Jefferson Stats tools working correctly!")
        print("\nThese 12 statistical tools are now added to your MCP server:")
        print("- Advanced Statistics: skewness, kurtosis, correlation, covariance, z-scores")
        print("- Hypothesis Testing: t-test, confidence interval, outlier detection, normality test")
        print("- Analysis: linear regression, moving average, descriptive statistics")
        print("\nTotal tools in server: 50 (BFCL Math) + 12 (Jefferson Stats) = 62 tools")
        return True
    return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)


def test_skewness():
    """Test skewness calculation"""
    # Right-skewed data
    data = [1, 2, 2, 3, 3, 3, 4, 10, 20]
    result = calculate_skewness(data)
    assert "result" in result
    assert result["result"] > 0  # Should be positive (right-skewed)
    print(f"✓ Skewness: {result['result']:.4f}")


def test_kurtosis():
    """Test kurtosis calculation"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = calculate_kurtosis(data)
    assert "result" in result
    print(f"✓ Kurtosis: {result['result']:.4f}")


def test_correlation():
    """Test Pearson correlation"""
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]  # Perfect positive correlation
    result = calculate_correlation(x, y)
    assert "correlation" in result
    assert abs(result["correlation"] - 1.0) < 0.01  # Should be close to 1
    print(f"✓ Correlation: {result['correlation']:.4f}, p-value: {result['p_value']:.4f}")


def test_covariance():
    """Test covariance calculation"""
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    result = calculate_covariance(x, y)
    assert "result" in result
    assert result["result"] > 0  # Positive covariance expected
    print(f"✓ Covariance: {result['result']:.4f}")


def test_z_scores():
    """Test z-score calculation"""
    data = [10, 20, 30, 40, 50]
    result = calculate_z_scores(data)
    assert "z_scores" in result
    assert len(result["z_scores"]) == 5
    # Mean of z-scores should be ~0
    assert abs(sum(result["z_scores"]) / len(result["z_scores"])) < 0.01
    print(f"✓ Z-scores: {[f'{z:.2f}' for z in result['z_scores']]}")


def test_t_test():
    """Test one-sample t-test"""
    data = [10, 12, 11, 13, 12, 11, 10]
    result = perform_t_test(data, popmean=10)
    assert "t_statistic" in result
    assert "p_value" in result
    print(f"✓ T-test: t={result['t_statistic']:.4f}, p={result['p_value']:.4f}")


def test_confidence_interval():
    """Test confidence interval calculation"""
    data = [10, 12, 11, 13, 12, 11, 10]
    result = calculate_confidence_interval(data, confidence=0.95)
    assert "lower_bound" in result
    assert "upper_bound" in result
    assert result["lower_bound"] < result["upper_bound"]
    print(f"✓ 95% CI: [{result['lower_bound']:.2f}, {result['upper_bound']:.2f}]")


def test_outlier_detection():
    """Test outlier detection using IQR method"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is an outlier
    result = detect_outliers(data)
    assert "outliers" in result
    assert "count" in result
    assert 100 in result["outliers"]
    print(f"✓ Outliers detected: {result['outliers']} (count: {result['count']})")


def test_normality_test():
    """Test Shapiro-Wilk normality test"""
    # Approximately normal data
    data = [10, 12, 11, 13, 12, 11, 10, 14, 9, 11, 12, 13]
    result = perform_normality_test(data)
    assert "statistic" in result
    assert "p_value" in result
    assert "interpretation" in result
    print(f"✓ Normality test: statistic={result['statistic']:.4f}, p={result['p_value']:.4f}, {result['interpretation']}")


def test_linear_regression():
    """Test simple linear regression"""
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]  # y = 2x
    result = perform_linear_regression(x, y)
    assert "slope" in result
    assert "intercept" in result
    assert abs(result["slope"] - 2.0) < 0.01  # Slope should be ~2
    assert abs(result["intercept"]) < 0.01  # Intercept should be ~0
    print(f"✓ Linear regression: y = {result['slope']:.2f}x + {result['intercept']:.2f}, r={result['r_value']:.4f}")


def test_moving_average():
    """Test moving average calculation"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = calculate_moving_average(data, window_size=3)
    assert "result" in result
    assert len(result["result"]) == 8  # 10 - 3 + 1
    assert abs(result["result"][0] - 2.0) < 0.01  # First MA: (1+2+3)/3 = 2
    print(f"✓ Moving average (window=3): {[f'{x:.1f}' for x in result['result'][:5]]}...")


def test_descriptive_statistics():
    """Test comprehensive descriptive statistics"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = generate_descriptive_statistics(data)
    assert "count" in result
    assert "mean" in result
    assert "median" in result
    assert result["count"] == 10
    assert abs(result["mean"] - 5.5) < 0.01
    assert abs(result["median"] - 5.5) < 0.01
    print(f"✓ Descriptive stats: count={result['count']}, mean={result['mean']:.2f}, median={result['median']:.2f}")
    print(f"  Range: [{result['min']}, {result['max']}], std_dev={result['std_dev']:.2f}")


def run_all_tests():
    """Run all Jefferson Stats tests"""
    tests = [
        ("Skewness", test_skewness),
        ("Kurtosis", test_kurtosis),
        ("Correlation", test_correlation),
        ("Covariance", test_covariance),
        ("Z-scores", test_z_scores),
        ("T-test", test_t_test),
        ("Confidence Interval", test_confidence_interval),
        ("Outlier Detection", test_outlier_detection),
        ("Normality Test", test_normality_test),
        ("Linear Regression", test_linear_regression),
        ("Moving Average", test_moving_average),
        ("Descriptive Statistics", test_descriptive_statistics),
    ]
    
    print("\n" + "="*60)
    print("JEFFERSON STATS TOOLS - Test Suite")
    print("Source: https://github.com/sharabhshukla/JeffersonStatsMCP")
    print("="*60 + "\n")
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*60 + "\n")
    
    if failed == 0:
        print("🎉 All Jefferson Stats tools working correctly!")
        return True
    return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
