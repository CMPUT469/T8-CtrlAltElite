"""
Threshold Testing for qwen2.5 with Real Gorilla BFCL Tools

Tests model limits using actual Berkeley Function Calling Leaderboard tools:
1. Number of tools (3, 5, 10, 14 tools)
2. Parameter complexity (simple, medium, list operations)
3. Multi-step reasoning (1, 2, 3, 4+ tool chains)
4. Error handling (division by zero, negative sqrt, empty lists)

Week 2 Deliverable: Find qwen2.5's threshold limits with real Gorilla tools
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================================
# TEST CASES - Real Gorilla BFCL Tools
# ============================================================================

# Simple: 1-2 parameters, basic arithmetic
SIMPLE_TESTS = [
    {
        "id": "simple_1",
        "query": "Add 15 and 27",
        "expected_tool": "add",
        "expected_params": {"a": 15, "b": 27},
        "expected_result": 42,
        "complexity": "simple"
    },
    {
        "id": "simple_2",
        "query": "What is 100 minus 37?",
        "expected_tool": "subtract",
        "expected_params": {"a": 100, "b": 37},
        "expected_result": 63,
        "complexity": "simple"
    },
    {
        "id": "simple_3",
        "query": "Multiply 12 by 8",
        "expected_tool": "multiply",
        "expected_params": {"a": 12, "b": 8},
        "expected_result": 96,
        "complexity": "simple"
    },
    {
        "id": "simple_4",
        "query": "Divide 144 by 12",
        "expected_tool": "divide",
        "expected_params": {"a": 144, "b": 12},
        "expected_result": 12,
        "complexity": "simple"
    },
    {
        "id": "simple_5",
        "query": "What is the absolute value of -42?",
        "expected_tool": "absolute_value",
        "expected_params": {"number": -42},
        "expected_result": 42,
        "complexity": "simple"
    }
]

# Medium: 2-3 parameters, some precision handling
MEDIUM_TESTS = [
    {
        "id": "medium_1",
        "query": "Calculate 2 raised to the power of 8",
        "expected_tool": "power",
        "expected_params": {"base": 2, "exponent": 8},
        "expected_result": 256,
        "complexity": "medium"
    },
    {
        "id": "medium_2",
        "query": "Find the square root of 144 with 2 decimal places",
        "expected_tool": "square_root",
        "expected_params": {"number": 144, "precision": 2},
        "expected_result": 12.0,
        "complexity": "medium"
    },
    {
        "id": "medium_3",
        "query": "Round 3.14159 to 2 decimal places",
        "expected_tool": "round_number",
        "expected_params": {"number": 3.14159, "decimal_places": 2},
        "expected_result": 3.14,
        "complexity": "medium"
    },
    {
        "id": "medium_4",
        "query": "What percentage is 25 of 200?",
        "expected_tool": "percentage",
        "expected_params": {"part": 25, "whole": 200},
        "expected_result": 12.5,
        "complexity": "medium"
    }
]

# Complex: List operations
COMPLEX_TESTS = [
    {
        "id": "complex_1",
        "query": "Find the sum of these numbers: 10, 20, 30, 40, 50",
        "expected_tool": "sum_values",
        "expected_params": {"numbers": [10, 20, 30, 40, 50]},
        "expected_result": 150,
        "complexity": "complex"
    },
    {
        "id": "complex_2",
        "query": "Calculate the mean of the values: 5, 10, 15, 20, 25",
        "expected_tool": "mean",
        "expected_params": {"numbers": [5, 10, 15, 20, 25]},
        "expected_result": 15,
        "complexity": "complex"
    },
    {
        "id": "complex_3",
        "query": "What is the minimum value in this list: 23, 8, 45, 12, 67",
        "expected_tool": "min_value",
        "expected_params": {"numbers": [23, 8, 45, 12, 67]},
        "expected_result": 8,
        "complexity": "complex"
    },
    {
        "id": "complex_4",
        "query": "Find the maximum in these numbers: 15, 42, 7, 89, 33",
        "expected_tool": "max_value",
        "expected_params": {"numbers": [15, 42, 7, 89, 33]},
        "expected_result": 89,
        "complexity": "complex"
    },
    {
        "id": "complex_5",
        "query": "Calculate standard deviation of: 2, 4, 6, 8, 10",
        "expected_tool": "standard_deviation",
        "expected_params": {"numbers": [2, 4, 6, 8, 10]},
        "complexity": "complex"
    }
]

# Multi-step: Requires chaining multiple tools
MULTISTEP_TESTS = [
    {
        "id": "multistep_2",
        "query": "Add 25 and 15, then multiply the result by 3",
        "expected_tools": ["add", "multiply"],
        "steps": 2,
        "expected_result": 120,  # (25+15)*3 = 40*3 = 120
        "complexity": "multi_2"
    },
    {
        "id": "multistep_3",
        "query": "Calculate the mean of 10, 20, 30, then take its square root with 2 decimal places",
        "expected_tools": ["mean", "square_root"],
        "steps": 2,
        "expected_result": 4.47,  # mean=20, sqrt(20)≈4.47
        "complexity": "multi_3"
    },
    {
        "id": "multistep_4",
        "query": "Find the sum of 5, 10, 15, 20, divide by 2, then round to 1 decimal place",
        "expected_tools": ["sum_values", "divide", "round_number"],
        "steps": 3,
        "expected_result": 25.0,  # sum=50, 50/2=25, round(25,1)=25.0
        "complexity": "multi_4"
    }
]

# Error handling tests
ERROR_TESTS = [
    {
        "id": "error_1",
        "query": "Divide 100 by 0",
        "expected_tool": "divide",
        "expected_params": {"a": 100, "b": 0},
        "should_error": True,
        "complexity": "error"
    },
    {
        "id": "error_2",
        "query": "Calculate square root of -16",
        "expected_tool": "square_root",
        "expected_params": {"number": -16, "precision": 2},
        "should_error": True,
        "complexity": "error"
    },
    {
        "id": "error_3",
        "query": "Find the mean of an empty list",
        "expected_tool": "mean",
        "expected_params": {"numbers": []},
        "should_error": True,
        "complexity": "error"
    }
]

# ============================================================================
# THRESHOLD TESTS - Progressive Tool Count
# ============================================================================

def get_tool_subsets():
    """Return Gorilla BFCL tool subsets for threshold testing"""
    all_tools = [
        "add", "subtract", "multiply", "divide",
        "power", "square_root", "absolute_value", 
        "round_number", "percentage", "sum_values",
        "mean", "min_value", "max_value", "standard_deviation"
    ]
    
    return {
        "3_tools": all_tools[:3],    # add, subtract, multiply
        "5_tools": all_tools[:5],    # + divide, power
        "7_tools": all_tools[:7],    # + square_root, absolute_value
        "10_tools": all_tools[:10],  # + round_number, percentage, sum_values
        "14_tools": all_tools        # All Gorilla BFCL tools
    }

# ============================================================================
# TEST RUNNER
# ============================================================================

class ThresholdTester:
    def __init__(self, model: str = "qwen2.5"):
        self.model = model
        self.results = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }
    
    def run_test_suite(self, test_cases: List[Dict], suite_name: str):
        """Run a suite of test cases"""
        print(f"\n{'='*60}")
        print(f"Running {suite_name} Tests (Real Gorilla BFCL Tools)")
        print(f"{'='*60}\n")
        
        suite_results = {
            "total": len(test_cases),
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for i, test in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] Testing: {test['query'][:60]}...")
            
            result = self.run_single_test(test)
            suite_results["details"].append(result)
            
            if result["success"]:
                suite_results["passed"] += 1
                print(f"  PASS")
            else:
                suite_results["failed"] += 1
                print(f"  FAIL: {result.get('reason', 'Unknown')}")
        
        suite_results["accuracy"] = suite_results["passed"] / suite_results["total"]
        self.results["tests"][suite_name] = suite_results
        
        print(f"\n{suite_name} Results: {suite_results['passed']}/{suite_results['total']} passed")
        print(f"Accuracy: {suite_results['accuracy']:.2%}\n")
        
        return suite_results
    
    def run_single_test(self, test: Dict) -> Dict:
        """
        Run a single test case.
        In a real implementation, this would:
        1. Send query to MCP client
        2. Capture tool calls
        3. Compare with expected
        
        For now, returns mock results for structure
        """
        # TODO: Implement actual MCP client integration
        # This is a placeholder showing the expected structure
        
        return {
            "test_id": test["id"],
            "query": test["query"],
            "expected_tool": test.get("expected_tool"),
            "actual_tool": "placeholder",  # Would come from MCP client
            "success": False,  # Would be calculated from comparison
            "reason": "Not implemented - placeholder",
            "execution_time_ms": 0
        }
    
    def run_threshold_analysis(self):
        """Run progressive threshold tests"""
        print("\n" + "="*60)
        print("THRESHOLD ANALYSIS - Progressive Tool Count")
        print("="*60)
        
        tool_subsets = get_tool_subsets()
        threshold_results = {}
        
        # Test with increasing tool counts
        for subset_name, tools in tool_subsets.items():
            print(f"\nTesting with {len(tools)} tools ({subset_name})")
            print(f"Tools: {', '.join(tools[:5])}{'...' if len(tools) > 5 else ''}")
            
            # Run simple tests with this tool subset
            # In real implementation, would configure MCP server with only these tools
            result = self.run_test_suite(SIMPLE_TESTS, f"threshold_{subset_name}")
            
            threshold_results[subset_name] = {
                "tool_count": len(tools),
                "accuracy": result["accuracy"],
                "passed": result["passed"],
                "total": result["total"]
            }
        
        self.results["threshold_analysis"] = threshold_results
        
        # Find degradation point
        print("\n" + "="*60)
        print("THRESHOLD ANALYSIS SUMMARY")
        print("="*60)
        for subset, data in threshold_results.items():
            print(f"{subset:15} ({data['tool_count']:2} tools): {data['accuracy']:.2%} accuracy")
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*70)
        print(f"  THRESHOLD TESTING FOR {self.model.upper()}")
        print("="*70)
        
        # 1. Complexity tests
        self.run_test_suite(SIMPLE_TESTS, "simple")
        self.run_test_suite(MEDIUM_TESTS, "medium")
        self.run_test_suite(COMPLEX_TESTS, "complex")
        self.run_test_suite(MULTISTEP_TESTS, "multistep")
        
        # 2. Threshold analysis
        self.run_threshold_analysis()
        
        # 3. Generate summary
        self.generate_summary()
        
        # 4. Save results
        self.save_results()
    
    def generate_summary(self):
        """Generate overall summary"""
        summary = {
            "model": self.model,
            "timestamp": self.results["timestamp"],
            "overall_accuracy": 0.0,
            "by_complexity": {},
            "threshold_limit": "Unknown",
            "recommendations": []
        }
        
        # Calculate overall accuracy
        total_tests = 0
        total_passed = 0
        
        for suite_name, suite_data in self.results["tests"].items():
            if suite_name.startswith("threshold_"):
                continue
            total_tests += suite_data["total"]
            total_passed += suite_data["passed"]
            summary["by_complexity"][suite_name] = suite_data["accuracy"]
        
        if total_tests > 0:
            summary["overall_accuracy"] = total_passed / total_tests
        
        # Recommendations based on results
        if summary["overall_accuracy"] > 0.8:
            summary["recommendations"].append("Model performs well on basic tool selection")
        else:
            summary["recommendations"].append("Model needs improvement on tool selection")
        
        self.results["summary"] = summary
        
        # Print summary
        print("\n" + "="*70)
        print("  FINAL SUMMARY")
        print("="*70)
        print(f"\nModel: {self.model}")
        print(f"Overall Accuracy: {summary['overall_accuracy']:.2%}")
        print("\nBy Complexity:")
        for complexity, accuracy in summary["by_complexity"].items():
            print(f"  {complexity:15}: {accuracy:.2%}")
        print("\nRecommendations:")
        for rec in summary["recommendations"]:
            print(f"  • {rec}")
    
    def save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"threshold_results_{self.model}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, indent=2, fp=f)
        
        print(f"\nResults saved to: {filename}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test qwen2.5 tool usage thresholds")
    parser.add_argument("--model", default="qwen2.5", help="Model name to test")
    parser.add_argument("--suite", choices=["simple", "medium", "complex", "multistep", "threshold", "all"],
                       default="all", help="Test suite to run")
    
    args = parser.parse_args()
    
    tester = ThresholdTester(model=args.model)
    
    if args.suite == "all":
        tester.run_all_tests()
    elif args.suite == "simple":
        tester.run_test_suite(SIMPLE_TESTS, "simple")
    elif args.suite == "medium":
        tester.run_test_suite(MEDIUM_TESTS, "medium")
    elif args.suite == "complex":
        tester.run_test_suite(COMPLEX_TESTS, "complex")
    elif args.suite == "multistep":
        tester.run_test_suite(MULTISTEP_TESTS, "multistep")
    elif args.suite == "threshold":
        tester.run_threshold_analysis()
    
    print("\n" + "="*70)
    print("  Week 2 Threshold Testing Complete!")
    print("="*70)
    print("\nNext Steps:")
    print("1. Review threshold_results_*.json for detailed metrics")
    print("2. Identify where accuracy drops")
    print("3. Test with other models in Week 3")
    print("4. Run full Gorilla evaluation in Week 4")


if __name__ == "__main__":
    main()
