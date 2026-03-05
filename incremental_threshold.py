"""
Incremental threshold testing for qwen2.5:7b
Tests at: 0, 5, 10, 15, 20, 30, 40 distractors to pinpoint degradation
"""

import subprocess
import json
from pathlib import Path
import time

def run_threshold_test(num_distractors):
    """Run evaluation with specific number of distractors"""
    
    output_file = f"results/qwen2.5_{num_distractors}_distractors.json"
    
    cmd = [
        "python3", "evaluate_bfcl.py",
        "--model", "qwen2.5:7b",
        "--category", "all",
        "--num-distractors", str(num_distractors),
        "--output", output_file
    ]
    
    print(f"\n{'='*70}")
    print(f"Testing with {num_distractors} distractors")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"✓ Completed: {output_file}")
        return output_file
    else:
        print(f"✗ Failed")
        return None

def summarize_incremental_results():
    """Show F1 scores across all distractor levels"""
    
    results_dir = Path("results")
    distractor_levels = [0, 5, 10, 15, 20, 30, 40, 49]
    
    print("\n" + "="*70)
    print("INCREMENTAL THRESHOLD TESTING - qwen2.5:7b")
    print("="*70)
    print(f"\n{'Distractors':<15} {'F1 Score':<12} {'Correct':<10} {'No Call':<10}")
    print("-" * 70)
    
    results_data = []
    for num_dist in distractor_levels:
        filepath = results_dir / f"qwen2.5_{num_dist}_distractors.json"
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
                metrics = data["metrics"]
                f1 = metrics["f1_score"]
                correct = f"{metrics['correct_function']}/{metrics['total_tests']}"
                no_call = metrics['no_tool_call']
                
                print(f"{num_dist:<15} {f1:<12.2f} {correct:<10} {no_call:<10}")
                results_data.append((num_dist, f1))
        else:
            print(f"{num_dist:<15} {'[Pending]':<12} {'-':<10} {'-':<10}")
    
    if len(results_data) > 1:
        print("\n" + "="*70)
        print("DEGRADATION ANALYSIS:")
        print("="*70)
        baseline_f1 = results_data[0][1]
        for i in range(1, len(results_data)):
            num_dist, f1 = results_data[i]
            degradation = baseline_f1 - f1
            print(f"  {num_dist} distractors: {degradation:+.2f}% from baseline")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, help="Run test with N distractors")
    parser.add_argument("--summary", action="store_true", help="Show summary of all results")
    args = parser.parse_args()
    
    if args.test is not None:
        run_threshold_test(args.test)
    elif args.summary:
        summarize_incremental_results()
    else:
        print("Usage:")
        print("  python3 incremental_threshold.py --test 0     # Run baseline")
        print("  python3 incremental_threshold.py --test 10    # Run with 10 distractors")
        print("  python3 incremental_threshold.py --summary    # Show all results")
