import os
import json
import csv
import re
from glob import glob
from typing import List, Tuple, Optional, Set
from dotenv import load_dotenv

load_dotenv()

# Set model name from environment variables
MODEL_NAME = os.getenv("MODEL_NAME")

# Set CSV filename and data directory
CSV_FILENAME = f"{MODEL_NAME}_results.csv"
DATA_DIR = f"data/{MODEL_NAME}"
FAILED_PAIRS_FILES = ["gemma3_1b_failed_pairs.json", "qwen3_0.6b_failed_pairs.json"]


def calculate_mae(ground_truth: List[int], predictions: List[Optional[int]]) -> float:
    """Calculate MAE, filtering out None predictions (failed LLM calls)."""
    valid_pairs = [
        (gt, pred) for gt, pred in zip(ground_truth, predictions) if pred is not None
    ]
    if not valid_pairs:
        return float("nan")
    return sum(abs(gt - pred) for gt, pred in valid_pairs) / len(valid_pairs)


def calculate_jer(ground_truth: List[int], predictions: List[Optional[int]]) -> float:
    """Calculate JER (Judgment Error Rate) - percentage of predictions differing from ground truth."""
    valid_pairs = [
        (gt, pred) for gt, pred in zip(ground_truth, predictions) if pred is not None
    ]
    if not valid_pairs:
        return float("nan")
    errors = sum(1 for gt, pred in valid_pairs if gt != pred)
    return (errors / len(valid_pairs)) * 100


def process_file(
    filepath: str, excluded_pairs: Set[Tuple[str, str]] = None
) -> Tuple[float, float, float, float, float, float, int, int, int, int, int]:
    """Process a single JSON file and return MAE and JER metrics."""
    if excluded_pairs is None:
        excluded_pairs = set()

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both old format (list) and new format (dict with results)
    if isinstance(data, dict) and "results" in data:
        results = data["results"]
    else:
        results = data

    # Filter out excluded pairs first
    filtered_results = []
    excluded_count = 0
    for item in results:
        qid = item.get("qid")
        docid = item.get("docid")
        if (qid, docid) not in excluded_pairs:
            filtered_results.append(item)
        else:
            excluded_count += 1

    # Extract all results, including those with None llm_scores
    all_results = filtered_results
    ground_truth = [item["ground_truth_score"] for item in all_results]
    llm_scores = [item.get("llm_score") for item in all_results]  # None if missing

    # Filter out None llm_scores for valid experiments only
    valid_results = [item for item in all_results if item.get("llm_score") is not None]

    if not valid_results:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0,
            0,
            0,
            len(all_results),
            excluded_count,
        )

    # Overall MAE and JER
    overall_mae = calculate_mae(ground_truth, llm_scores)
    overall_jer = calculate_jer(ground_truth, llm_scores)

    # MAE and JER for ground truth = 0 (only valid experiments)
    gt_zero_pairs = [
        (gt, pred)
        for gt, pred in zip(ground_truth, llm_scores)
        if gt == 0 and pred is not None
    ]
    gt_zero_mae = (
        sum(abs(gt - pred) for gt, pred in gt_zero_pairs) / len(gt_zero_pairs)
        if gt_zero_pairs
        else float("nan")
    )
    gt_zero_jer = (
        (sum(1 for gt, pred in gt_zero_pairs if gt != pred) / len(gt_zero_pairs) * 100)
        if gt_zero_pairs
        else float("nan")
    )

    # MAE and JER for positive non-zero ground truth (only valid experiments)
    gt_positive_pairs = [
        (gt, pred)
        for gt, pred in zip(ground_truth, llm_scores)
        if gt > 0 and pred is not None
    ]
    gt_positive_mae = (
        sum(abs(gt - pred) for gt, pred in gt_positive_pairs) / len(gt_positive_pairs)
        if gt_positive_pairs
        else float("nan")
    )
    gt_positive_jer = (
        (
            sum(1 for gt, pred in gt_positive_pairs if gt != pred)
            / len(gt_positive_pairs)
            * 100
        )
        if gt_positive_pairs
        else float("nan")
    )

    return (
        overall_mae,
        gt_zero_mae,
        gt_positive_mae,
        overall_jer,
        gt_zero_jer,
        gt_positive_jer,
        len(valid_results),
        len(gt_zero_pairs),
        len(gt_positive_pairs),
        len(all_results),
        excluded_count,
    )


def parse_filename(filename: str) -> Tuple[str, str, str]:
    """Parse filename to extract prompt_type, attack_type, mitigation_type."""
    # Expected format: results_PROMPT_ATTACK_MITIGATION.json
    match = re.match(r"results_([^_]+)_([^_]+)_([^.]+)\.json", filename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return "unknown", "unknown", "unknown"


def load_failed_pairs(failed_pairs_files: List[str]) -> Set[Tuple[str, str]]:
    """Load the set of failed (qid, docid) pairs to exclude from analysis (union across all files)."""
    all_failed_pairs = set()

    for failed_pairs_file in failed_pairs_files:
        if not os.path.exists(failed_pairs_file):
            print(f"Warning: Failed pairs file not found: {failed_pairs_file}")
            continue

        try:
            with open(failed_pairs_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            file_failed_pairs = set()
            for pair in data.get("failed_pairs", []):
                qid = pair.get("qid")
                docid = pair.get("docid")
                if qid and docid:
                    file_failed_pairs.add((qid, docid))

            print(f"  {failed_pairs_file}: {len(file_failed_pairs)} failed pairs")
            all_failed_pairs.update(file_failed_pairs)

        except Exception as e:
            print(f"Error loading {failed_pairs_file}: {e}")
            continue

    return all_failed_pairs


def main():
    folder = DATA_DIR
    json_files = glob(os.path.join(folder, "*.json"))

    if not json_files:
        print(f"No JSON files found in {folder}")
        return

    print(f"Processing {len(json_files)} files from {folder}")
    print("=" * 80)

    # Load failed pairs to exclude (union across all files)
    print("\nLoading failed pairs from multiple files:")
    excluded_pairs = load_failed_pairs(FAILED_PAIRS_FILES)
    print(
        f"\nExcluding {len(excluded_pairs)} unique QID-document pairs (union across all files)"
    )
    print("=" * 80)

    # Prepare CSV output
    csv_filename = CSV_FILENAME
    csv_data = []

    # Store baseline metrics (BASIC, none, none)
    baseline_metrics = None

    total_valid = 0
    total_gt_zero = 0
    total_gt_positive = 0
    total_excluded = 0

    # First pass: find and process baseline first
    baseline_file = None
    other_files = []

    for file in sorted(json_files):
        filename = os.path.basename(file)
        prompt_type, attack_type, mitigation_type = parse_filename(filename)
        if (
            prompt_type == "BASIC"
            and attack_type == "none"
            and mitigation_type == "none"
        ):
            baseline_file = file
        else:
            other_files.append(file)

    # Process baseline first, then others
    files_to_process = []
    if baseline_file:
        files_to_process.append(baseline_file)
    files_to_process.extend(other_files)

    for file in files_to_process:
        try:
            (
                overall_mae,
                gt_zero_mae,
                gt_positive_mae,
                overall_jer,
                gt_zero_jer,
                gt_positive_jer,
                valid_count,
                gt_zero_count,
                gt_positive_count,
                total_count,
                excluded_count,
            ) = process_file(file, excluded_pairs)

            filename = os.path.basename(file)
            prompt_type, attack_type, mitigation_type = parse_filename(filename)

            print(f"\n{filename}:")
            print(f"  Overall MAE: {overall_mae:.4f} (n={valid_count})")
            print(f"  MAE for GT=0: {gt_zero_mae:.4f} (n={gt_zero_count})")
            print(f"  MAE for GT>0: {gt_positive_mae:.4f} (n={gt_positive_count})")
            print(f"  Overall JER: {overall_jer:.2f}% (n={valid_count})")
            print(f"  JER for GT=0: {gt_zero_jer:.2f}% (n={gt_zero_count})")
            print(f"  JER for GT>0: {gt_positive_jer:.2f}% (n={gt_positive_count})")
            print(f"  Excluded pairs: {excluded_count}")

            # Show failed experiments count
            failed_count = total_count - valid_count
            if failed_count > 0:
                print(f"  Failed experiments (no LLM score): {failed_count}")

            total_excluded += excluded_count

            # Check if this is the baseline (BASIC, none, none)
            is_baseline = (
                prompt_type == "BASIC"
                and attack_type == "none"
                and mitigation_type == "none"
            )

            # Store baseline metrics for delta calculations
            if is_baseline:
                baseline_metrics = {
                    "overall_mae": overall_mae,
                    "mae_gt_zero": gt_zero_mae,
                    "mae_gt_positive": gt_positive_mae,
                    "overall_jer": overall_jer,
                    "jer_gt_zero": gt_zero_jer,
                    "jer_gt_positive": gt_positive_jer,
                }
                print("  *** BASELINE EXPERIMENT ***")

            # Calculate deltas if baseline exists
            delta_overall_mae = "NaN"
            delta_mae_gt_zero = "NaN"
            delta_mae_gt_positive = "NaN"
            delta_overall_jer = "NaN"
            delta_jer_gt_zero = "NaN"
            delta_jer_gt_positive = "NaN"

            if baseline_metrics and not is_baseline:
                # Calculate deltas (current - baseline)
                if not (overall_mae != overall_mae) and not (
                    baseline_metrics["overall_mae"] != baseline_metrics["overall_mae"]
                ):
                    delta_overall_mae = round(
                        overall_mae - baseline_metrics["overall_mae"], 4
                    )
                if not (gt_zero_mae != gt_zero_mae) and not (
                    baseline_metrics["mae_gt_zero"] != baseline_metrics["mae_gt_zero"]
                ):
                    delta_mae_gt_zero = round(
                        gt_zero_mae - baseline_metrics["mae_gt_zero"], 4
                    )
                if not (gt_positive_mae != gt_positive_mae) and not (
                    baseline_metrics["mae_gt_positive"]
                    != baseline_metrics["mae_gt_positive"]
                ):
                    delta_mae_gt_positive = round(
                        gt_positive_mae - baseline_metrics["mae_gt_positive"], 4
                    )
                if not (overall_jer != overall_jer) and not (
                    baseline_metrics["overall_jer"] != baseline_metrics["overall_jer"]
                ):
                    delta_overall_jer = round(
                        overall_jer - baseline_metrics["overall_jer"], 2
                    )
                if not (gt_zero_jer != gt_zero_jer) and not (
                    baseline_metrics["jer_gt_zero"] != baseline_metrics["jer_gt_zero"]
                ):
                    delta_jer_gt_zero = round(
                        gt_zero_jer - baseline_metrics["jer_gt_zero"], 2
                    )
                if not (gt_positive_jer != gt_positive_jer) and not (
                    baseline_metrics["jer_gt_positive"]
                    != baseline_metrics["jer_gt_positive"]
                ):
                    delta_jer_gt_positive = round(
                        gt_positive_jer - baseline_metrics["jer_gt_positive"], 2
                    )

            # Add to CSV data
            csv_data.append(
                {
                    "filename": filename,
                    "prompt_type": prompt_type,
                    "attack_type": attack_type,
                    "mitigation_type": mitigation_type,
                    "overall_mae": round(overall_mae, 4)
                    if not (overall_mae != overall_mae)
                    else "NaN",  # Check for NaN
                    "mae_gt_zero": round(gt_zero_mae, 4)
                    if not (gt_zero_mae != gt_zero_mae)
                    else "NaN",
                    "mae_gt_positive": round(gt_positive_mae, 4)
                    if not (gt_positive_mae != gt_positive_mae)
                    else "NaN",
                    "overall_jer": round(overall_jer, 2)
                    if not (overall_jer != overall_jer)
                    else "NaN",
                    "jer_gt_zero": round(gt_zero_jer, 2)
                    if not (gt_zero_jer != gt_zero_jer)
                    else "NaN",
                    "jer_gt_positive": round(gt_positive_jer, 2)
                    if not (gt_positive_jer != gt_positive_jer)
                    else "NaN",
                    "delta_overall_mae": delta_overall_mae,
                    "delta_mae_gt_zero": delta_mae_gt_zero,
                    "delta_mae_gt_positive": delta_mae_gt_positive,
                    "delta_overall_jer": delta_overall_jer,
                    "delta_jer_gt_zero": delta_jer_gt_zero,
                    "delta_jer_gt_positive": delta_jer_gt_positive,
                    "valid_experiments": valid_count,
                    "gt_zero_count": gt_zero_count,
                    "gt_positive_count": gt_positive_count,
                    "failed_experiments": failed_count,
                    "total_experiments": total_count,
                }
            )

            total_valid += valid_count
            total_gt_zero += gt_zero_count
            total_gt_positive += gt_positive_count

        except Exception as e:
            print(f"Error processing {file}: {e}")

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"Total unique QID-document pairs excluded: {len(excluded_pairs)}")
    print(f"Total exclusions across all files: {total_excluded}")
    print(f"Total valid experiments (after exclusions): {total_valid}")
    print(f"Total GT=0 experiments: {total_gt_zero}")
    print(f"Total GT>0 experiments: {total_gt_positive}")

    # Ensure baseline row appears first in the CSV
    if csv_data:
        baseline_first = [
            row
            for row in csv_data
            if row.get("prompt_type") == "BASIC"
            and row.get("attack_type") == "none"
            and row.get("mitigation_type") == "none"
        ]
        non_baseline = [row for row in csv_data if row not in baseline_first]
        ordered_csv_data = baseline_first + non_baseline

        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "filename",
                "prompt_type",
                "attack_type",
                "mitigation_type",
                "overall_mae",
                "mae_gt_zero",
                "mae_gt_positive",
                "overall_jer",
                "jer_gt_zero",
                "jer_gt_positive",
                "delta_overall_mae",
                "delta_mae_gt_zero",
                "delta_mae_gt_positive",
                "delta_overall_jer",
                "delta_jer_gt_zero",
                "delta_jer_gt_positive",
                "valid_experiments",
                "gt_zero_count",
                "gt_positive_count",
                "failed_experiments",
                "total_experiments",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in ordered_csv_data:
                writer.writerow(row)

        print(f"\nResults saved to: {csv_filename}")


if __name__ == "__main__":
    main()
