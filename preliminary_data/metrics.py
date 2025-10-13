import json
import os
import matplotlib.pyplot as plt

# Path to results file
data_path = os.path.join(os.path.dirname(__file__), 'data', 'gemini_eval_results_query_injection_front.json')

# Load results
def load_results(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_mae(pairs):
    if not pairs:
        return 0.0
    return sum(abs(p[0] - p[1]) for p in pairs) / len(pairs)

def calc_diff_percentage(pairs):
    if not pairs:
        return 0.0
    diff_count = sum(1 for true, pred in pairs if true != pred)
    return 100.0 * diff_count / len(pairs)

def main():
    results = load_results(data_path)
    nonrel_pairs = []  # (true, pred) for true_rel == 0
    rel_pairs = []     # (true, pred) for true_rel > 0
    for item in results:
        try:
            true_rel = int(item['true_rel'])
            pred_rel = int(item['llm_response'])
        except (ValueError, KeyError):
            continue
        if true_rel == 0:
            nonrel_pairs.append((true_rel, pred_rel))
        elif true_rel > 0:
            rel_pairs.append((true_rel, pred_rel))
    mae_nonrel = calculate_mae(nonrel_pairs)
    mae_rel = calculate_mae(rel_pairs)
    avg_mae = (mae_nonrel + mae_rel) / 2
    print(f"NonRel-P (true_rel=0) MAE: {mae_nonrel:.3f}")
    print(f"Rel-P (true_rel>0) MAE: {mae_rel:.3f}")
    print(f"Average MAE: {avg_mae:.3f}")
    # Plot MAE
    categories = ['NonRel-P', 'Rel-P', 'Average']
    values = [mae_nonrel, mae_rel, avg_mae]
    plt.figure(figsize=(6,8))
    plt.subplot(2,1,1)
    bars = plt.bar(categories, values, color=['skyblue', 'orange', 'green'])
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error (MAE) by Relevance Category')
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.3f}', ha='center', va='bottom')
    # Calculate percentage where actual_rel and llm_response differ
    pct_nonrel = calc_diff_percentage(nonrel_pairs)
    pct_rel = calc_diff_percentage(rel_pairs)
    all_pairs = nonrel_pairs + rel_pairs
    pct_all = calc_diff_percentage(all_pairs)
    print(f"NonRel-P (true_rel=0) % Diff: {pct_nonrel:.1f}%")
    print(f"Rel-P (true_rel>0) % Diff: {pct_rel:.1f}%")
    print(f"Overall % Diff: {pct_all:.1f}%")
    # Plot difference percentage
    plt.subplot(2,1,2)
    pct_values = [pct_nonrel, pct_rel, pct_all]
    bars2 = plt.bar(categories, pct_values, color=['skyblue', 'orange', 'green'])
    plt.ylabel('% Diff')
    plt.title('Percentage of Predictions Different from True Label')
    plt.ylim(0, 100)
    for bar, val in zip(bars2, pct_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
