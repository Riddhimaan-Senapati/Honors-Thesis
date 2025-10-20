"""
Track failed QID-document pairs across all experiments.

This script identifies query-document pairs that failed (llm_score is None) 
in any experiment run, allowing you to filter them out for fair comparisons.
"""

import os
import json
from glob import glob
from pathlib import Path
from typing import Dict, Set, Tuple
from collections import defaultdict

# =============================
# Configuration
# =============================

#RESULTS_DIR = 'data/gemma3_1b'
#OUTPUT_FILE = 'gemma3_1b_failed_pairs.json'
RESULTS_DIR = 'data/qwen3_0.6b'
OUTPUT_FILE = 'qwen3_0.6b_failed_pairs.json'
ATTACK_DATASETS_DIR = 'attack_datasets'


def extract_failed_pairs(results_dir: str) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], Dict]]:
    """
    Extract all failed pairs from result files.
    
    Returns:
        - experiments_with_failures: dict mapping experiment name to set of failed (qid, docid) pairs
        - pair_details: dict mapping (qid, docid) to {experiments_failed}
    """
    json_files = glob(os.path.join(results_dir, '*.json'))
    
    experiments_with_failures = {}
    pair_details = defaultdict(lambda: {
        'experiments_failed': []
    })
    
    for file_path in json_files:
        filename = os.path.basename(file_path)
        experiment_name = filename.replace('results_', '').replace('.json', '')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both old format (list) and new format (dict with results)
            if isinstance(data, dict) and 'results' in data:
                results = data['results']
            else:
                results = data
            
            failed_pairs_in_experiment = set()
            
            for item in results:
                qid = item.get('qid')
                docid = item.get('docid')
                llm_score = item.get('llm_score')
                
                # Check if this pair failed (llm_score is None)
                if llm_score is None:
                    pair_key = (qid, docid)
                    failed_pairs_in_experiment.add(f"{qid}_{docid}")
                    
                    # Track which experiments this pair failed in
                    pair_details[pair_key]['experiments_failed'].append(experiment_name)
            
            if failed_pairs_in_experiment:
                experiments_with_failures[experiment_name] = failed_pairs_in_experiment
                print(f"  {experiment_name}: {len(failed_pairs_in_experiment)} failed pairs")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return experiments_with_failures, pair_details


def load_attack_datasets_for_text(attack_datasets_dir: Path) -> Dict[Tuple[str, str], Dict]:
    """Load query and attacked document text from all attack dataset files."""
    pair_text_map = {}
    
    attack_types = ['none', 'prepend', 'append', 'scatter']
    
    for attack_type in attack_types:
        attack_file = attack_datasets_dir / f"{attack_type}_attacks.json"
        
        if not attack_file.exists():
            print(f"Warning: {attack_file} not found, skipping {attack_type}")
            continue
        
        try:
            with open(attack_file, 'r', encoding='utf-8') as f:
                attack_data = json.load(f)
            
            print(f"  Loaded {attack_type}_attacks.json: {len(attack_data)} pairs")
            
            for item in attack_data:
                qid = item.get('qid')
                docid = item.get('docid')
                pair_key = (qid, docid)
                
                # Initialize if first time seeing this pair
                if pair_key not in pair_text_map:
                    pair_text_map[pair_key] = {
                        'query_text': item.get('query_text', ''),
                        'attacked_documents': {}
                    }
                
                # Store the attacked document text for this attack type
                pair_text_map[pair_key]['attacked_documents'][attack_type] = item.get('attacked_doc_text', '')
        
        except Exception as e:
            print(f"Error loading {attack_file}: {e}")
    
    return pair_text_map


def main():
    results_dir = RESULTS_DIR
    output_file = OUTPUT_FILE
    attack_datasets_dir = Path(ATTACK_DATASETS_DIR)
    
    print(f"Analyzing results from: {results_dir}")
    print(f"Output file: {output_file}")
    print(f"Attack datasets dir: {attack_datasets_dir}")
    print("=" * 60)
    
    # Extract failed pairs
    experiments_with_failures, pair_details = extract_failed_pairs(results_dir)
    
    # Load attacked documents from attack datasets
    pair_text_map = {}
    if attack_datasets_dir.exists():
        print("\nLoading attacked documents from attack datasets...")
        pair_text_map = load_attack_datasets_for_text(attack_datasets_dir)
        print(f"Loaded text for {len(pair_text_map)} pairs from attack datasets")
    else:
        print(f"\nWarning: Attack datasets directory not found: {attack_datasets_dir}")
        print("Attacked document text will not be included.")
    
    # Convert to list for JSON serialization
    failed_pairs_list = []
    for (qid, docid), details in sorted(pair_details.items()):
        pair_key = (qid, docid)
        
        # Get query text and attacked documents from attack datasets
        query_text = ''
        attacked_documents = {}
        
        if pair_key in pair_text_map:
            query_text = pair_text_map[pair_key].get('query_text', '')
            attacked_documents = pair_text_map[pair_key].get('attacked_documents', {})
        
        failed_pairs_list.append({
            'qid': qid,
            'docid': docid,
            'query_text': query_text,
            'attacked_documents': attacked_documents,
            'num_experiments_failed': len(details['experiments_failed']),
            'experiments_failed': sorted(details['experiments_failed'])
        })
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Total experiments analyzed: {len(experiments_with_failures)}")
    print(f"Total unique failed pairs: {len(failed_pairs_list)}")
    
    # Count how many experiments each pair failed in
    fail_counts = defaultdict(int)
    for pair in failed_pairs_list:
        fail_counts[pair['num_experiments_failed']] += 1
    
    print("\nFailed pairs by number of experiments:")
    for num_fails in sorted(fail_counts.keys(), reverse=True):
        count = fail_counts[num_fails]
        print(f"  Failed in {num_fails} experiments: {count} pairs")
    
    # Create output structure
    output_data = {
        'summary': {
            'results_directory': results_dir,
            'total_experiments': len(experiments_with_failures),
            'total_failed_pairs': len(failed_pairs_list),
            'experiments_analyzed': sorted(experiments_with_failures.keys())
        },
        'failed_pairs': failed_pairs_list
    }
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nFailed pairs saved to: {output_file}")
    print("\nYou can use this file to filter out inconsistent pairs from your analysis.")


if __name__ == "__main__":
    main()

