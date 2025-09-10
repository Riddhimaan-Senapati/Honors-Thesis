# Attack Datasets Workflow

This workflow pre-generates attack datasets to make experiments more efficient and reproducible.

## Step 1: Create Attack Datasets

First, run the dataset creation script to generate 4 JSON files with pre-processed attack data:

```bash
python create_attack_datasets.py
```

This creates the following files in `attack_datasets/`:
- `none_attacks.json` - Original query-document pairs (no attack)
- `prepend_attacks.json` - Query prepended to document
- `append_attacks.json` - Query appended to document  
- `scatter_attacks.json` - Query terms scattered throughout document

Each file contains a list of dictionaries with:
- `qid`: Query ID
- `docid`: Document ID
- `query_text`: Original query text
- `original_doc_text`: Original document text
- `attacked_doc_text`: Document text after applying attack
- `ground_truth_score`: Relevance score from qrels

## Step 2: Run Experiments

Now you can run experiments using the pre-generated datasets:

```bash
# Basic experiment (no attack, no mitigation)
python run_experiments.py --prompt_type BASIC --attack_type none --mitigation_type none --limit 50

# Prepend attack with user prompt hardening
python run_experiments.py --prompt_type BASIC --attack_type prepend --mitigation_type user_prompt_hardening --limit 50

# Scatter attack with few-shot mitigation
python run_experiments.py --prompt_type RATIONALE --attack_type scatter --mitigation_type few_shot --limit 50
```

## Benefits

1. **Reproducibility**: All attacks are pre-generated with a fixed seed, ensuring consistent results across runs
2. **Efficiency**: No need to re-compute attacks during experiments
3. **Debugging**: You can inspect the attack datasets to verify they look correct
4. **Flexibility**: Easy to modify attack strategies by re-running the dataset creation script

## File Structure

```
Honors-Thesis/
├── create_attack_datasets.py     # Creates the 4 attack dataset files
├── run_experiments.py            # Runs experiments using pre-generated datasets
├── attack_datasets/              # Directory containing the 4 JSON files
│   ├── none_attacks.json
│   ├── prepend_attacks.json
│   ├── append_attacks.json
│   └── scatter_attacks.json
└── results/                      # Directory where experiment results are saved
    └── results_*.json
```

## Notes

- The `--seed` parameter in `run_experiments.py` is now just for documentation (the actual seed is used during dataset creation)
- If you modify attack strategies, re-run `create_attack_datasets.py` to regenerate the datasets
- The scatter attack uses a fixed seed (42) for reproducibility
