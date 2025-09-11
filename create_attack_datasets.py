"""
Create attack datasets for LLM query injection experiments.

This script generates 4 JSON files, one for each attack type:
- none_attacks.json: Original query-document pairs
- prepend_attacks.json: Query prepended to document
- append_attacks.json: Query appended to document  
- scatter_attacks.json: Query terms scattered throughout document

Each file contains a list of dictionaries with:
- qid: Query ID
- docid: Document ID
- query_text: Original query text
- original_doc_text: Original document text
- attacked_doc_text: Document text after applying attack
- ground_truth_score: Relevance score from qrels
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import jsonlines

# seed vale
SEED = 42

def load_qrels(filepath: Path) -> Dict[str, List[Tuple[str, int]]]:
    """Load qrels-like mapping from file of format: qid 0 docid rel."""
    qrels: Dict[str, List[Tuple[str, int]]] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            qid, _, docid, rel_str = parts
            try:
                rel = int(rel_str)
            except ValueError:
                continue
            qrels.setdefault(qid, []).append((docid, rel))
    return qrels


def load_documents(filepath: Path) -> Dict[str, str]:
    """Load documents from a jsonl file with fields: { 'docid': str, 'doc': str }"""
    mapping: Dict[str, str] = {}
    with jsonlines.open(str(filepath), mode="r") as reader:
        for obj in reader:
            docid = obj.get("docid")
            doc = obj.get("doc")
            if isinstance(docid, str) and isinstance(doc, str):
                mapping[docid] = doc
    return mapping


def load_queries(filepath: Path) -> Dict[str, str]:
    """Load queries from a TSV with lines: qid<TAB>query_text"""
    mapping: Dict[str, str] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            qid, qtext = parts
            mapping[qid] = qtext
    return mapping


def apply_attack(attack_type: str, doc_text: str, query_text: str, seed: Optional[int] = None) -> str:
    """Apply the specified attack to the document text."""
    if attack_type == "none":
        return doc_text
    elif attack_type == "prepend":
        return f"{query_text} {doc_text}"
    elif attack_type == "append":
        return f"{doc_text} {query_text}"
    elif attack_type == "scatter":
        rng = random.Random(seed)
        doc_tokens = doc_text.split()
        query_tokens = query_text.split()
        # Insert each query token at a random position in the document tokens
        for token in query_tokens:
            pos = rng.randint(0, max(0, len(doc_tokens)))
            doc_tokens.insert(pos, token)
        return " ".join(doc_tokens)
    else:
        raise ValueError(f"Unknown attack_type: {attack_type}")


def create_attack_dataset(attack_type: str, qrels: Dict[str, List[Tuple[str, int]]], 
                         queries: Dict[str, str], documents: Dict[str, str], 
                         seed: int = SEED) -> List[Dict]:
    """Create dataset for a specific attack type."""
    dataset = []
    
    for qid, pairs in qrels.items():
        query_text = queries.get(qid, "")
        if not query_text:
            print(f"Warning: Missing query text for qid={qid}")
            continue
            
        for docid, rel in pairs:
            doc_text = documents.get(docid, "")
            if not doc_text:
                print(f"Warning: Missing document text for docid={docid}")
                continue
                
            attacked_doc_text = apply_attack(attack_type, doc_text, query_text, seed=seed)
            
            dataset.append({
                "qid": qid,
                "docid": docid,
                "query_text": query_text,
                "original_doc_text": doc_text,
                "attacked_doc_text": attacked_doc_text,
                "ground_truth_score": rel
            })
    
    return dataset


def main():
    # Configuration
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "attack_datasets"
    
    # Data paths
    qrels_path = DATA_DIR / "llm4eval_dev_qrel_2024.txt"
    queries_path = DATA_DIR / "llm4eval_query_2024.txt"
    docs_path = DATA_DIR / "llm4eval_document_2024.jsonl"
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("Loading data...")
    qrels = load_qrels(qrels_path)
    queries = load_queries(queries_path)
    documents = load_documents(docs_path)
    
    print(f"Loaded {len(qrels)} queries, {len(queries)} query texts, {len(documents)} documents")
    
    # Attack types to generate
    attack_types = ["none", "prepend", "append", "scatter"]
    
    # Generate datasets for each attack type
    for attack_type in attack_types:
        print(f"\nCreating {attack_type} attack dataset...")
        dataset = create_attack_dataset(attack_type, qrels, queries, documents, seed=42)
        
        output_file = OUTPUT_DIR / f"{attack_type}_attacks.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(dataset)} pairs to {output_file}")
    
    print(f"\nAll attack datasets created in: {OUTPUT_DIR}")
    print("Files created:")
    for attack_type in attack_types:
        print(f"  - {attack_type}_attacks.json")


if __name__ == "__main__":
    main()
