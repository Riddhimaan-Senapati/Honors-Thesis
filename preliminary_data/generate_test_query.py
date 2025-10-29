import os
from collections import defaultdict

input_path = os.path.join(os.path.dirname(__file__), "llm4eval_dev_qrel_2024.txt")
output_path = os.path.join(os.path.dirname(__file__), "test_query.txt")

# Store per-query docs by relevance
query_docs = defaultdict(lambda: {"zero": [], "pos": []})

with open(input_path, "r", encoding="utf-8") as infile:
    for line in infile:
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        qid, _, docid, rel = parts
        rel = int(rel)
        if rel == 0:
            if len(query_docs[qid]["zero"]) < 5:
                query_docs[qid]["zero"].append(line)
        else:
            if len(query_docs[qid]["pos"]) < 5:
                query_docs[qid]["pos"].append(line)

with open(output_path, "w", encoding="utf-8") as outfile:
    for qid in query_docs:
        for line in query_docs[qid]["zero"]:
            outfile.write(line)
        for line in query_docs[qid]["pos"]:
            outfile.write(line)

print(
    f"Created {output_path} with up to 5 zero-relevance and 5 positive-relevance docs per query."
)
