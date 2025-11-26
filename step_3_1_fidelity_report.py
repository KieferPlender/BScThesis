import json
import os
import numpy as np
from tqdm import tqdm
from bert_score import score
import logging

# Hide HuggingFace warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = 'data/mt_bench/neutralized_answers'
ORIGINAL_DIR = 'data/mt_bench/model_answers'

# Pairs to check (Original vs. Intervention)
COMPARISONS = [
    {
        "name": "GPT-4 Neutral Paraphrase",
        "orig": f"{ORIGINAL_DIR}/gpt-4-turbo.jsonl",
        "new":  f"{DATA_DIR}/gpt-4-turbo_neutral.jsonl"
    },
    {
        "name": "GPT-4 Back-Translation",
        "orig": f"{ORIGINAL_DIR}/gpt-4-turbo.jsonl",
        "new":  f"{DATA_DIR}/gpt-4-turbo_backtrans.jsonl"
    },
    {
        "name": "Vicuna Neutral Paraphrase",
        "orig": f"{ORIGINAL_DIR}/vicuna_13b.jsonl",
        "new":  f"{DATA_DIR}/vicuna_13b_neutral.jsonl"
    },
    {
        "name": "Vicuna Back-Translation",
        "orig": f"{ORIGINAL_DIR}/vicuna_13b.jsonl",
        "new":  f"{DATA_DIR}/vicuna_13b_backtrans.jsonl"
    }
]

def load_texts(filepath):
    texts = {}
    if not os.path.exists(filepath):
        print(f"⚠️ Missing: {filepath}")
        return {}
    with open(filepath, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Store by Question ID
            # We analyze the FINAL turn
            qid = item['question_id']
            text = item['choices'][0]['turns'][-1]
            texts[qid] = text
    return texts

def calculate_stats(pair):
    print(f"\n📊 Analyzing: {pair['name']}...")
    
    orig_dict = load_texts(pair['orig'])
    new_dict = load_texts(pair['new'])
    
    if not orig_dict or not new_dict:
        return

    # Match IDs
    common_ids = sorted(list(set(orig_dict.keys()) & set(new_dict.keys())))
    
    refs = [orig_dict[qid] for qid in common_ids]
    cands = [new_dict[qid] for qid in common_ids]
    
    print(f"   Calculating BERTScore for {len(refs)} pairs...")
    
    # Calculate Score
    # We use "roberta-large" (default) and lang="en"
    P, R, F1 = score(cands, refs, lang="en", verbose=True)
    
    # Statistics
    mean_score = F1.mean().item()
    std_score = F1.std().item()
    min_score = F1.min().item()
    
    print("-" * 40)
    print(f"   ✅ Mean Fidelity: {mean_score:.4f}")
    print(f"   📉 Std Dev:       {std_score:.4f}")
    print(f"   ⚠️ Lowest Score:  {min_score:.4f}")
    print("-" * 40)

def main():
    print("=== STEP 3.1: SEMANTIC FIDELITY REPORT ===")
    for comp in COMPARISONS:
        calculate_stats(comp)

if __name__ == "__main__":
    main()