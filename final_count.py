import pandas as pd
import re
import html
from datasets import load_dataset
from tqdm import tqdm
from langdetect import detect, LangDetectException

# ==========================================
# CONFIGURATION
# ==========================================
TARGET_MODELS = [
    'vicuna-13b', 'koala-13b', 'oasst-pythia-12b', 'gpt-3.5-turbo', 
    'alpaca-13b', 'gpt-4', 'claude-v1', 'RWKV-4-Raven-14B', 
    'chatglm-6b', 'palm-2'
]

def clean_text(text):
    if not isinstance(text, str): return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text) # Strip tags
    text = re.sub(r'\bdiv\s+div\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def check_final_counts():
    print("Loading dataset...")
    dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    
    # Initialize counters
    model_counts = {m: 0 for m in TARGET_MODELS}
    
    print("Scanning and Filtering (Simulating Step 0)...")
    for row in tqdm(dataset):
        # Check A
        if row['model_a'] in TARGET_MODELS:
            # Get response (usually index 1 is assistant)
            if len(row['conversation_a']) > 1:
                raw_text = row['conversation_a'][1]['content']
                cleaned = clean_text(raw_text)
                # Only count if it survives cleaning AND is English
                if cleaned and len(cleaned) > 10 and is_english(cleaned):
                    model_counts[row['model_a']] += 1

        # Check B
        if row['model_b'] in TARGET_MODELS:
            if len(row['conversation_b']) > 1:
                raw_text = row['conversation_b'][1]['content']
                cleaned = clean_text(raw_text)
                if cleaned and len(cleaned) > 10 and is_english(cleaned):
                    model_counts[row['model_b']] += 1

    print("\n" + "="*40)
    print("FINAL VALID SAMPLE COUNTS (Post-Cleaning)")
    print("="*40)
    
    # Sort by count to see the ranking
    sorted_counts = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (model, count) in enumerate(sorted_counts, 1):
        print(f"{rank}. {model.ljust(20)}: {count} samples")
        
    print("="*40)
    min_count = sorted_counts[-1][1]
    print(f"Lowest Class Size: {min_count}")
    if min_count >= 1000:
        print("✅ SAFE: All models have > 1000 samples.")
    else:
        print(f"⚠️ WARNING: {sorted_counts[-1][0]} has fewer than 1000 samples.")

if __name__ == "__main__":
    check_final_counts()