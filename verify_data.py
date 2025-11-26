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
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\bdiv\s+div\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def generate_retention_report():
    print("Loading dataset...")
    dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    
    # Dictionary to hold counts: {model: [raw_count, clean_count]}
    stats = {m: {'raw': 0, 'clean': 0} for m in TARGET_MODELS}
    
    print("Analyzing Data Retention (This checks every row)...")
    
    # We scan the full dataset to get accurate numbers
    for row in tqdm(dataset):
        # Check Model A
        if row['model_a'] in TARGET_MODELS:
            stats[row['model_a']]['raw'] += 1 # Raw count
            
            # Check if it survives cleaning
            if len(row['conversation_a']) > 1:
                raw_content = row['conversation_a'][1]['content']
                cleaned = clean_text(raw_content)
                if cleaned and len(cleaned) > 10 and is_english(cleaned):
                    stats[row['model_a']]['clean'] += 1

        # Check Model B
        if row['model_b'] in TARGET_MODELS:
            stats[row['model_b']]['raw'] += 1 # Raw count
            
            # Check if it survives cleaning
            if len(row['conversation_b']) > 1:
                raw_content = row['conversation_b'][1]['content']
                cleaned = clean_text(raw_content)
                if cleaned and len(cleaned) > 10 and is_english(cleaned):
                    stats[row['model_b']]['clean'] += 1

    # Print The Table
    print("\n" + "="*85)
    print(f"{'Model Name':<20} | {'Raw Count':<10} | {'Valid (Clean)':<15} | {'Retention Rate':<10}")
    print("-" * 85)
    
    for model, data in stats.items():
        raw = data['raw']
        clean = data['clean']
        # Avoid divide by zero
        rate = (clean / raw * 100) if raw > 0 else 0
        print(f"{model:<20} | {raw:<10} | {clean:<15} | {rate:.1f}%")
    print("="*85)

if __name__ == "__main__":
    generate_retention_report()