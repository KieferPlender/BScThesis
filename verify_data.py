from datasets import load_dataset
import pandas as pd
from collections import Counter
from langdetect import detect, LangDetectException
import random

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def main():
    print("Loading dataset...")
    dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    df = pd.DataFrame(dataset)
    
    print("\n=== STEP 1: Extract all assistant responses ===")
    rows = []
    for _, row in df.iterrows():
        # Model A
        for turn in row['conversation_a']:
            if turn['role'] == 'assistant':
                rows.append({
                    'model': row['model_a'], 
                    'text': turn['content'],
                    'conv_id': row.get('conversation_id', 'unknown')
                })
        # Model B
        for turn in row['conversation_b']:
            if turn['role'] == 'assistant':
                rows.append({
                    'model': row['model_b'], 
                    'text': turn['content'],
                    'conv_id': row.get('conversation_id', 'unknown')
                })
    
    df_all = pd.DataFrame(rows)
    print(f"Total assistant responses: {len(df_all)}")
    
    print("\n=== STEP 2: Filter for English FIRST ===")
    print("Detecting English (this may take a while)...")
    # Sample for speed - detect on first 10k to estimate
    sample_size = min(10000, len(df_all))
    df_sample = df_all.sample(sample_size, random_state=42)
    df_sample['is_english'] = df_sample['text'].apply(is_english)
    
    english_ratio = df_sample['is_english'].mean()
    estimated_english = int(len(df_all) * english_ratio)
    print(f"Estimated English responses: {estimated_english:,} ({english_ratio*100:.1f}%)")
    
    # For verification, actually filter a subset
    print("\nFiltering sample for language detection...")
    df_sample_english = df_sample[df_sample['is_english']]
    
    print("\n=== STEP 3: Top 10 models from ENGLISH responses ===")
    english_counts = df_sample_english['model'].value_counts()
    print("\nTop 15 models by English response count (from sample):")
    for i, (model, count) in enumerate(english_counts.head(15).items(), 1):
        # Extrapolate to full dataset
        estimated_full = int(count * (len(df_all) / sample_size))
        print(f"{i:2d}. {model:25s} {count:5d} (est. full: {estimated_full:,})")
    
    print("\n=== STEP 4: Random sample verification ===")
    print("Sampling 3 random responses from each Top 10 model for manual verification:\n")
    
    top_10_models = english_counts.head(10).index.tolist()
    
    for model in top_10_models:
        model_responses = df_sample_english[df_sample_english['model'] == model]
        samples = model_responses.sample(min(3, len(model_responses)), random_state=42)
        
        print(f"\n{'='*80}")
        print(f"MODEL: {model}")
        print(f"{'='*80}")
        
        for idx, (_, row) in enumerate(samples.iterrows(), 1):
            text_preview = row['text'][:200].replace('\n', ' ')
            print(f"\nSample {idx}:")
            print(f"  {text_preview}...")
            print(f"  [Full length: {len(row['text'])} chars]")
    
    print("\n\n=== COMPARISON ===")
    print("Current Top 10 (overall frequency):")
    current_top_10 = ['vicuna-13b', 'koala-13b', 'oasst-pythia-12b', 'gpt-3.5-turbo', 
                      'alpaca-13b', 'gpt-4', 'claude-v1', 'RWKV-4-Raven-14B', 
                      'chatglm-6b', 'fastchat-t5-3b']
    for i, m in enumerate(current_top_10, 1):
        print(f"{i:2d}. {m}")
    
    print("\nTop 10 from English-first approach:")
    for i, m in enumerate(top_10_models, 1):
        status = "✓ SAME" if m in current_top_10 else "✗ DIFFERENT"
        print(f"{i:2d}. {m:25s} {status}")

if __name__ == "__main__":
    main()
