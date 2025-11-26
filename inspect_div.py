import pandas as pd
from datasets import load_dataset
import re

# ==========================================
# CONFIGURATION
# ==========================================
TARGET_MODELS = [
    'vicuna-13b', 'koala-13b', 'oasst-pythia-12b', 'gpt-3.5-turbo', 
    'alpaca-13b', 'gpt-4', 'claude-v1', 'RWKV-4-Raven-14B', 
    'chatglm-6b', 'palm-2'
]
KEYWORD = 'div'  # The suspect artifact

def inspect_all_models():
    print("Loading dataset...")
    dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    
    # Initialize counters
    stats = {model: {'count': 0, 'examples': []} for model in TARGET_MODELS}
    
    print(f"🔍 Scanning for '{KEYWORD}' artifacts across all models...")
    
    # We will scan a subset to be fast (e.g., first 20k rows), or remove [:20000] to scan all
    for i, row in enumerate(dataset):
        if i > 20000: break 
        
        # Check both Model A and Model B
        for key in ['model_a', 'model_b']:
            model_name = row[key]
            
            if model_name in TARGET_MODELS:
                # Get the response content
                idx = 0 if key == 'model_a' else 1
                # conversation structure is list of dicts. usually [user, model]
                if len(row[f'conversation_{key[-1]}']) > 1:
                    content = row[f'conversation_{key[-1]}'][1]['content']
                    
                    # Check for the artifact (unescaped or escaped)
                    if f"<{KEYWORD}" in content or f"&lt;{KEYWORD}" in content:
                        stats[model_name]['count'] += 1
                        
                        # Save the first 3 examples for inspection
                        if len(stats[model_name]['examples']) < 3:
                            # Snippet logic
                            match_index = content.find(KEYWORD)
                            start = max(0, match_index - 40)
                            end = min(len(content), match_index + 60)
                            snippet = content[start:end].replace('\n', ' ')
                            stats[model_name]['examples'].append(snippet)

    # Print Report
    print("\n" + "="*60)
    print(f"ARTIFACT REPORT: '{KEYWORD}'")
    print("="*60)
    
    for model in TARGET_MODELS:
        count = stats[model]['count']
        print(f"\n🔹 {model.upper()}: Found {count} times")
        
        if count > 0:
            print("   Examples:")
            for ex in stats[model]['examples']:
                print(f"   - \"...{ex}...\"")
        else:
            print("   (Clean)")

if __name__ == "__main__":
    inspect_all_models()