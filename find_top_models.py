from datasets import load_dataset
import pandas as pd
from collections import Counter

def find_top_models():
    print("Loading dataset...")
    dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    
    # Count model frequencies
    all_models = list(dataset['model_a']) + list(dataset['model_b'])
    counts = Counter(all_models)
    
    print("\nTop 10 Most Frequent Models:")
    top_10 = counts.most_common(10)
    for model, count in top_10:
        print(f"{model}: {count}")
        
    return [model for model, count in top_10]

if __name__ == "__main__":
    find_top_models()
