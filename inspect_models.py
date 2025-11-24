from datasets import load_dataset
import pandas as pd

def inspect_models():
    print("Loading dataset...")
    # Using the specific subset if needed, but usually 'default' or just the path works
    dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    
    print(f"Dataset loaded. Size: {len(dataset)}")
    
    # Extract unique models
    models_a = set(dataset['model_a'])
    models_b = set(dataset['model_b'])
    all_models = sorted(list(models_a.union(models_b)))
    
    print("\nUnique Models found:")
    for model in all_models:
        print(model)

if __name__ == "__main__":
    inspect_models()
