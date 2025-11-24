import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm

# Configuration
TARGET_MODELS = [
    'vicuna-13b', 'koala-13b', 'oasst-pythia-12b', 'gpt-3.5-turbo', 
    'alpaca-13b', 'gpt-4', 'claude-v1', 'RWKV-4-Raven-14B', 
    'chatglm-6b', 'palm-2'  # palm-2 replaces fastchat-t5-3b (English-first methodology)
]
OUTPUT_DIR = 'figures/step_0'
SAMPLE_SIZE = 1000  # Number of samples per model to analyze

# Function Word Categories
FUNCTION_WORDS = {
    'Pronouns': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'],
    'Prepositions': ['in', 'on', 'at', 'to', 'from', 'with', 'by', 'of', 'for', 'about'],
    'Articles_Determiners': ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'their'],
    'Conjunctions': ['and', 'or', 'but', 'because', 'although', 'while', 'if', 'when', 'however', 'therefore', 'moreover', 'furthermore', 'nevertheless', 'thus'],
    'Auxiliary_Verbs': ['be', 'am', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did', 
                        'will', 'would', 'can', 'could', 'should', 'may', 'might'],
    'Particles': ['not', 'up', 'off', 'out'], # 'to' is already in prepositions, handling overlap by keeping it there
    'Quantifiers': ['some', 'many', 'few', 'all', 'each']
}

# Flatten the list for easy iteration
ALL_FUNCTION_WORDS = [word for category in FUNCTION_WORDS.values() for word in category]

def setup_directories():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

from langdetect import detect, LangDetectException

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def load_and_filter_data():
    print("Loading dataset...")
    dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    # Load dataset from HuggingFace
    dataset = load_dataset("lmsys/chatbot_arena_conversations")
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(dataset['train'])
    
    # Flatten the conversation pairs
    # The dataset has columns like 'model_a', 'model_b', 'conversation_a', 'conversation_b'
    # We want a dataframe with 'model' and 'text'
    
    print("Flattening dataset...")
    rows = []
    # Limit to a subset for speed if needed, but for EDA we want good coverage. 
    # Let's process all but be mindful of memory.
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Model A
        if row['model_a'] in TARGET_MODELS:
            # conversation_a is a list of dicts [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}]
            # We only want the assistant's response
            for turn in row['conversation_a']:
                if turn['role'] == 'assistant':
                    rows.append({'model': row['model_a'], 'text': turn['content']})
        
        # Model B
        if row['model_b'] in TARGET_MODELS:
            for turn in row['conversation_b']:
                if turn['role'] == 'assistant':
                    rows.append({'model': row['model_b'], 'text': turn['content']})
                    
    df_filtered = pd.DataFrame(rows)
    print(f"Total responses for Top 10 models: {len(df_filtered)}")
    
    # Filter for English only
    print("Filtering for English text (this may take a while)...")
    # Use a sample if dataset is huge to speed up detection? 
    # No, user wants strict filtering. But langdetect is slow.
    # Let's apply it.
    tqdm.pandas()
    df_filtered = df_filtered[df_filtered['text'].progress_apply(is_english)]
    print(f"English responses: {len(df_filtered)}")
    
    # Balance the dataset
    # User requested "as much data as possible" WHILE balancing.
    # We limit all models to the size of the smallest class.
    min_count = df_filtered['model'].value_counts().min()
    sample_n = min_count
    
    print(f"Balancing dataset to {sample_n} samples per model (limited by smallest class)...")
    df_balanced = df_filtered.groupby('model').apply(lambda x: x.sample(sample_n, random_state=42)).reset_index(drop=True)
    
    print(f"Balanced dataset size: {len(df_balanced)} ({sample_n} per model)")
    return df_balanced

# Universal POS Tags
ALL_POS_TAGS = [
    'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 
    'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'
]

def analyze_style(df):
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    
    results = []
    
    print("Analyzing text features...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        doc = nlp(row['text'])
        
        # POS Counts
        pos_counts = Counter([token.pos_ for token in doc])
        total_tokens = len(doc)
        if total_tokens == 0: continue
        
        # Calculate frequency per 1000 tokens for ALL POS tags
        pos_data = {}
        for tag in ALL_POS_TAGS:
            pos_data[f'pos_{tag}'] = (pos_counts.get(tag, 0) / total_tokens) * 1000
        
        # Lexical Diversity
        unique_tokens = len(set([token.text.lower() for token in doc]))
        ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
        avg_sent_len = total_tokens / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0
        
        # Function Words
        text_lower = row['text'].lower()
        token_texts = [token.text.lower() for token in doc]
        token_counts = Counter(token_texts)
        
        fw_counts = {fw: token_counts[fw] for fw in ALL_FUNCTION_WORDS}
        
        entry = {
            'model': row['model'],
            'ttr': ttr,
            'avg_sent_len': avg_sent_len
        }
        entry.update(pos_data)
        entry.update(fw_counts)
        results.append(entry)
        
    return pd.DataFrame(results)

from sklearn.feature_extraction.text import CountVectorizer

def analyze_ngrams(df):
    print("Analyzing N-grams...")
    # Group texts by model
    model_texts = df.groupby('model')['text'].apply(lambda x: " ".join(x)).to_dict()
    
    ngram_results = {}
    
    for model, text in model_texts.items():
        # Unigrams
        vec_uni = CountVectorizer(ngram_range=(1, 1), max_features=20)
        X_uni = vec_uni.fit_transform([text])
        freq_uni = zip(vec_uni.get_feature_names_out(), X_uni.toarray()[0])
        ngram_results[f'{model}_unigrams'] = sorted(freq_uni, key=lambda x: x[1], reverse=True)
        
        # Bigrams
        vec_bi = CountVectorizer(ngram_range=(2, 2), max_features=20)
        X_bi = vec_bi.fit_transform([text])
        freq_bi = zip(vec_bi.get_feature_names_out(), X_bi.toarray()[0])
        ngram_results[f'{model}_bigrams'] = sorted(freq_bi, key=lambda x: x[1], reverse=True)
        
    return ngram_results

def plot_ngrams(ngram_results):
    print("Generating N-gram plots...")
    with open(f'{OUTPUT_DIR}/top_ngrams.txt', 'w') as f:
        for key, val in ngram_results.items():
            f.write(f"\n--- {key} ---\n")
            for word, count in val:
                f.write(f"{word}: {count}\n")
    print(f"N-gram analysis saved to {OUTPUT_DIR}/top_ngrams.txt")

def plot_results(df_stats):
    # Set improved theme with larger fonts
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams['figure.dpi'] = 300  # High resolution
    plt.rcParams['savefig.dpi'] = 300
    
    # Colorblind-friendly palette
    colors = sns.color_palette("tab10", n_colors=10)
    
    # 1. POS Distribution (All Tags)
    print("Generating POS plots...")
    
    # Split into three groups for readability: Content, Function, and Misc
    content_tags = ['pos_NOUN', 'pos_VERB', 'pos_ADJ', 'pos_ADV', 'pos_PROPN']
    function_tags = ['pos_DET', 'pos_ADP', 'pos_PRON', 'pos_AUX', 'pos_CCONJ', 'pos_SCONJ', 'pos_PART']
    misc_tags = ['pos_PUNCT', 'pos_NUM', 'pos_SYM', 'pos_INTJ', 'pos_X']
    
    # Helper to plot group
    def plot_pos_group(tags, title, filename):
        plt.figure(figsize=(16, 9))
        # Filter only existing columns
        valid_tags = [t for t in tags if t in df_stats.columns]
        if not valid_tags: return
        
        pos_data = pd.melt(df_stats, id_vars=['model'], value_vars=valid_tags, 
                           var_name='POS Type', value_name='Frequency (per 1000 tokens)')
        
        # Remove 'pos_' prefix for cleaner x-labels
        pos_data['POS Type'] = pos_data['POS Type'].str.replace('pos_', '')
        
        # Sort models by mean frequency for better visualization
        model_order = df_stats.groupby('model')[valid_tags].mean().mean(axis=1).sort_values(ascending=False).index
        
        ax = sns.boxplot(x='POS Type', y='Frequency (per 1000 tokens)', hue='model', 
                        data=pos_data, showfliers=False, palette=colors, hue_order=model_order)
        
        # Add mean markers
        for i, tag in enumerate(pos_data['POS Type'].unique()):
            tag_data = pos_data[pos_data['POS Type'] == tag]
            mean_val = tag_data['Frequency (per 1000 tokens)'].mean()
            ax.plot(i, mean_val, 'k*', markersize=10, label='Overall Mean' if i == 0 else '')
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('POS Tag', fontsize=14, fontweight='bold')
        plt.ylabel('Frequency (per 1000 tokens)', fontsize=14, fontweight='bold')
        
        # Improve legend
        handles, labels = ax.get_legend_handles_labels()
        # Filter out duplicate "Overall Mean" labels
        unique_labels = []
        unique_handles = []
        for h, l in zip(handles, labels):
            if l not in unique_labels:
                unique_labels.append(l)
                unique_handles.append(h)
        ax.legend(unique_handles, unique_labels, bbox_to_anchor=(1.02, 1), 
                 loc='upper left', frameon=True, shadow=True, ncol=1)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/{filename}', bbox_inches='tight')
        plt.close()

    plot_pos_group(content_tags, 'Content Words: Meaning-Carrying POS Tags', 'pos_content.png')
    plot_pos_group(function_tags, 'Function Words: Grammatical POS Tags', 'pos_function.png')
    plot_pos_group(misc_tags, 'Miscellaneous: Punctuation, Numbers & Symbols', 'pos_misc.png')
    
    # 2. Lexical Diversity
    print("Generating Lexical Diversity plots...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Sort models by TTR for better visualization
    ttr_order = df_stats.groupby('model')['ttr'].mean().sort_values(ascending=False).index
    
    # TTR Plot
    sns.boxplot(x='model', y='ttr', data=df_stats, ax=axes[0], 
                order=ttr_order, palette=colors, showfliers=False)
    axes[0].axhline(df_stats['ttr'].mean(), color='red', linestyle='--', 
                   linewidth=2, label='Overall Mean', alpha=0.7)
    axes[0].set_title('Type-Token Ratio (Lexical Diversity)', fontsize=16, fontweight='bold', pad=15)
    axes[0].set_xlabel('Model', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('TTR', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Sort models by sentence length
    sent_order = df_stats.groupby('model')['avg_sent_len'].mean().sort_values(ascending=False).index
    
    # Sentence Length Plot
    sns.boxplot(x='model', y='avg_sent_len', data=df_stats, ax=axes[1], 
                order=sent_order, palette=colors, showfliers=False)
    axes[1].axhline(df_stats['avg_sent_len'].mean(), color='red', linestyle='--', 
                   linewidth=2, label='Overall Mean', alpha=0.7)
    axes[1].set_title('Average Sentence Length', fontsize=16, fontweight='bold', pad=15)
    axes[1].set_xlabel('Model', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Tokens per Sentence', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lexical_diversity.png', bbox_inches='tight')
    plt.close()
    
    # 3. Function Words (Split by Category)
    print("Generating Function Word plots (Split by Category)...")
    
    fw_means = df_stats.groupby('model')[ALL_FUNCTION_WORDS].mean()
    
    for category, words in FUNCTION_WORDS.items():
        # Filter words that are actually in the dataframe (just in case)
        valid_words = [w for w in words if w in fw_means.columns]
        if not valid_words: continue
        
        # Melt for plotting
        fw_melted = pd.melt(fw_means.reset_index(), id_vars=['model'], value_vars=valid_words, 
                            var_name='Word', value_name='Avg Count')
        
        # Sort words by overall mean frequency for better visualization
        word_order = fw_melted.groupby('Word')['Avg Count'].mean().sort_values(ascending=False).index
        
        plt.figure(figsize=(18, 9))
        ax = sns.barplot(x='Word', y='Avg Count', hue='model', data=fw_melted, 
                        order=word_order, palette=colors)
        
        plt.title(f'Function Word Usage: {category.replace("_", " ")}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Word', fontsize=14, fontweight='bold')
        plt.ylabel('Average Count per Response', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Improve legend
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                  frameon=True, shadow=True, ncol=1, title='Model')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/fw_{category}.png', bbox_inches='tight')
        plt.close()

def save_statistics(df_stats):
    print("Saving summary statistics...")
    # Calculate mean and std for all numeric columns
    summary = df_stats.groupby('model').agg(['mean', 'std'])
    summary.to_csv(f'{OUTPUT_DIR}/summary_stats.csv')
    print(f"Statistics saved to {OUTPUT_DIR}/summary_stats.csv")

def main():
    setup_directories()
    df_data = load_and_filter_data()
    df_stats = analyze_style(df_data)
    
    # N-gram Analysis
    ngram_results = analyze_ngrams(df_data)
    plot_ngrams(ngram_results)
    
    plot_results(df_stats)
    save_statistics(df_stats)
    print(f"Analysis complete. Plots and stats saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
