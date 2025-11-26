import pandas as pd
import numpy as np

def format_value(mean, std):
    """Format mean and std: 12.34 (1.20)"""
    if pd.isna(mean) or pd.isna(std):
        return "N/A"
    return f"{mean:.2f} ({std:.2f})"

def generate_lexical_diversity_table(df):
    """Generate LaTeX table for lexical diversity metrics."""
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Lexical Diversity Metrics by Model}")
    latex.append("\\label{tab:lexical_diversity}")
    latex.append("\\begin{tabular}{lcc}")
    latex.append("\\toprule")
    latex.append("Model & TTR & Avg. Sentence Length \\\\")
    latex.append("\\midrule")
    
    for model in df.index:
        # SAFETY FIX: Escape underscores for LaTeX
        model_name = model.replace('_', '\\_')
        
        ttr = format_value(df.loc[model, ('ttr', 'mean')], df.loc[model, ('ttr', 'std')])
        sent_len = format_value(df.loc[model, ('avg_sent_len', 'mean')], df.loc[model, ('avg_sent_len', 'std')])
        latex.append(f"{model_name} & {ttr} & {sent_len} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    return "\n".join(latex)

def generate_pos_table(df, pos_tags, caption, label):
    """Generate LaTeX table for a subset of POS tags."""
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    
    # Build column spec
    col_spec = "l" + "c" * len(pos_tags)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    
    # Header
    header = "Model & " + " & ".join(pos_tags) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")
    
    # Data rows
    for model in df.index:
        # SAFETY FIX: Escape underscores
        row = [model.replace('_', '\\_')]
        for tag in pos_tags:
            col_name = f'pos_{tag}'
            if (col_name, 'mean') in df.columns:
                val = format_value(df.loc[model, (col_name, 'mean')], df.loc[model, (col_name, 'std')])
                row.append(val)
            else:
                row.append("N/A")
        latex.append(" & ".join(row) + " \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    return "\n".join(latex)

def generate_function_word_table(df, words, caption, label, max_cols=6):
    """Generate LaTeX table for function words, splitting if needed."""
    tables = []
    
    # Split into chunks if too many words
    for i in range(0, len(words), max_cols):
        chunk = words[i:i+max_cols]
        
        latex = []
        if len(words) > max_cols:
            # Add part number if split
            part_num = (i // max_cols) + 1
            total_parts = (len(words) + max_cols - 1) // max_cols
            latex.append("\\begin{table}[htbp]")
            latex.append("\\centering")
            latex.append(f"\\caption{{{caption} (Part {part_num}/{total_parts})}}")
            latex.append(f"\\label{{{label}_part{part_num}}}")
        else:
            latex.append("\\begin{table}[htbp]")
            latex.append("\\centering")
            latex.append(f"\\caption{{{caption}}}")
            latex.append(f"\\label{{{label}}}")
        
        # Build column spec
        col_spec = "l" + "c" * len(chunk)
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex.append("\\toprule")
        
        # Header
        header = "Model & " + " & ".join([f"\\textit{{{w}}}" for w in chunk]) + " \\\\"
        latex.append(header)
        latex.append("\\midrule")
        
        # Data rows
        for model in df.index:
            row = [model.replace('_', '\\_')]
            for word in chunk:
                if (word, 'mean') in df.columns:
                    val = format_value(df.loc[model, (word, 'mean')], df.loc[model, (word, 'std')])
                    row.append(val)
                else:
                    row.append("N/A")
            latex.append(" & ".join(row) + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        tables.append("\n".join(latex))
    
    return "\n\n".join(tables)

def generate_summary_table(df):
    """Generate summary table with key distinguishing features."""
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Summary of Key Stylistic Features by Model (Mean (SD))}")
    latex.append("\\label{tab:summary_stylistic}")
    latex.append("\\begin{tabular}{lccccc}")
    latex.append("\\toprule")
    latex.append("Model & TTR & Sent. Len & NOUN & VERB & \\textit{you} \\\\")
    latex.append("\\midrule")
    
    for model in df.index:
        # Use format_value to get Mean (SD)
        ttr = format_value(df.loc[model, ('ttr', 'mean')], df.loc[model, ('ttr', 'std')])
        sent = format_value(df.loc[model, ('avg_sent_len', 'mean')], df.loc[model, ('avg_sent_len', 'std')])
        noun = format_value(df.loc[model, ('pos_NOUN', 'mean')], df.loc[model, ('pos_NOUN', 'std')])
        verb = format_value(df.loc[model, ('pos_VERB', 'mean')], df.loc[model, ('pos_VERB', 'std')])
        you = format_value(df.loc[model, ('you', 'mean')], df.loc[model, ('you', 'std')])
        
        model_name = model.replace('_', '\\_')
        latex.append(f"{model_name} & {ttr} & {sent} & {noun} & {verb} & {you} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    return "\n".join(latex)

def parse_ngrams_file(filename):
    """Parse the top_ngrams.txt file."""
    ngrams = {}
    current_model = None
    current_type = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('---'):
                # New section
                parts = line.replace('---', '').strip().split('_')
                if len(parts) >= 2:
                    current_model = '_'.join(parts[:-1])
                    current_type = parts[-1]
                    if current_model not in ngrams:
                        ngrams[current_model] = {}
                    ngrams[current_model][current_type] = []
            elif ':' in line and current_model and current_type:
                # Data line
                word, count = line.rsplit(':', 1)
                ngrams[current_model][current_type].append((word.strip(), int(count.strip())))
    
    return ngrams

def generate_ngram_table(ngrams, n=10):
    """Generate LaTeX tables for top N-grams."""
    tables = []
    
    for model in sorted(ngrams.keys()):
        model_name_latex = model.replace('_', '\\_').replace('-', '\\-') # Safe escaping
        
        # Unigrams table
        latex_uni = []
        latex_uni.append("\\begin{table}[htbp]")
        latex_uni.append("\\centering")
        latex_uni.append(f"\\caption{{Top {n} Unigrams for {model_name_latex}}}")
        latex_uni.append(f"\\label{{tab:unigrams_{model.replace('-', '_')}}}")
        latex_uni.append("\\begin{tabular}{lr}")
        latex_uni.append("\\toprule")
        latex_uni.append("Word & Frequency \\\\")
        latex_uni.append("\\midrule")
        
        if 'unigrams' in ngrams[model]:
            for word, count in ngrams[model]['unigrams'][:n]:
                latex_uni.append(f"\\textit{{{word}}} & {count:,} \\\\")
        
        latex_uni.append("\\bottomrule")
        latex_uni.append("\\end{tabular}")
        latex_uni.append("\\end{table}")
        
        # Bigrams table
        latex_bi = []
        latex_bi.append("\\begin{table}[htbp]")
        latex_bi.append("\\centering")
        latex_bi.append(f"\\caption{{Top {n} Bigrams for {model_name_latex}}}")
        latex_bi.append(f"\\label{{tab:bigrams_{model.replace('-', '_')}}}")
        latex_bi.append("\\begin{tabular}{lr}")
        latex_bi.append("\\toprule")
        latex_bi.append("Bigram & Frequency \\\\")
        latex_bi.append("\\midrule")
        
        if 'bigrams' in ngrams[model]:
            for word, count in ngrams[model]['bigrams'][:n]:
                latex_bi.append(f"\\textit{{{word}}} & {count:,} \\\\")
        
        latex_bi.append("\\bottomrule")
        latex_bi.append("\\end{tabular}")
        latex_bi.append("\\end{table}")
        
        tables.append("\n".join(latex_uni))
        tables.append("\n".join(latex_bi))
    
    return "\n\n".join(tables)

def main():
    # Load the CORRECT CSV (The normalized one)
    print("Loading normalized statistics...")
    df = pd.read_csv('figures/step_0/summary_stats_normalized.csv', header=[0, 1], index_col=0)
    
    # Parse the CORRECT N-grams text file
    print("Loading N-gram report...")
    ngrams = parse_ngrams_file('figures/step_0/top_ngrams_report.txt')
    
    # Create two output files: main tables and appendix tables
    main_output = 'figures/step_0/latex_tables_main.tex'
    appendix_output = 'figures/step_0/latex_tables_appendix.tex'
    
    # ========== MAIN TABLES ==========
    print("Generating Main Tables...")
    with open(main_output, 'w') as f:
        f.write("% Main Document Tables - Summary and Key Results\n")
        f.write("% Add \\usepackage{booktabs} to your LaTeX preamble\n\n")
        
        # Summary table (for main text)
        f.write("% ========== SUMMARY TABLE ==========\n")
        f.write(generate_summary_table(df))
        f.write("\n\n")
        
        # Lexical Diversity
        f.write("% ========== LEXICAL DIVERSITY ==========\n")
        f.write(generate_lexical_diversity_table(df))
        f.write("\n\n")
        
        # POS Tables
        f.write("% ========== PART-OF-SPEECH TAGS ==========\n")
        content_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']
        f.write(generate_pos_table(df, content_tags, 
                                   "Content Word POS Distribution (per 1000 tokens)", 
                                   "tab:pos_content"))
        f.write("\n\n")
        
        function_tags = ['DET', 'ADP', 'PRON', 'AUX', 'CCONJ', 'SCONJ', 'PART']
        f.write(generate_pos_table(df, function_tags, 
                                   "Function Word POS Distribution (per 1000 tokens)", 
                                   "tab:pos_function"))
        f.write("\n\n")
        
        misc_tags = ['PUNCT', 'NUM', 'SYM', 'INTJ', 'X']
        f.write(generate_pos_table(df, misc_tags, 
                                   "Miscellaneous POS Distribution (per 1000 tokens)", 
                                   "tab:pos_misc"))
        f.write("\n\n")
    
    # ========== APPENDIX TABLES ==========
    print("Generating Appendix Tables...")
    with open(appendix_output, 'w') as f:
        f.write("% Appendix Tables - Complete Analysis Results\n")
        f.write("% Add \\usepackage{booktabs} to your LaTeX preamble\n\n")
        
        # Add Lexical Diversity to appendix
        f.write("% ========== LEXICAL DIVERSITY ==========\n")
        f.write(generate_lexical_diversity_table(df))
        f.write("\n\n")
        
        # Add POS Tables to appendix
        f.write("% ========== PART-OF-SPEECH TAGS ==========\n")
        content_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']
        f.write(generate_pos_table(df, content_tags, 
                                   "Content Word POS Distribution (per 1000 tokens)", 
                                   "tab:pos_content_appendix"))
        f.write("\n\n")
        
        function_tags = ['DET', 'ADP', 'PRON', 'AUX', 'CCONJ', 'SCONJ', 'PART']
        f.write(generate_pos_table(df, function_tags, 
                                   "Function Word POS Distribution (per 1000 tokens)", 
                                   "tab:pos_function_appendix"))
        f.write("\n\n")
        
        misc_tags = ['PUNCT', 'NUM', 'SYM', 'INTJ', 'X']
        f.write(generate_pos_table(df, misc_tags, 
                                   "Miscellaneous POS Distribution (per 1000 tokens)", 
                                   "tab:pos_misc_appendix"))
        f.write("\n\n")
        
        f.write("% ========== COMPLETE FUNCTION WORD TABLES ==========\n\n")
        
        # Define all function word categories
        function_words = {
            'Pronouns': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'],
            'Prepositions': ['in', 'on', 'at', 'to', 'from', 'with', 'by', 'of', 'for', 'about'],
            'Articles_Determiners': ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'their'],
            'Conjunctions': ['and', 'or', 'but', 'because', 'although', 'while', 'if', 'when', 'however', 'therefore', 'moreover', 'furthermore', 'nevertheless', 'thus'],
            'Auxiliary_Verbs': ['be', 'am', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 
                               'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should', 'may', 'might'],
            'Particles': ['not', 'up', 'off', 'out'],
            'Quantifiers': ['some', 'many', 'few', 'all', 'each']
        }
        
        for category, words in function_words.items():
            f.write(f"% --- {category} ---\n")
            caption = f"{category.replace('_', ' ')} Usage (mean count per response)"
            label = f"tab:fw_{category.lower()}"
            f.write(generate_function_word_table(df, words, caption, label, max_cols=6))
            f.write("\n\n")
        
        # N-gram tables
        f.write("% ========== N-GRAM FREQUENCY TABLES ==========\n\n")
        f.write(generate_ngram_table(ngrams, n=10))
        f.write("\n")
    
    print(f"✅ Main tables saved to: {main_output}")
    print(f"✅ Appendix tables saved to: {appendix_output}")
    print("\nTables now include Mean and Standard Deviation (e.g., '12.34 (1.50)')")
    print("Files read: 'summary_stats_normalized.csv' and 'top_ngrams_report.txt'")

if __name__ == "__main__":
    main()