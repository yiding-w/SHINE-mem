#!/usr/bin/env python3
"""Script to generate explore_SHINE_SWE_Pro.ipynb notebook."""
import json

def make_md_cell(source):
    """Create a markdown cell."""
    if isinstance(source, str):
        source = source.split('\n')
        source = [line + '\n' for line in source[:-1]] + [source[-1]]
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source}

def make_code_cell(source):
    """Create a code cell."""
    if isinstance(source, str):
        source = source.split('\n')
        source = [line + '\n' for line in source[:-1]] + [source[-1]]
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source}

cells = []

# Title
cells.append(make_md_cell([
    "# SHINE_SWE_Pro Dataset Analysis\n",
    "\n",
    "This notebook provides comprehensive analysis of the SHINE_SWE_Pro dataset:\n",
    "1. **Repo Statistics**: Number of repos, trajectories per repo, correctness breakdown\n",
    "2. **Token Length Distribution**: Using Qwen3.6-27B tokenizer with parallel processing\n",
    "   - Per repo / per role (system, user, assistant, tool)\n",
    "   - Summary tables with mean, max, min\n",
    "3. **Model Statistics**: Breakdown by model used for trajectory generation"
]))

# Cell 1: Imports and setup
cells.append(make_code_cell([
    "import json\n",
    "import os\n",
    "import sys\n",
    "import time\n",
    "from pathlib import Path\n",
    "from collections import Counter, defaultdict\n",
    "from concurrent.futures import ThreadPoolExecutor, as_completed\n",
    "from multiprocessing import cpu_count\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib\n",
    "matplotlib.rcParams['font.size'] = 11\n",
    "matplotlib.rcParams['figure.figsize'] = (14, 6)\n",
    "matplotlib.rcParams['figure.dpi'] = 100\n",
    "\n",
    "DATA_DIR = Path('../data/SHINE_SWE_Pro')\n",
    "MODEL_DIR = '../models/Qwen3.6-27B'\n",
    "\n",
    "jsonl_file = DATA_DIR / 'SWE_Pro-trajs.openai.jsonl'\n",
    "print(f'Data directory: {DATA_DIR}')\n",
    "print(f'JSONL file: {jsonl_file.name}')\n",
    "size = jsonl_file.stat().st_size\n",
    "size_str = f'{size/1e9:.2f} GB' if size > 1e9 else f'{size/1e6:.2f} MB'\n",
    "print(f'File size: {size_str}')"
]))

# Section 1: Tokenizer
cells.append(make_md_cell("## 1. Load Tokenizer (Qwen3.6-27B)"))
cells.append(make_code_cell([
    "from transformers import AutoTokenizer\n",
    "\n",
    "print(f'Loading tokenizer from: {MODEL_DIR}')\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)\n",
    "print(f'Tokenizer loaded: vocab_size={tokenizer.vocab_size}')\n",
    "print(f'Tokenizer type: {type(tokenizer).__name__}')"
]))

# Section 2: Data Loading
cells.append(make_md_cell([
    "## 2. Data Loading & Tokenization\n",
    "\n",
    "Since SHINE_SWE_Pro is relatively small (340 samples, ~55MB), we load all data and tokenize directly.\n",
    "Each sample's metadata (source_dataset, repo, correctness, model, resolved) and per-message token lengths are extracted."
]))

cells.append(make_code_cell("""import threading

NUM_TOKENIZE_WORKERS = min(64, cpu_count())

def process_sample(line, encode_fn):
    \"\"\"Process a single JSONL line: parse JSON and tokenize all messages.\"\"\"
    try:
        d = json.loads(line)
        messages = d.get('messages') or []
        source_dataset = d.get('source_dataset') or 'unknown'
        repo = d.get('repo') or 'unknown'
        correctness = d.get('correctness') or 'unknown'
        instance_id = d.get('instance_id') or ''
        trajectory_id = d.get('trajectory_id') or ''
        model = d.get('model') or 'unknown'
        resolved = d.get('resolved')
        
        msg_token_info = []
        for msg in messages:
            role = msg.get('role') or 'unknown'
            content = msg.get('content') or ''
            token_count = len(encode_fn(content))
            msg_token_info.append((role, token_count))
        
        return {
            'source_dataset': source_dataset,
            'repo': repo,
            'correctness': correctness,
            'instance_id': instance_id,
            'trajectory_id': trajectory_id,
            'model': model,
            'resolved': resolved,
            'msg_token_info': msg_token_info,
            'num_messages': len(messages),
        }
    except Exception as e:
        print(f'Error processing sample: {e}')
        return None

print(f'Processing {jsonl_file.name} with {NUM_TOKENIZE_WORKERS} workers...')
start_time = time.time()

# Read all lines
with open(jsonl_file, 'r', encoding='utf-8') as fp:
    lines = fp.readlines()

print(f'Loaded {len(lines):,} lines from file.')

# Process in parallel using ThreadPoolExecutor
all_data = []
encode_fn = tokenizer.encode

with ThreadPoolExecutor(max_workers=NUM_TOKENIZE_WORKERS) as executor:
    futures = {executor.submit(process_sample, line, encode_fn): idx 
               for idx, line in enumerate(lines)}
    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            all_data.append(result)

elapsed = time.time() - start_time
print(f'\\nDone! Processed {len(all_data):,} samples in {elapsed:.1f}s')
print(f'Unique repos: {len(set(s["repo"] for s in all_data))}')
print(f'Unique models: {len(set(s["model"] for s in all_data))}')"""))

# Section 3: Repo Statistics
cells.append(make_md_cell([
    "## 3. Repo Statistics\n",
    "\n",
    "For each repo:\n",
    "- How many trajectories\n",
    "- How many correct / incorrect / unknown\n",
    "- Resolved rate"
]))

cells.append(make_code_cell("""# Build per-repo statistics
repo_stats = defaultdict(lambda: Counter())
repo_resolved = defaultdict(lambda: {'resolved': 0, 'unresolved': 0})

for sample in all_data:
    repo = sample['repo']
    correctness = sample['correctness']
    repo_stats[repo][correctness] += 1
    if sample['resolved']:
        repo_resolved[repo]['resolved'] += 1
    else:
        repo_resolved[repo]['unresolved'] += 1

# Summary table
print('=' * 110)
print(f'{"Repo":<45} {"#Trajectories":>14} {"#Correct":>9} {"#Incorrect":>11} {"#Unknown":>9} {"Resolved%":>10}')
print('=' * 110)

summary_rows = []
sorted_repos = sorted(repo_stats.items(), key=lambda x: sum(x[1].values()), reverse=True)
for repo, counts in sorted_repos:
    total = sum(counts.values())
    correct = counts.get('correct', 0)
    incorrect = counts.get('incorrect', 0)
    unknown = total - correct - incorrect
    resolved = repo_resolved[repo]['resolved']
    resolved_pct = resolved / total * 100 if total > 0 else 0
    print(f'{repo:<45} {total:>14,} {correct:>9,} {incorrect:>11,} {unknown:>9,} {resolved_pct:>9.1f}%')
    summary_rows.append({
        'Repo': repo, '#Trajectories': total,
        '#Correct': correct, '#Incorrect': incorrect, '#Unknown': unknown,
        'Resolved%': resolved_pct
    })

print('=' * 110)
total_all = len(all_data)
total_correct = sum(r['#Correct'] for r in summary_rows)
total_incorrect = sum(r['#Incorrect'] for r in summary_rows)
total_unknown = sum(r['#Unknown'] for r in summary_rows)
total_resolved = sum(repo_resolved[r]['resolved'] for r in repo_resolved)
total_resolved_pct = total_resolved / total_all * 100 if total_all > 0 else 0
print(f'{"TOTAL":<45} {total_all:>14,} {total_correct:>9,} {total_incorrect:>11,} {total_unknown:>9,} {total_resolved_pct:>9.1f}%')"""))

# Visualization: bar charts
cells.append(make_code_cell("""# Visualization: Trajectories per repo (bar chart)
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Left: Trajectory count per repo
repos_sorted = [r['Repo'] for r in sorted(summary_rows, key=lambda x: x['#Trajectories'], reverse=True)]
counts_sorted = [r['#Trajectories'] for r in sorted(summary_rows, key=lambda x: x['#Trajectories'], reverse=True)]

colors = plt.cm.Set3(np.linspace(0, 1, len(repos_sorted)))
axes[0].barh(repos_sorted, counts_sorted, color=colors)
axes[0].set_xlabel('Number of Trajectories')
axes[0].set_title('Trajectories per Repo')
axes[0].invert_yaxis()
for i, v in enumerate(counts_sorted):
    axes[0].text(v + 0.5, i, str(v), va='center', fontsize=9)

# Right: Correctness breakdown (stacked bar)
repos_for_plot = repos_sorted
correct_counts = [repo_stats[r].get('correct', 0) for r in repos_for_plot]
incorrect_counts = [repo_stats[r].get('incorrect', 0) for r in repos_for_plot]

y_pos = np.arange(len(repos_for_plot))
axes[1].barh(y_pos, correct_counts, color='#4CAF50', label='Correct')
axes[1].barh(y_pos, incorrect_counts, left=correct_counts, color='#F44336', label='Incorrect')
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(repos_for_plot)
axes[1].set_xlabel('Number of Trajectories')
axes[1].set_title('Correctness Breakdown per Repo')
axes[1].invert_yaxis()
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.show()"""))

# Pie charts
cells.append(make_code_cell("""# Pie chart: Overall correctness distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Correctness
labels_corr = ['Correct', 'Incorrect']
sizes_corr = [total_correct, total_incorrect]
colors_corr = ['#4CAF50', '#F44336']
if total_unknown > 0:
    labels_corr.append('Unknown')
    sizes_corr.append(total_unknown)
    colors_corr.append('#9E9E9E')

axes[0].pie(sizes_corr, labels=labels_corr, colors=colors_corr, autopct='%1.1f%%', startangle=90)
axes[0].set_title(f'Overall Correctness (N={total_all})')

# Right: Resolved
labels_res = ['Resolved', 'Unresolved']
sizes_res = [total_resolved, total_all - total_resolved]
colors_res = ['#2196F3', '#FF9800']

axes[1].pie(sizes_res, labels=labels_res, colors=colors_res, autopct='%1.1f%%', startangle=90)
axes[1].set_title(f'Overall Resolved Rate (N={total_all})')

plt.tight_layout()
plt.show()"""))

# Section 4: Model Statistics
cells.append(make_md_cell([
    "## 4. Model Statistics\n",
    "\n",
    "Breakdown by model used for trajectory generation."
]))

cells.append(make_code_cell("""# Model statistics
model_stats = defaultdict(lambda: Counter())
model_resolved = defaultdict(lambda: {'resolved': 0, 'unresolved': 0})

for sample in all_data:
    model = sample['model']
    correctness = sample['correctness']
    model_stats[model][correctness] += 1
    if sample['resolved']:
        model_resolved[model]['resolved'] += 1
    else:
        model_resolved[model]['unresolved'] += 1

print('=' * 100)
print(f'{"Model":<40} {"#Trajectories":>14} {"#Correct":>9} {"#Incorrect":>11} {"Resolved%":>10}')
print('=' * 100)

for model in sorted(model_stats.keys()):
    counts = model_stats[model]
    total = sum(counts.values())
    correct = counts.get('correct', 0)
    incorrect = counts.get('incorrect', 0)
    resolved = model_resolved[model]['resolved']
    resolved_pct = resolved / total * 100 if total > 0 else 0
    print(f'{model:<40} {total:>14,} {correct:>9,} {incorrect:>11,} {resolved_pct:>9.1f}%')

print('=' * 100)"""))

# Section 5: Token Length Distribution
cells.append(make_md_cell([
    "## 5. Token Length Distribution\n",
    "\n",
    "Compute token length statistics:\n",
    "- Per repo: total, system, user, assistant, tool\n",
    "- Overall distribution"
]))

cells.append(make_code_cell("""# Build token length data structures
repo_token_stats = defaultdict(lambda: {'total': [], 'system': [], 'user': [], 'assistant': [], 'tool': []})
overall_token_stats = {'total': [], 'system': [], 'user': [], 'assistant': [], 'tool': []}

for sample in all_data:
    repo = sample['repo']
    msg_info = sample['msg_token_info']
    
    total_tokens = 0
    system_tokens = 0
    user_tokens = 0
    assistant_tokens = 0
    tool_tokens = 0
    
    for role, tc in msg_info:
        total_tokens += tc
        if role == 'system':
            system_tokens += tc
        elif role == 'user':
            user_tokens += tc
        elif role == 'assistant':
            assistant_tokens += tc
        elif role == 'tool':
            tool_tokens += tc
    
    repo_token_stats[repo]['total'].append(total_tokens)
    repo_token_stats[repo]['system'].append(system_tokens)
    repo_token_stats[repo]['user'].append(user_tokens)
    repo_token_stats[repo]['assistant'].append(assistant_tokens)
    repo_token_stats[repo]['tool'].append(tool_tokens)
    
    overall_token_stats['total'].append(total_tokens)
    overall_token_stats['system'].append(system_tokens)
    overall_token_stats['user'].append(user_tokens)
    overall_token_stats['assistant'].append(assistant_tokens)
    overall_token_stats['tool'].append(tool_tokens)

print(f'Token stats computed for {len(repo_token_stats)} repos.')"""))

# 5.1 Overall summary
cells.append(make_md_cell("### 5.1 Overall Token Length Summary"))
cells.append(make_code_cell("""def compute_stats(values):
    \"\"\"Compute mean, min, max, median, std for a list of values.\"\"\"
    if not values:
        return 0, 0, 0, 0, 0
    return int(np.mean(values)), int(np.min(values)), int(np.max(values)), int(np.median(values)), int(np.std(values))

print('Overall Token Length Distribution (tokens per sample)')
print('=' * 90)
print(f'{"Role":<12} {"Mean":>8} {"Min":>8} {"Max":>8} {"Median":>8} {"Std":>8}')
print('-' * 90)
for role in ['total', 'system', 'user', 'assistant', 'tool']:
    mean, mn, mx, med, std = compute_stats(overall_token_stats[role])
    print(f'{role:<12} {mean:>8,} {mn:>8,} {mx:>8,} {med:>8,} {std:>8,}')
print('=' * 90)"""))

# 5.2 Per-repo table
cells.append(make_md_cell("### 5.2 Per-Repo Token Length Summary Table"))
cells.append(make_code_cell("""# Build DataFrame for per-repo token stats
rows = []
for repo in sorted(repo_token_stats.keys()):
    stats = repo_token_stats[repo]
    n = len(stats['total'])
    total_mean, total_min, total_max, _, _ = compute_stats(stats['total'])
    sys_mean, sys_min, sys_max, _, _ = compute_stats(stats['system'])
    user_mean, user_min, user_max, _, _ = compute_stats(stats['user'])
    asst_mean, asst_min, asst_max, _, _ = compute_stats(stats['assistant'])
    tool_mean, tool_min, tool_max, _, _ = compute_stats(stats['tool'])
    rows.append({
        'Repo': repo, 'N': n,
        'Total_Mean': total_mean, 'Total_Min': total_min, 'Total_Max': total_max,
        'System_Mean': sys_mean, 'System_Min': sys_min, 'System_Max': sys_max,
        'User_Mean': user_mean, 'User_Min': user_min, 'User_Max': user_max,
        'Assistant_Mean': asst_mean, 'Assistant_Min': asst_min, 'Assistant_Max': asst_max,
        'Tool_Mean': tool_mean, 'Tool_Min': tool_min, 'Tool_Max': tool_max,
    })

df_repo = pd.DataFrame(rows)

print('Token Length Distribution per Repo (tokens per sample)')
print('=' * 160)
print(f'{"Repo":<35} {"N":>4} | {"Total":^19} | {"System":^19} | {"User":^19} | {"Assistant":^19} | {"Tool":^19}')
print(f'{"":<35} {"":>4} | {"Mean":>6} {"Min":>5} {"Max":>6} | {"Mean":>6} {"Min":>5} {"Max":>6} | {"Mean":>6} {"Min":>5} {"Max":>6} | {"Mean":>6} {"Min":>5} {"Max":>6} | {"Mean":>6} {"Min":>5} {"Max":>6}')
print('-' * 160)
for _, row in df_repo.iterrows():
    repo_name = row['Repo'][:33]
    print(f'{repo_name:<35} {row["N"]:>4} | '
          f'{row["Total_Mean"]:>6,} {row["Total_Min"]:>5,} {row["Total_Max"]:>6,} | '
          f'{row["System_Mean"]:>6,} {row["System_Min"]:>5,} {row["System_Max"]:>6,} | '
          f'{row["User_Mean"]:>6,} {row["User_Min"]:>5,} {row["User_Max"]:>6,} | '
          f'{row["Assistant_Mean"]:>6,} {row["Assistant_Min"]:>5,} {row["Assistant_Max"]:>6,} | '
          f'{row["Tool_Mean"]:>6,} {row["Tool_Min"]:>5,} {row["Tool_Max"]:>6,}')
print('=' * 160)"""))

# DataFrame display
cells.append(make_code_cell("""# Display as styled pandas DataFrame
display_df = df_repo.set_index('Repo')
display_df.columns = pd.MultiIndex.from_tuples([
    ('Count', 'N'),
    ('Total', 'Mean'), ('Total', 'Min'), ('Total', 'Max'),
    ('System', 'Mean'), ('System', 'Min'), ('System', 'Max'),
    ('User', 'Mean'), ('User', 'Min'), ('User', 'Max'),
    ('Assistant', 'Mean'), ('Assistant', 'Min'), ('Assistant', 'Max'),
    ('Tool', 'Mean'), ('Tool', 'Min'), ('Tool', 'Max'),
])
display_df"""))

# 5.3 Visualizations
cells.append(make_md_cell("### 5.3 Token Length Distribution (Visualization)"))

# Box plot
cells.append(make_code_cell("""# Box plot: Total token length distribution per repo
fig, ax = plt.subplots(figsize=(16, 8))

repos_vis = sorted(repo_token_stats.keys(), key=lambda r: np.mean(repo_token_stats[r]['total']), reverse=True)
data_to_plot = [repo_token_stats[repo]['total'] for repo in repos_vis]
repo_labels = [str(r)[:35] for r in repos_vis]

bp = ax.boxplot(data_to_plot, vert=False, tick_labels=repo_labels, patch_artist=True, showfliers=False, whis=[5, 95])
colors_bp = plt.cm.Set3(np.linspace(0, 1, len(repos_vis)))
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color)
ax.set_xlabel('Total Tokens per Sample')
ax.set_title('Total Token Length Distribution per Repo (5th-95th percentile)')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()"""))

# Histograms
cells.append(make_code_cell("""# Histogram: Overall total token length distribution
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

roles_to_plot = ['total', 'system', 'assistant', 'tool']
titles = ['Total Tokens', 'System Tokens', 'Assistant Tokens', 'Tool Tokens']
colors_hist = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

for idx, (role, title, color) in enumerate(zip(roles_to_plot, titles, colors_hist)):
    ax = axes[idx // 2][idx % 2]
    data = overall_token_stats[role]
    ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {int(np.mean(data)):,}')
    ax.axvline(np.median(data), color='green', linestyle=':', linewidth=1.5, label=f'Median: {int(np.median(data)):,}')
    ax.set_xlabel('Token Count')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{title} Distribution (N={len(data)})')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()"""))

# Stacked bar
cells.append(make_code_cell("""# Stacked bar chart: Token composition per repo (mean)
fig, ax = plt.subplots(figsize=(16, 8))

repos_vis = sorted(repo_token_stats.keys(), key=lambda r: np.mean(repo_token_stats[r]['total']), reverse=True)

system_means = [int(np.mean(repo_token_stats[r]['system'])) for r in repos_vis]
user_means = [int(np.mean(repo_token_stats[r]['user'])) for r in repos_vis]
assistant_means = [int(np.mean(repo_token_stats[r]['assistant'])) for r in repos_vis]
tool_means = [int(np.mean(repo_token_stats[r]['tool'])) for r in repos_vis]

y_pos = np.arange(len(repos_vis))
bar_height = 0.6

ax.barh(y_pos, system_means, bar_height, color='#4CAF50', label='System')
ax.barh(y_pos, user_means, bar_height, left=system_means, color='#2196F3', label='User')
left_2 = [s + u for s, u in zip(system_means, user_means)]
ax.barh(y_pos, assistant_means, bar_height, left=left_2, color='#FF9800', label='Assistant')
left_3 = [l + a for l, a in zip(left_2, assistant_means)]
ax.barh(y_pos, tool_means, bar_height, left=left_3, color='#9C27B0', label='Tool')

ax.set_yticks(y_pos)
ax.set_yticklabels([str(r)[:35] for r in repos_vis])
ax.set_xlabel('Mean Token Count')
ax.set_title('Mean Token Composition per Repo (by Role)')
ax.invert_yaxis()
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()"""))

# Section 6: Message Count
cells.append(make_md_cell([
    "## 6. Message Count Distribution\n",
    "\n",
    "Analyze the number of messages (turns) per trajectory."
]))

cells.append(make_code_cell("""# Message count statistics
msg_counts = [s['num_messages'] for s in all_data]

print('Message Count Statistics:')
print(f'  Mean: {np.mean(msg_counts):.1f}')
print(f'  Median: {np.median(msg_counts):.1f}')
print(f'  Min: {np.min(msg_counts)}')
print(f'  Max: {np.max(msg_counts)}')
print(f'  Std: {np.std(msg_counts):.1f}')

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: Histogram of message counts
axes[0].hist(msg_counts, bins=30, color='#2196F3', alpha=0.7, edgecolor='black', linewidth=0.5)
axes[0].axvline(np.mean(msg_counts), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(msg_counts):.1f}')
axes[0].axvline(np.median(msg_counts), color='green', linestyle=':', linewidth=1.5, label=f'Median: {np.median(msg_counts):.1f}')
axes[0].set_xlabel('Number of Messages')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Message Count Distribution')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Right: Box plot per repo
repo_msg_counts = defaultdict(list)
for s in all_data:
    repo_msg_counts[s['repo']].append(s['num_messages'])

repos_vis = sorted(repo_msg_counts.keys(), key=lambda r: np.mean(repo_msg_counts[r]), reverse=True)
data_msg = [repo_msg_counts[r] for r in repos_vis]
labels_msg = [str(r)[:35] for r in repos_vis]

bp = axes[1].boxplot(data_msg, vert=False, tick_labels=labels_msg, patch_artist=True, showfliers=True)
colors_bp = plt.cm.Set3(np.linspace(0, 1, len(repos_vis)))
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color)
axes[1].set_xlabel('Number of Messages')
axes[1].set_title('Message Count per Repo')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()"""))

# Section 7: Correctness vs Token Length
cells.append(make_md_cell([
    "## 7. Correctness vs Token Length\n",
    "\n",
    "Analyze whether there is a relationship between token length and correctness."
]))

cells.append(make_code_cell("""# Compare token lengths for correct vs incorrect trajectories
correct_tokens = [overall_token_stats['total'][i] for i, s in enumerate(all_data) if s['correctness'] == 'correct']
incorrect_tokens = [overall_token_stats['total'][i] for i, s in enumerate(all_data) if s['correctness'] == 'incorrect']

print('Token Length by Correctness:')
print(f'  Correct   (N={len(correct_tokens):>3}): Mean={int(np.mean(correct_tokens)):>8,}, Median={int(np.median(correct_tokens)):>8,}, Std={int(np.std(correct_tokens)):>8,}')
print(f'  Incorrect (N={len(incorrect_tokens):>3}): Mean={int(np.mean(incorrect_tokens)):>8,}, Median={int(np.median(incorrect_tokens)):>8,}, Std={int(np.std(incorrect_tokens)):>8,}')

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: Overlapping histograms
axes[0].hist(correct_tokens, bins=25, color='#4CAF50', alpha=0.6, label=f'Correct (N={len(correct_tokens)})', edgecolor='black', linewidth=0.3)
axes[0].hist(incorrect_tokens, bins=25, color='#F44336', alpha=0.6, label=f'Incorrect (N={len(incorrect_tokens)})', edgecolor='black', linewidth=0.3)
axes[0].set_xlabel('Total Tokens')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Token Length Distribution: Correct vs Incorrect')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Right: Box plot comparison
bp = axes[1].boxplot([correct_tokens, incorrect_tokens], tick_labels=['Correct', 'Incorrect'], 
                     patch_artist=True, showfliers=False, whis=[5, 95])
bp['boxes'][0].set_facecolor('#4CAF50')
bp['boxes'][1].set_facecolor('#F44336')
axes[1].set_ylabel('Total Tokens')
axes[1].set_title('Token Length: Correct vs Incorrect (5th-95th percentile)')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()"""))

# Section 8: Summary
cells.append(make_md_cell([
    "## 8. Summary\n",
    "\n",
    "Key findings from the SHINE_SWE_Pro dataset analysis."
]))

cells.append(make_code_cell("""print('=' * 70)
print('SHINE_SWE_Pro Dataset Summary')
print('=' * 70)
print(f'Total samples:          {total_all:,}')
print(f'Unique repos:           {len(repo_token_stats)}')
print(f'Unique models:          {len(model_stats)}')
print(f'Correct trajectories:   {total_correct:,} ({total_correct/total_all*100:.1f}%)')
print(f'Incorrect trajectories: {total_incorrect:,} ({total_incorrect/total_all*100:.1f}%)')
print(f'Resolved trajectories:  {total_resolved:,} ({total_resolved_pct:.1f}%)')
print(f'\\nToken Statistics (per sample):')
print(f'  Total:     Mean={int(np.mean(overall_token_stats["total"])):>8,}, Median={int(np.median(overall_token_stats["total"])):>8,}')
print(f'  System:    Mean={int(np.mean(overall_token_stats["system"])):>8,}, Median={int(np.median(overall_token_stats["system"])):>8,}')
print(f'  User:      Mean={int(np.mean(overall_token_stats["user"])):>8,}, Median={int(np.median(overall_token_stats["user"])):>8,}')
print(f'  Assistant: Mean={int(np.mean(overall_token_stats["assistant"])):>8,}, Median={int(np.median(overall_token_stats["assistant"])):>8,}')
print(f'  Tool:      Mean={int(np.mean(overall_token_stats["tool"])):>8,}, Median={int(np.median(overall_token_stats["tool"])):>8,}')
print(f'\\nMessage Count (per sample):')
print(f'  Mean={np.mean(msg_counts):.1f}, Median={np.median(msg_counts):.1f}, Min={np.min(msg_counts)}, Max={np.max(msg_counts)}')
print('=' * 70)"""))

# Build notebook
notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.13.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

output_path = '/apdcephfs_zwfy/share_303937731/xiyuanwang/liuyewei/SHINE_V2_tmp/data_visualize/explore_SHINE_SWE_Pro.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'Notebook created: {output_path}')
