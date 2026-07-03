import json

nb_path = '/apdcephfs_zwfy/share_303937731/xiyuanwang/liuyewei/SHINE_V2_tmp/data_visualize/explore_SHINE_SWE_OPENSOURCE_v2.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

# Find the cell with load_all_datasets function
target_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        if 'def load_all_datasets' in source_text:
            target_idx = i
            break

assert target_idx is not None, "Could not find load_all_datasets cell!"

# New source code for this cell - with the CORRECT fix
new_source = [
    "import pyarrow as pa\n",
    "import pyarrow.ipc as ipc\n",
    "\n",
    "def load_all_datasets(data_dir):\n",
    "    \"\"\"Load all sub-datasets from HuggingFace Arrow format and concatenate them.\n",
    "    \n",
    "    Each sub-dataset directory contains Arrow IPC file(s) (data-XXXXX-of-YYYYY.arrow).\n",
    "    We use memory-mapped reading for efficiency.\n",
    "    \"\"\"\n",
    "    dataset_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.endswith('.openai')])\n",
    "    tables = []\n",
    "    dataset_names_list = []\n",
    "    \n",
    "    for ds_dir in dataset_dirs:\n",
    "        ds_name = ds_dir.name.replace('.openai', '')\n",
    "        print(f'  Loading: {ds_name}...')\n",
    "        arrow_files = sorted(ds_dir.glob('*.arrow'))\n",
    "        if not arrow_files:\n",
    "            print(f'    -> No arrow files found, skipping.')\n",
    "            continue\n",
    "        try:\n",
    "            ds_tables = []\n",
    "            for arrow_file in arrow_files:\n",
    "                reader = ipc.open_stream(str(arrow_file))\n",
    "                ds_tables.append(reader.read_all())\n",
    "            table = pa.concat_tables(ds_tables) if len(ds_tables) > 1 else ds_tables[0]\n",
    "            \n",
    "            # Add source_dataset column if not present\n",
    "            if 'source_dataset' not in table.column_names:\n",
    "                source_col = pa.array([ds_name] * table.num_rows, type=pa.string())\n",
    "                table = table.append_column('source_dataset', source_col)\n",
    "            \n",
    "            tables.append(table)\n",
    "            dataset_names_list.append(ds_name)\n",
    "            print(f'    -> {table.num_rows:,} samples loaded')\n",
    "        except Exception as e:\n",
    "            print(f'    -> ERROR loading dataset: {e}')\n",
    "            continue\n",
    "    \n",
    "    if not tables:\n",
    "        raise RuntimeError('No datasets could be loaded!')\n",
    "    \n",
    "    # Unify schema types across tables before concatenation.\n",
    "    # Some datasets have conflicting types for the same column (e.g., 'resolved' is bool\n",
    "    # in some datasets but string in others; 'tools' has different struct schemas).\n",
    "    # Strategy: for simple scalar conflicts, cast to string; for complex types, drop the column.\n",
    "    all_field_types = defaultdict(set)\n",
    "    for t in tables:\n",
    "        for field in t.schema:\n",
    "            all_field_types[field.name].add(str(field.type))\n",
    "    \n",
    "    # Find columns with conflicting types\n",
    "    conflict_cols = {col for col, types in all_field_types.items() if len(types) > 1}\n",
    "    if conflict_cols:\n",
    "        print(f'\\n  [INFO] Resolving type conflicts for columns: {conflict_cols}')\n",
    "        # Classify: simple scalar types can be cast to string; complex types must be dropped\n",
    "        simple_type_prefixes = ('bool', 'int', 'uint', 'float', 'double', 'string', 'utf8', 'large_string', 'large_utf8')\n",
    "        castable_cols = set()\n",
    "        drop_cols = set()\n",
    "        for col in conflict_cols:\n",
    "            types = all_field_types[col]\n",
    "            if all(any(t.startswith(p) for p in simple_type_prefixes) for t in types):\n",
    "                castable_cols.add(col)\n",
    "            else:\n",
    "                drop_cols.add(col)\n",
    "        \n",
    "        if drop_cols:\n",
    "            print(f'    Dropping complex conflicting columns: {drop_cols}')\n",
    "        if castable_cols:\n",
    "            print(f'    Casting scalar conflicting columns to string: {castable_cols}')\n",
    "        \n",
    "        unified_tables = []\n",
    "        for t in tables:\n",
    "            # Drop complex conflicting columns\n",
    "            for col in drop_cols:\n",
    "                if col in t.column_names:\n",
    "                    t = t.drop(col)\n",
    "            # Cast simple conflicting columns to string\n",
    "            for col in castable_cols:\n",
    "                if col in t.column_names:\n",
    "                    col_idx = t.schema.get_field_index(col)\n",
    "                    if t.schema.field(col_idx).type != pa.string():\n",
    "                        new_col = t.column(col).cast(pa.string())\n",
    "                        t = t.set_column(col_idx, pa.field(col, pa.string()), new_col)\n",
    "            unified_tables.append(t)\n",
    "        tables = unified_tables\n",
    "    \n",
    "    # Concatenate all datasets into one PyArrow Table\n",
    "    full_table = pa.concat_tables(tables, promote_options='default')\n",
    "    print(f'\\nTotal samples loaded: {full_table.num_rows:,}')\n",
    "    return full_table, dataset_names_list\n",
    "\n",
    "print('Loading all sub-datasets...')\n",
    "full_table, dataset_names_loaded = load_all_datasets(DATA_DIR)\n",
    "\n",
    "# Convert to list of dicts for easy access\n",
    "print('\\nConverting to records...')\n",
    "full_dataset = full_table.to_pylist()\n",
    "print(f'Conversion done. {len(full_dataset):,} records.')\n",
    "\n",
    "# Show dataset schema\n",
    "print(f'\\nDataset schema:')\n",
    "for field in full_table.schema:\n",
    "    print(f'  {field.name}: {field.type}')\n",
    "print(f'\\nFirst sample keys: {list(full_dataset[0].keys())}')"
]

# Update the cell
nb['cells'][target_idx]['source'] = new_source
nb['cells'][target_idx]['outputs'] = []
nb['cells'][target_idx]['execution_count'] = None

# Clear all outputs from all code cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# Write back
with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done! Notebook fixed successfully.")
