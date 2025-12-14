import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyfaidx
from intervaltree import Interval, IntervalTree
import argparse
import pickle

# params / paths
species = "Homo_sapiens"
rng = np.random.default_rng(42)

parser = argparse.ArgumentParser()
parser.add_argument("--window_size", type=int, default=400,
                    help="default: 400bp")
args = parser.parse_args()
window_size = args.window_size
window_left = window_size // 2
window_right = window_size // 2
window_abbr = f"{window_size/1000:.1f}"

print(f"window_size: {window_size}")
print(f"window_abbr: {window_abbr}")


fasta_path = f"../hg38/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna"
sv_insdel_path = f"../HGSVC3/annotation_table/variants_GRCh38_sv_insdel_HGSVC2024v1.0.tsv"
sv_inv_path = f"../HGSVC3/annotation_table/variants_GRCh38_sv_inv_HGSVC2024v1.0.tsv"
sv_small_path = f"../HGSVC3/annotation_table/variants_GRCh38_indel_insdel_HGSVC2024v1.0.tsv"

out_csv = f"{species}/{species}_balanced_data_{window_abbr}kb.csv"
pos_out_csv = f"{species}/{species}_balanced_data_{window_abbr}kb_pos.csv"
hist_pdf = f"{species}/{species}_neg_nearest_breakpoint_hist_{window_abbr}kb.pdf"

if not os.path.exists(species):
    os.makedirs(species)

fasta = pyfaidx.Fasta(fasta_path, as_raw=False, sequence_always_upper=True)

sv_inv = pd.read_csv(sv_inv_path, sep="\t") # only INV
sv_insdel = pd.read_csv(sv_insdel_path, sep="\t") # min SVLEN = 50; only indels
indel_insdel = pd.read_csv(sv_small_path, sep="\t") # max SVLEN = 49; only indels

columns_to_use = ['ID', '#CHROM', 'POS', 'END', 'SVTYPE', 'SVLEN']
sv_insdel_inv = pd.concat([sv_insdel[columns_to_use], sv_inv[columns_to_use]], ignore_index=True)
sv_insdel_inv.rename(columns={'#CHROM': 'CHROM'}, inplace=True)
indel_insdel.rename(columns={'#CHROM': 'CHROM'}, inplace=True)

# Filter for only placed sequences
chroms = list(range(1, 23)) + ['X', 'Y']
chrom_list = [f'chr{chrom}' for chrom in chroms]
sv_insdel_inv = sv_insdel_inv[sv_insdel_inv['CHROM'].isin(chrom_list)]
indel_insdel = indel_insdel[indel_insdel['CHROM'].isin(chrom_list)]

# 0-based indexing
sv_insdel_inv['POS'] = sv_insdel_inv['POS'] - 1
sv_insdel_inv['END'] = sv_insdel_inv['END'] - 1
indel_insdel['POS'] = indel_insdel['POS'] - 1
indel_insdel['END'] = indel_insdel['END'] - 1

sv_counts = sv_insdel_inv['CHROM'].value_counts().to_dict()

# Define a basic mapping from RefSeq accession to chromosome names
accession_to_chr = {
    'NC_000001.11': 'chr1', 
    'NC_000002.12': 'chr2', 
    'NC_000003.12': 'chr3', 
    'NC_000004.12': 'chr4', 
    'NC_000005.10': 'chr5', 
    'NC_000006.12': 'chr6', 
    'NC_000007.14': 'chr7', 
    'NC_000008.11': 'chr8', 
    'NC_000009.12': 'chr9', 
    'NC_000010.11': 'chr10', 
    'NC_000011.10': 'chr11', 
    'NC_000012.12': 'chr12', 
    'NC_000013.11': 'chr13', 
    'NC_000014.9': 'chr14', 
    'NC_000015.10': 'chr15', 
    'NC_000016.10': 'chr16', 
    'NC_000017.11': 'chr17', 
    'NC_000018.10': 'chr18', 
    'NC_000019.10': 'chr19', 
    'NC_000020.11': 'chr20', 
    'NC_000021.9': 'chr21', 
    'NC_000022.11': 'chr22', 
    'NC_000023.11': 'chrX', 
    'NC_000024.10': 'chrY'
}

chr_to_accession = {value: key for key, value in accession_to_chr.items()}


# SV trees to populate dataframe
big_trees = {}
for (chrom, svt), g in sv_insdel_inv.groupby(["CHROM", "SVTYPE"], dropna=False):
    t = IntervalTree()
    for _, r in g.iterrows():
        pos = int(r["POS"])
        end = int(r["END"])
        t.addi(pos, end)
    big_trees[(chrom, svt)] = t

small_trees = {}
for (chrom, svt), g in indel_insdel.groupby(["CHROM", "SVTYPE"], dropna=False):
    if svt not in {"DEL", "INS"}:
        continue
    t = IntervalTree()
    for _, r in g.iterrows():
        pos = int(r["POS"])
        end = int(r["END"])
        t.addi(pos, end)
    small_trees[(chrom, svt)] = t

# ---- SAVE ----
with open("big_trees.pkl", "wb") as f:
    pickle.dump(big_trees, f)

with open("small_trees.pkl", "wb") as f:
    pickle.dump(small_trees, f)
    

def count_from_trees(chrom, start_left, end_left, start_right, end_right):
    def c(dct, key):
        t = dct.get(key)
        if not t:
            return 0
        return len(t.overlap(start_left, end_left)) + len(t.overlap(start_right, end_right))
    return {
        "num_del": c(big_trees, (chrom, "DEL")),
        "num_ins": c(big_trees, (chrom, "INS")),
        "num_inv": c(big_trees, (chrom, "INV")),
        "num_smalldel": c(small_trees, (chrom, "DEL")),
        "num_smallins": c(small_trees, (chrom, "INS")),
    }


def extract_discontig_seq(chrom_chr, left_start, left_end, right_start, right_end):
    chrom_acc = chr_to_accession.get(chrom_chr)
    L = len(fasta[chrom_acc])
    ls, le = max(0, left_start), min(L, left_end)
    rs, re = max(0, right_start), min(L, right_end)
    left = fasta[chrom_acc][ls:le].seq.upper()
    right = fasta[chrom_acc][rs:re].seq.upper()
    return left + right

# positives (label=1):
pos_rows = []
for r in sv_insdel_inv.itertuples(index=False):
    chrom = str(r.CHROM)
    pos = int(r.POS)
    end = int(r.END)
    svt = str(r.SVTYPE)
    svlen = int(r.SVLEN)

    left_start = max(0, pos - window_left)
    left_end = pos
    right_start = end
    right_end = end + window_right

    counts = count_from_trees(chrom, left_start, left_end, right_start, right_end)
    seq = extract_discontig_seq(chrom, left_start, left_end, right_start, right_end)

    pos_rows.append({
        "chrom": chrom,
        "sequence": seq,
        "label": 1,
        "sv_type": svt,
        "sv_len": svlen,
        "pos": pos,
        "end": end,
        "left_start": left_start, "left_end": left_end,
        "right_start": right_start, "right_end": right_end,
        **counts
    })

pos_df = pd.DataFrame(pos_rows)


# negatives (label=0):
all_bp = (pd.concat([
    sv_insdel_inv[["CHROM", "POS", "END"]],
    indel_insdel[["CHROM", "POS", "END"]],
], ignore_index=True))

bp_by_chrom = {}
for chrom, g in all_bp.groupby("CHROM", dropna=False):
    s = set(map(int, g["POS"].tolist()))
    s.update(map(int, g["END"].tolist()))
    bp_by_chrom[str(chrom)] = s

# chromosome lengths
chrom_lengths = {}
for acc in fasta.keys():
    if accession_to_chr.get(acc):
        chrom_lengths[accession_to_chr.get(acc)] = len(fasta[acc])

n_pos = len(pos_df)
n_neg = n_pos + 10000

# sampling from chromosomes
total_len = sum(chrom_lengths.values()) or 1
quota = {c: max(1, int(round(n_neg * (L / total_len)))) for c, L in chrom_lengths.items()}
diff = n_neg - sum(quota.values())
if diff != 0:
    keys = list(quota.keys())
    for i in range(abs(diff)):
        k = keys[i % len(keys)]
        quota[k] += 1 if diff > 0 else -1
        if quota[k] < 0:
            quota[k] = 0
            
neg_rows = []
distances = {}

for chrom, L in chrom_lengths.items():
    need = quota.get(chrom, 0)
    if need <= 0:
        continue
    bp = bp_by_chrom.get(chrom)
    lo = window_left
    hi = max(lo + 1, L - window_right)
    picked = []
    tries = 0
    cap = max(1000, need * 50)
    while len(picked) < need and tries < cap:
        tries += 1
        pos = int(rng.integers(low=lo, high=hi))
        if pos in bp:
            continue
        picked.append(pos)

    # nearest-breakpoint distances (for plotting)
    arr = np.array(sorted(bp), dtype=int)
    distances[chrom] = []
    for pos in picked:
        left_start = max(0, pos - window_left)
        left_end = pos
        right_start = pos
        right_end = min(L, pos + window_right)

        counts = count_from_trees(chrom, left_start, left_end, right_start, right_end)
        seq = extract_discontig_seq(chrom, left_start, left_end, right_start, right_end)

        neg_rows.append({
            "chrom": chrom,
            "sequence": seq,
            "label": 0,
            "sv_type": "nonSV",
            "sv_len": 0,
            "pos": pos,
            "end": pos,
            "left_start": left_start, "left_end": left_end,
            "right_start": right_start, "right_end": right_end,            
            **counts
        })

        if arr.size:
            idx = np.searchsorted(arr, pos)
            d = []
            if idx > 0:
                d.append(abs(pos - arr[idx - 1]))
            if idx < arr.size:
                d.append(abs(pos - arr[idx]))
            if d:
                distances[chrom].append(int(min(d)))

neg_df = pd.DataFrame(neg_rows)

pattern = r'^[ATCG]+$'
balanced_data = pd.concat([pos_df, neg_df], ignore_index=True)
balanced_data = balanced_data[balanced_data['sequence'].str.len() == window_size]
balanced_data = balanced_data[balanced_data['sequence'].str.match(pattern)]


# Keep all positives; sample negatives to match
pos = balanced_data[balanced_data["label"] == 1]
neg = balanced_data[balanced_data["label"] == 0]

target = len(pos)
neg_bal = neg.sample(n=target, random_state=42, replace=False)
balanced_data = pd.concat([pos, neg_bal], ignore_index=True)

# Shuffle
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Final counts — pos(1): {sum(balanced_data['label']==1)}, neg(0): {sum(balanced_data['label']==0)}")
print(f"balanced_data.shape: {balanced_data.shape}")
balanced_data.to_csv(out_csv, index=False)

# distance histogram 
dist_rows = [(chrom, d) for chrom, lst in distances.items() for d in lst]
dist_df = pd.DataFrame(dist_rows, columns=["chrom", "nearest_breakpoint_distance"])
dist_csv = f"{species}/{species}_neg_nearest_breakpoint_distances_{window_abbr}kb.csv"
dist_df.to_csv(dist_csv, index=False)
print(f"[saved] {dist_csv}")
    
# plot histograms and save as PDF
data = {chrom: [d for d in lst if d > 0] for chrom, lst in distances.items() if lst}
n = len(data)
fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 3 * n), squeeze=False)
axes = [ax for row in axes for ax in row]

for ax, (chrom, dist_list) in zip(axes, sorted(data.items())):
    dmin, dmax = min(dist_list), max(dist_list)
    bins = np.logspace(np.log10(dmin), np.log10(dmax), 50)

    ax.hist(dist_list, bins=bins, edgecolor='black')
    ax.set_xscale('log')
    ax.set_title(f'Log-Scale Distances for {chrom}')
    ax.set_xlabel('Distance to nearest SV breakpoint (bp)')
    ax.set_ylabel('Frequency')

    pmin = int(np.floor(np.log10(dmin)))
    pmax = int(np.ceil(np.log10(dmax)))
    xticks = [10 ** p for p in range(pmin, pmax + 1)]
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

fig.tight_layout()
fig.savefig(hist_pdf, dpi=200)
plt.close(fig)
print(f"[saved] {hist_pdf}")
