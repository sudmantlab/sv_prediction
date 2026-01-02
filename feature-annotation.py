import pandas as pd
import numpy as np
import pyBigWig
import pyfaidx
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
import argparse

# Parameters
num_processes = 32
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

input_file = f"{species}/{species}_balanced_data_{window_abbr}kb.csv"
output_file = f"{species}/{species}_balanced_data_{window_abbr}kb_annotated.csv"

# Define a mapping from RefSeq accession to chromosome names
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

# Filter for only placed sequences
chroms = list(range(1, 23)) + ['X', 'Y']
chrom_list = [f'chr{chrom}' for chrom in chroms]

one_hot_map = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}



### SET UP ###
balanced_data = pd.read_csv(input_file)
print(f"balanced_data.shape: {balanced_data.shape}")
print(balanced_data['label'].value_counts())

# Windows half-open and match sequence length
seq_len_ok = (balanced_data['left_end'] - balanced_data['left_start'] + balanced_data['right_end'] - balanced_data['right_start']) == balanced_data['sequence'].str.len()

# Breakpoints should be integer points
balanced_data['pos'] = balanced_data['pos'].astype(np.int64)
balanced_data['end'] = balanced_data['end'].astype(np.int64)



# SNP positions
with open(f'../HGSVC3/snp_positions_processed.pkl', 'rb') as f:
    snp_positions = pickle.load(f)

# Genes
gff_path = "../hg38/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.gff"
new_gene_annotations = pd.read_csv(
    gff_path,
    sep="\t",
    comment="#",
    header=None,
    names=[
        "SEQ_ID", "source", "FEATURE", "START", "END", "score",
        "strand", "phase", "attributes"
    ],
    dtype={"SEQ_ID": str, "source": str, "FEATURE": str}
)

gene_annotations = new_gene_annotations[['SEQ_ID', 'START', 'END', 'FEATURE']]

# Repeats
repeat_annotations = pd.read_csv(f"../UCSC_repeats/hg38_repeats_processed.csv")
repeat_type_list = list(repeat_annotations['REPEAT_TYPE'].unique())

# phyloP
phyloP_path = f"../phyloP/hg38.phyloP20way.bw"

# Recombination rate
bed_file_path = f"../recombination_rate/recombination_rate_hg38.bed"
recomb = pd.read_csv(bed_file_path, sep='\t', header=None, names=['chrom', 'start', 'end', 'score'])

# ENCODE cCREs
ENCODE_cCREs = pd.read_csv(f"../UCSC_regulation/ENCODE_cCREs.tsv", sep='\t')
ENCODE_cCREs.rename(columns={'#chrom': 'chrom'}, inplace=True)
ENCODE_cCREs.drop(['strand', 'thickStart', 'thickEnd', 'ccre', 'reserved', 'ucscLabel', 'accessionLabel', 'description'], axis=1, inplace=True)
cCRE_label_list = ENCODE_cCREs['encodeLabel'].unique()
cCRE_label_list.sort()

# DNase
DNase = pd.read_csv(f"../UCSC_regulation/DNase_Clusters.tsv", sep='\t')
DNase = DNase[DNase['chrom'].isin(chrom_list)]
DNase.drop(['#bin', 'name', 'sourceCount', 'sourceIds', 'sourceScores'], axis=1, inplace=True)

# TF peak clusters
tf_peak = pd.read_csv(f"../UCSC_regulation/TF_rPeak_Clusters.tsv", sep='\t')
tf_peak.rename(columns={'#chrom': 'chrom'}, inplace=True)
tf_peak.drop(['strand', 'thickStart', 'thickEnd', 'color', 'exp', 'json_table'], axis=1, inplace=True)




### ANNOTATIONS ###
# GC content
def calculate_gc_content(sequence):
    gc_count = sequence.count('G') + sequence.count('C')
    total_bases = len(sequence)
    if total_bases == 0:
        return 0
    return (gc_count / total_bases) * 100

balanced_data['gc_content'] = balanced_data['sequence'].apply(calculate_gc_content)
balanced_data.to_csv(output_file, index=False)
print("balanced_data with GC content saved")


# SNPs
snp_positions = {str(k): np.asarray(sorted(v), dtype=np.int64) for k, v in snp_positions.items()}
snps_per_chrom = {key: len(snp_positions[key]) for key in snp_positions}

def count_snps_in_interval(chrom: str, start: int, end: int, snp_dict: dict) -> int:
    """
    Count SNPs in half-open interval [start, end) using binary search.
    """
    arr = snp_dict.get(str(chrom))
    if arr is None or arr.size == 0:
        return 0
    lo = np.searchsorted(arr, start, side="left")
    hi = np.searchsorted(arr, end,   side="left")
    return int(hi - lo)

def apply_func_snp(row, snp_dict):
    # sum over both flanks
    c1 = count_snps_in_interval(row["chrom"], int(row["left_start"]),  int(row["left_end"]),  snp_dict)
    c2 = count_snps_in_interval(row["chrom"], int(row["right_start"]), int(row["right_end"]), snp_dict)
    return c1 + c2

records = balanced_data.to_dict("records")
with Pool(processes=num_processes) as pool:
    metrics = pool.starmap(apply_func_snp, [(r, snp_positions) for r in records])

balanced_data["full_snp_count"] = metrics
balanced_data.to_csv(output_file, index=False)
print("balanced_data snp counts saved")



# Genes, exons, CDS
def _build_feature_arrays_genes(feature_type_df: pd.DataFrame):
    d = {}
    for seq_id, g in feature_type_df.groupby("SEQ_ID", dropna=False):
        d[seq_id] = (
            g["START"].to_numpy(dtype=np.int64),
            g["END"].to_numpy(dtype=np.int64), 
        )
    return d

# distance from a point p to mulitple intervals
def _min_dist_point_to_intervals_genes(p: int, starts: np.ndarray, ends: np.ndarray):
    if starts.size == 0:
        return np.inf
    # If p < S -> S - p; if p > E -> p - E; else 0
    dist = np.where(p < starts, starts - p, np.where(p > ends, p - ends, 0))
    return int(dist.min())

# compute min distance per row to a gene feature from a point
def calculate_min_distances_gene(df_chunk, feature_arrays, chromosome_mapping):
    out = []
    for _, row in df_chunk.iterrows():
        chrom = str(row["chrom"])
        seq_id = chromosome_mapping.get(chrom)
        if seq_id is None or seq_id not in feature_arrays:
            out.append(np.inf)
            continue

        starts, ends = feature_arrays[seq_id]

        if int(row["label"]) == 1:
            p1 = int(row["pos"])
            p2 = int(row["end"])
            d1 = _min_dist_point_to_intervals_genes(p1, starts, ends)
            d2 = _min_dist_point_to_intervals_genes(p2, starts, ends)
            out.append(min(d1, d2))
        else:
            # background row: center point
            p = int(row["pos"])
            out.append(_min_dist_point_to_intervals_genes(p, starts, ends))
    return out

def parallel_process_gene(df, func, feature_type_df, chromosome_mapping, n_cores):
    feature_arrays = _build_feature_arrays_genes(feature_type_df)
    df_split = np.array_split(df, n_cores)
    with Pool(n_cores) as pool:
        parts = pool.starmap(func, [(chunk, feature_arrays, chromosome_mapping) for chunk in df_split])
    return np.concatenate(parts)

gene_feature_list = gene_annotations["FEATURE"].unique()
gene_annotations['START'] = gene_annotations['START'].astype(np.int64) - 1
gene_annotations['END'] = gene_annotations['END'].astype(np.int64) - 1

for feature in gene_feature_list:
    col = f"distance_to_{feature}"
    filtered = gene_annotations[gene_annotations["FEATURE"] == feature]
    tmp = parallel_process_gene(
    balanced_data, calculate_min_distances_gene, filtered, chr_to_accession, num_processes
    )
    tmp = np.asarray(tmp, dtype=float)
    tmp = np.where(np.isfinite(tmp), np.minimum(tmp, window_left), window_left).astype(int)
    balanced_data[col] = tmp

balanced_data.to_csv(output_file, index=False)
print("balanced_data with gene distances saved")


# Repeats
def _build_repeat_arrays(repeat_type_df: pd.DataFrame):
    d = {}
    for seq_id, g in repeat_type_df.groupby("QUERY_SEQUENCE", dropna=False):
        d[seq_id] = (
            g["BEGIN"].to_numpy(dtype=np.int64),
            g["END"].to_numpy(dtype=np.int64), 
        )
    return d

def _min_dist_point_to_intervals_repeats(p: int, starts: np.ndarray, ends: np.ndarray):
    if starts.size == 0:
        return np.inf
    dist = np.where(p < starts, starts - p, np.where(p > ends, p - ends, 0))
    return int(dist.min())

def calculate_min_distances_repeats(df_chunk, repeat_arrays, chromosome_mapping):
    out = []
    for _, row in df_chunk.iterrows():
        chrom = str(row["chrom"])
        seq_id = chromosome_mapping.get(chrom)
        if seq_id is None or seq_id not in repeat_arrays:
            out.append(np.inf)
            continue

        starts, ends = repeat_arrays[seq_id]

        if int(row["label"]) == 1:
            p1 = int(row["pos"])
            p2 = int(row["end"])
            d1 = _min_dist_point_to_intervals_repeats(p1, starts, ends)
            d2 = _min_dist_point_to_intervals_repeats(p2, starts, ends)
            out.append(min(d1, d2))
        else:
            p = int(row["pos"])
            out.append(_min_dist_point_to_intervals_repeats(p, starts, ends))
    return out

def parallel_process_repeats(df, func, repeat_type_df, chromosome_mapping, n_cores):
    repeat_arrays = _build_repeat_arrays(repeat_type_df)
    df_split = np.array_split(df, n_cores)
    with Pool(n_cores) as pool:
        parts = pool.starmap(func, [(chunk, repeat_arrays, chromosome_mapping) for chunk in df_split])
    return np.concatenate(parts)

# Run per repeat type
repeat_annotations['BEGIN'] = repeat_annotations['BEGIN'].astype(np.int64) - 1
for repeat_type in repeat_type_list:
    col = f"distance_to_{repeat_type}"
    filtered_repeat_type_df = repeat_annotations[repeat_annotations["REPEAT_TYPE"] == repeat_type]
    tmp = parallel_process_repeats(
    balanced_data, calculate_min_distances_repeats, filtered_repeat_type_df, chr_to_accession, num_processes
    )
    tmp = np.asarray(tmp, dtype=float)
    tmp = np.where(np.isfinite(tmp), np.minimum(tmp, window_left), window_left).astype(int)
    balanced_data[col] = tmp


balanced_data.to_csv(output_file, index=False)
print("balanced_data with repeat distances saved")


# PhyloP scores (average over both flanks of the context window)
def fetch_and_compute_phyloP(args):
    row, bw_path = args
    bw = pyBigWig.open(bw_path)

    ls, le = int(row["left_start"]), int(row["left_end"])
    rs, re = int(row["right_start"]), int(row["right_end"])
    chrom = str(row["chrom"])

    left_vals = np.array(bw.values(chrom, ls, le))
    right_vals = np.array(bw.values(chrom, rs, re))
    bw.close()

    scores = np.concatenate([left_vals, right_vals])
    if scores.size == 0:
        return 0.0
    avg_score = float(np.nanmean(scores))
    if np.isnan(avg_score):
        return 0.0
    return avg_score

def parallel_process_phylop(df, func, num_processes):
    with Pool(num_processes) as pool:
        args = [(row, phyloP_path) for _, row in df.iterrows()]
        results = pool.map(func, args)
    return results

results = parallel_process_phylop(balanced_data, fetch_and_compute_phyloP, num_processes)
phylop_df = pd.DataFrame({"avg_phyloP_scores": results})
balanced_data = pd.concat([balanced_data.reset_index(drop=True), phylop_df], axis=1)

balanced_data.to_csv(output_file, index=False)
print("balanced_data with phyloP scores saved")


recomb_by_chrom = {c: g.copy() for c, g in recomb.groupby('chrom')}

def _weighted_avg_over_flanks_recomb(recomb_chr_df, L1, R1, L2, R2):
    if recomb_chr_df.empty: 
        return 0.0
    starts = recomb_chr_df["start"].to_numpy(np.int64)
    ends   = recomb_chr_df["end"].to_numpy(np.int64)
    rates  = recomb_chr_df["score"].to_numpy(float)

    def _accum(L, R):
        ov = np.maximum(0, np.minimum(R, ends) - np.maximum(L, starts))
        valid = ov > 0
        if not valid.any(): 
            return 0.0, 0.0
        w = ov[valid].astype(float)
        return float(np.dot(w, rates[valid])), float(w.sum())

    n1,d1 = _accum(int(L1), int(R1))
    n2,d2 = _accum(int(L2), int(R2))
    den = d1 + d2
    return (n1 + n2)/den if den > 0 else 0.0

def process_chunk_recomb(chunk):
    out = []
    for _, row in chunk.iterrows():
        chrom = str(row["chrom"])
        L1,R1 = int(row["left_start"]),  int(row["left_end"])
        L2,R2 = int(row["right_start"]), int(row["right_end"])
        avg_rate = _weighted_avg_over_flanks_recomb(recomb_by_chrom.get(chrom, pd.DataFrame()), L1,R1,L2,R2)
        out.append(avg_rate)
    return out

def parallel_process_recomb(df, num_cores):
    chunks = np.array_split(df, num_cores)
    with Pool(num_cores) as pool:
        results = pool.map(process_chunk_recomb, chunks)
    flat = [x for sub in results for x in sub]
    return pd.Series(flat, index=df.index, name="avg_recomb_rate_full")

balanced_data["avg_recomb_rate_full"] = parallel_process_recomb(balanced_data, num_processes)
balanced_data.to_csv(output_file, index=False)
print("balanced_data with recombination rates saved")


# ENCODE cCREs
def _avg_ccre_over_interval(encode_chr_df: pd.DataFrame, L: int, R: int) -> float:
    if encode_chr_df.empty:
        return 0.0
    # overlap if (chromStart < R) & (chromEnd > L)
    m = (encode_chr_df["chromStart"].to_numpy(dtype=np.int64) < int(R)) & \
        (encode_chr_df["chromEnd"].to_numpy(dtype=np.int64)   > int(L))
    if not np.any(m):
        return 0.0
    z = encode_chr_df.loc[m, "zScore"].to_numpy(dtype=float)
    if z.size == 0:
        return 0.0
    v = np.nanmean(z)
    return float(0.0 if np.isnan(v) else v)

def process_ccre_rows(rows, encode_data_by_chrom):
    results = []
    for row in rows:
        chrom = str(row["chrom"])
        enc = encode_data_by_chrom.get(chrom, pd.DataFrame())

        L1, R1 = int(row["left_start"]),  int(row["left_end"])
        L2, R2 = int(row["right_start"]), int(row["right_end"])

        a1 = _avg_ccre_over_interval(enc, L1, R1)
        a2 = _avg_ccre_over_interval(enc, L2, R2)

        avg_z = (a1 + a2) / 2.0
        results.append(avg_z)
    return results

def parallel_process_ccre(df, encode_data, num_cores):
    chunks = np.array_split(df.to_dict("records"), num_cores)
    encode_by_chrom = {chrom: g for chrom, g in encode_data.groupby("chrom")}
    with Pool(num_cores) as pool:
        results = pool.starmap(process_ccre_rows, [(chunk, encode_by_chrom) for chunk in chunks])
    flat = [x for sub in results for x in sub]
    return pd.DataFrame({"avg_cCRE_full": flat}, index=df.index)

metrics_df = parallel_process_ccre(balanced_data, ENCODE_cCREs, num_processes)
balanced_data["avg_cCRE_full"] = metrics_df["avg_cCRE_full"]
balanced_data.to_csv(output_file, index=False)
print("balanced_data ENCODE cCREs saved")


# DNase 
def _avg_dnase_over_interval(dnase_chr_df: pd.DataFrame, L: int, R: int) -> float:
    if dnase_chr_df.empty:
        return 0.0
    # overlap if (chromStart < R) & (chromEnd > L)
    m = (dnase_chr_df["chromStart"].to_numpy(dtype=np.int64) < int(R)) & \
        (dnase_chr_df["chromEnd"].to_numpy(dtype=np.int64)   > int(L))
    if not np.any(m):
        return 0.0
    s = dnase_chr_df.loc[m, "score"].to_numpy(dtype=float)
    if s.size == 0:
        return 0.0
    v = np.nanmean(s)
    return float(0.0 if np.isnan(v) else v)

def process_chunk_dnase(chunk, dnase_by_chrom):
    results = []
    for _, row in chunk.iterrows():
        chrom = str(row["chrom"])
        dnase_chr = dnase_by_chrom.get(chrom, pd.DataFrame())

        L1, R1 = int(row["left_start"]),  int(row["left_end"])
        L2, R2 = int(row["right_start"]), int(row["right_end"])

        a1 = _avg_dnase_over_interval(dnase_chr, L1, R1)
        a2 = _avg_dnase_over_interval(dnase_chr, L2, R2)

        results.append((a1 + a2) / 2.0)
    return results

def parallel_process_dnase(df, dnase_data, num_cores):
    chunks = np.array_split(df, num_cores)
    dnase_by_chrom = {chrom: g for chrom, g in dnase_data.groupby("chrom")}
    with Pool(num_cores) as pool:
        results = pool.starmap(process_chunk_dnase, [(c, dnase_by_chrom) for c in chunks])
    flat = [x for sub in results for x in sub]
    return pd.DataFrame({"avg_DNase_score_full": flat}, index=df.index)

metrics_df = parallel_process_dnase(balanced_data, DNase, num_processes)
balanced_data["avg_DNase_score_full"] = metrics_df["avg_DNase_score_full"]
balanced_data.to_csv(output_file, index=False)
print("balanced_data dnase saved")


# TF peaks
def _avg_tf_over_interval(tf_chr_df: pd.DataFrame, L: int, R: int) -> float:
    if tf_chr_df.empty:
        return 0.0
    # overlap if (chromStart < R) & (chromEnd > L)
    m = (tf_chr_df["chromStart"].to_numpy(dtype=np.int64) < int(R)) & \
        (tf_chr_df["chromEnd"].to_numpy(dtype=np.int64)   > int(L))
    if not np.any(m):
        return 0.0
    s = tf_chr_df.loc[m, "score"].to_numpy(dtype=float)
    if s.size == 0:
        return 0.0
    v = np.nanmean(s)
    return float(0.0 if np.isnan(v) else v)

def process_chunk_tf(chunk, tf_data_by_chrom):
    results = []
    for _, row in chunk.iterrows():
        chrom = str(row["chrom"])
        tf_chr = tf_data_by_chrom.get(chrom, pd.DataFrame())

        L1, R1 = int(row["left_start"]),  int(row["left_end"])
        L2, R2 = int(row["right_start"]), int(row["right_end"])

        a1 = _avg_tf_over_interval(tf_chr, L1, R1)
        a2 = _avg_tf_over_interval(tf_chr, L2, R2)

        results.append((a1 + a2) / 2.0)
    return results

def parallel_process_tf(df, tf_data, num_cores):
    chunks = np.array_split(df, num_cores)
    tf_by_chrom = {chrom: g for chrom, g in tf_data.groupby("chrom")}
    with Pool(num_cores) as pool:
        results = pool.starmap(process_chunk_tf, [(c, tf_by_chrom) for c in chunks])
    flat = [x for sub in results for x in sub]
    return pd.DataFrame({"avg_tf_score_full": flat}, index=df.index)

print(f"num_processes for tf: {num_processes}")
metrics_df = parallel_process_tf(balanced_data, tf_peak, num_processes)
balanced_data["avg_tf_score_full"] = metrics_df["avg_tf_score_full"]
balanced_data.to_csv(output_file, index=False)
print("balanced_data tf peaks saved")