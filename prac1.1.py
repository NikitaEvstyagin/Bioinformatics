import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv_rows_pandas(file_path, start_line, end_line, header_row_present=True):
    df = pd.read_csv(file_path, header=0 if header_row_present else None)
    return df.iloc[start_line - 1: end_line]

#1.1.1
def index_to_kmer_numeric(idx, k):
    return list(int((idx // (4 ** i)) % 4 + 1) for i in reversed(range(k)))

def kmers_convolution(seq, k=7):
    base = 4
    n = len(seq)
    if n < k:
        return np.zeros(base ** k, dtype=int)

    s = seq - 1

    code = 0
    for i in range(k):
        code = code * base + s[i]

    counts = np.zeros(base ** k, dtype=int)
    counts[code] += 1

    # rolling hash
    base_pow = base ** (k - 1)
    for i in range(k, n):
        code = (code - s[i - k] * base_pow) * base + s[i]
        counts[code] += 1

    return counts

#1.1.2
def rabin_karp_search(text, pattern, base=256, prime=101):
    n, m = len(text), len(pattern)
    if m > n:
        return 0
    h = base**(m-1) % prime
    p_hash = t_hash = 0
    for i in range(m):
        p_hash = (p_hash * base + pattern[i]) % prime
        t_hash = (t_hash * base + text[i]) % prime
    count = 0
    for i in range(n - m + 1):
        if p_hash == t_hash:
            if text[i:i+m] == pattern:
                count += 1
        if i < n - m:
            t_hash = (t_hash - text[i] * h) % prime
            t_hash = (t_hash * base + text[i + m]) % prime
            t_hash = (t_hash + prime) % prime
    return count

#1.1.3
def sliding_window_counts(seq, window=50):
    n = len(seq)
    counts = []
    for i in range(n - window + 1):
        window_seq = seq[i:i+window]
        a = np.sum(window_seq == 1)
        t = np.sum(window_seq == 2)
        g = np.sum(window_seq == 3)
        c = np.sum(window_seq == 4)
        counts.append([a, t, g, c])
    return np.array(counts)

def compute_skews(counts):
    A, T, G, C = counts[:,0], counts[:,1], counts[:,2], counts[:,3]
    at_skew = (A - T) / (A + T)
    gc_skew = (G - C) / (G + C)
    return at_skew, gc_skew

def find_skew_crossings(skew):
    return np.where(np.diff(np.sign(skew)) != 0)[0]


df = read_csv_rows_pandas("cccna_data.csv", 1000*(3-1)+1, 3000)
d = 3%6+4 #7
v_columns = [c for c in df.columns if c.startswith("V")]
sequences = df[v_columns].astype(int).to_numpy()
all_counts = np.array([kmers_convolution(seq, d) for seq in sequences])
total_counts = all_counts.sum(axis=0) 
top6_idx = total_counts.argsort()[-6:][::-1]
top6_kmers = [index_to_kmer_numeric(idx, d) for idx in top6_idx]

print("Топ-6 7-меров:", top6_kmers)

results = {}
all_sequences = df[v_columns].values.tolist()
for kmer in top6_kmers:
    total = 0
    for seq in all_sequences:
        total += rabin_karp_search(seq, kmer)
    results[tuple(kmer)] = total
    
print(f"\nЧисло вхождений выбранных {7}-меров во всём датасете:")
for kmer, count in results.items():
    print(f"{kmer}: {count}")


for i, seq in enumerate(sequences[:10]):
    counts = sliding_window_counts(seq, window=100)
    at_skew, gc_skew = compute_skews(counts)
    crossings = find_skew_crossings(at_skew)

    bind_flags = df.iloc[i]['bind']  # True или False
    print(f" {i}, bind={bind_flags}")
    print(f"Потенциальные точки начала репликации (Ori) по AT-skew:", crossings)


