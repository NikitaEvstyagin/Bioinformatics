import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv_rows_pandas(file_path, start_line, end_line, header_row_present=True):
    df = pd.read_csv(file_path, header=0 if header_row_present else None)
    return df.iloc[start_line - 1: end_line]

# Функция вычисления расстояния Левенштейна
def levenshtein_distance(s1, s2):
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )

    return dp[len(s1)][len(s2)]

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
#1.2.3
def score_motif(consensus, motifs):
    t = len(consensus)
    count_matrix = np.zeros((t, 4), dtype=int)

    for motif in motifs:
        for i, letter in enumerate(motif):
            count_matrix[i, letter - 1] += 1

    for i, letter in enumerate(consensus):
        count_matrix[i, letter - 1] += 1
    
    score = sum(count_matrix[i, letter-1] for i, letter in enumerate(consensus))
    return score
    
df = read_csv_rows_pandas("cccna_data.csv", 1000*(3-1)+1, 3000)
d = 3%6+4 #7
kmers = [
    [2,2,2,2,2,2,2],
    [2,3,2,3,2,3,2],
    [3,2,3,2,3,2,3],
    [1,1,1,1,1,1,1],
    [1,3,1,3,3,3,4],
    [2,4,2,4,2,4,2],
]

v_columns = [c for c in df.columns if c.startswith("V")]
sequences = df[v_columns].astype(int).to_numpy()
all_counts = np.array([kmers_convolution(seq, d) for seq in sequences])
total_counts = all_counts.sum(axis=0) 
unique_kmers = [index_to_kmer_numeric(i, 7) for i, c in enumerate(total_counts) if c > 0]

distances = {}
for query in kmers:
    distances[tuple(query)] = {
        tuple(uk): levenshtein_distance(query, uk) for uk in unique_kmers
    }

for kmer in kmers:
    query = tuple(kmer)
    sorted_dists = sorted(distances[query].items(), key=lambda x: x[1])[:5]
    
    print(f"Ближайшие 5 k-меров для {query}:")
    for _, dist in sorted_dists:
        print(f"{kmer} -> расстояние {dist}")
print("="*120)  

#1.1.2
motifs_distance_1 = {}
for kmer in kmers:
    kmer_tuple = tuple(kmer)
    motifs = [uk for uk, dist in distances[kmer_tuple].items() if dist == 1]
    motifs_distance_1[kmer_tuple] = motifs
    
for kmer, motifs in motifs_distance_1.items():
    print(f"\nМотивы на расстоянии 1 от {kmer}:")
    for motif in motifs:
        print(motif)
    print(len(motifs))
        
print("="*120)

for kmer, motifs in motifs_distance_1.items():
    total_distance = score_motif(kmer, motifs)
    print(f"Консенсус: {kmer}")
    print(f"  Оценочная функция (сумма Lev): {total_distance}")