import pandas as pd

def read_csv_rows_pandas(file_path, start_line, end_line, header_row_present=True):
    df = pd.read_csv(file_path, header=0 if header_row_present else None)
    return df.iloc[start_line - 1: end_line]

def get_kmers_numeric(sequence, k):
    if len(sequence) < k:
        return []
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

def levenshtein_distance_numeric(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("Последовательности должны быть одинаковой длины")
    return sum(a != b for a, b in zip(seq1, seq2))

# === Читаем данные ===
df = read_csv_rows_pandas("cccna_data.csv", 3001, 4001)

NUMERIC_START_COL_INDEX = 2  # числа начинаются с 3-го столбца

seq1_numeric = df.iloc[0].values[NUMERIC_START_COL_INDEX:].astype(int).tolist()
print(f"\nПоследовательность из пункта 1.1 (длина {len(seq1_numeric)}): {seq1_numeric[:20]}...")

third_numeric = df.iloc[2].values[NUMERIC_START_COL_INDEX:].astype(int).tolist()
print(f"Третья строка варианта (длина {len(third_numeric)}): {third_numeric[:20]}...")

# === Параметры ===
k = 3
mer_length = k % 6 + 4  # = 7

kmers_step1 = get_kmers_numeric(seq1_numeric, mer_length)
print(f"\nНайдено {len(kmers_step1)} семимеров в строке")

if len(kmers_step1) == 0:
    raise ValueError("Слишком короткая последовательность — не удалось извлечь 7-меры!")

# === Выбираем два 7-мера (например, первый и второй) ===
selected_kmers = [kmers_step1[0], kmers_step1[5]] 
print("\nВыбранные 7-меры:")
for i, kmer in enumerate(selected_kmers):
    print(f"{i+1}: {kmer}")

# === Получаем все 7-меры из третьей строки ===
windows_third = get_kmers_numeric(third_numeric, mer_length)
print(f"\nНайдено {len(windows_third)} семимеров в третьей строке.")

results = {}
for kmer in selected_kmers:
    matches = []
    for window in windows_third:
        if levenshtein_distance_numeric(kmer, window) == 1:
            matches.append(window)
    results[tuple(kmer)] = matches


found_any = False
for kmer, matches in results.items():
    if matches:
        found_any = True
        print(f"\nДля 7-мера {list(kmer)} найдены совпадения (расстояние = 1):")
        for m in matches:
            print(f"   {m}")
    else:
        print(f"\nДля 7-мера {list(kmer)} не найдено подстрок с расстоянием 1.")

if not found_any:
    print("\n❗Ни для одного выбранного 7-мера не найдено подходящих подстрок в третьей строке.")