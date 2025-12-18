# File ini digunakan untuk mengubah format dataset agar bisa dipakai untuk TrOCR sebagai training

import csv
import os

input_file = r"Datasets/IAM_Words/words.txt"
output_file = r"Datasets_TrOCR/IAM_Words/dataset_TrOCR.csv"

def build_image_path(word_id):
    parts = word_id.split('-')
    folder1 = parts[0]
    folder2 = parts[0] + "-" + parts[1]
    return f"words\\{folder1}\\{folder2}\\{word_id}.png"

rows = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        word_id = parts[0]
        transcription = parts[-1]

        if transcription == ",":
            transcription = "\",\""[1:-1]

        image_path = build_image_path(word_id)

        rows.append([image_path, transcription])

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "text"])
    writer.writerows(rows)

print("File CSV tersimpan di:", output_file)
