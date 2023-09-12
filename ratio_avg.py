

import os
import csv

cwd = os.getcwd()
files = sorted([f for f in os.listdir(cwd) if f.endswith('.csv')])

for file in files:
    with open(file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        col2_sum = 0
        col2_count = 0
        for row in reader:
            col2_sum += float(row[1])
            col2_count += 1
        col2_mean = col2_sum / col2_count
        print(f"{file}\t{col2_mean}")
