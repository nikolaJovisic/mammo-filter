import glob
import csv

output_file = 'results.csv'
input_files = sorted(glob.glob('results_gpu*.csv'))

with open(output_file, mode='w', newline='') as out_f:
    writer = None
    for i, file in enumerate(input_files):
        with open(file, mode='r', newline='') as in_f:
            reader = csv.reader(in_f)
            header = next(reader)
            if i == 0:
                writer = csv.writer(out_f)
                writer.writerow(header)
            for row in reader:
                writer.writerow(row)

print(f"Merged {len(input_files)} files into {output_file}")
