import pandas as pd

df = pd.read_csv("results.csv")

filtered_rows = []

for dataset_name, group in df.groupby('dataset_config_name'):
    keep_indices = []
    for i, row_i in group.iterrows():
        dominated = False
        for j, row_j in group.iterrows():
            if (
                row_j['specificity'] > row_i['specificity']
                and row_j['sensitivity'] > row_i['sensitivity']
            ):
                dominated = True
                break
        if not dominated:
            keep_indices.append(i)
    filtered_rows.append(df.loc[keep_indices])

filtered_df = pd.concat(filtered_rows, ignore_index=True)

filtered_df.to_csv("results_filtered.csv", index=False)
