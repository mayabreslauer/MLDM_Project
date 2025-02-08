import pandas as pd
file_name = 'SB-Full Feature Extracted RSP ECG.csv'
df = pd.read_csv(file_name)

def fill_missing_ones(group):
    ones_indices = group.index[group['Stress Level'] == 1].tolist()
    if len(ones_indices) < 3:
        last_two_ones_indices = ones_indices[-2:]
        if len(last_two_ones_indices) == 2:
            avg_row_values = group.loc[last_two_ones_indices].mean()
            avg_row_values['Stress Level'] = 1
            new_row_df = pd.DataFrame([avg_row_values], columns=group.columns)
            group = pd.concat([group, new_row_df], ignore_index=True)
    
    return group

df = df.groupby('Subject').apply(fill_missing_ones).reset_index(drop=True)

output_file_name = 'SB-Full Feature Processed.csv'
df.to_csv(output_file_name, index=False)
processed_file_name = 'SB-Full Feature Processed.csv'
df_processed = pd.read_csv(processed_file_name)
subject_row_counts = df_processed.groupby('Subject').size()
subjects_with_incorrect_rows = subject_row_counts[subject_row_counts != 19]

if subjects_with_incorrect_rows.empty:
    print("All subjects have 19 rows.")
else:
    print("Subjects with incorrect number of rows:")
    print(subjects_with_incorrect_rows)
df = pd.read_csv("SB-Full Feature Processed.csv")

new_rows = []

for i in range(0, len(df), 19):
    chunk = df.iloc[i:i+19]
    for j in range(0, len(chunk)-3, 4):
        weighted_avg = (chunk.iloc[j:j+4].mul([1/4, 1/3, 1/2, 1], axis=0).sum()) / (1/4 + 1/3 + 1/2 + 1)
        new_rows.append(weighted_avg)
    regular_avg = chunk.tail(3).mean()
    new_rows.append(regular_avg)
new_df = pd.DataFrame(new_rows, columns=df.columns)
new_df.to_csv("processed_data.csv", index=False)
