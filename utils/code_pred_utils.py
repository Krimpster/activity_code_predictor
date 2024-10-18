import pandas as pd
from tqdm import tqdm

def get_top_n_predictions(data, n = 5):

    top_n_predictions = data.apply(func=lambda row: row.nlargest(n))

    top_n_results = [f"Result {i+1}: {top_n_predictions.columns[i]}" for i in range(n)]

    return top_n_results

def accuracy_calculator(data):

    total_true = 0
    total_false = 0

    for _,row in tqdm(data.iterrows()):
        if row['KPB AKTIVITETSKOD'] == row['kpb_activity_code']:
            total_true += 1
        if row['KPB AKTIVITETSKOD'] != row['kpb_activity_code']:
            total_false += 1

    return total_true, total_false, float(total_true/(total_false + total_true))