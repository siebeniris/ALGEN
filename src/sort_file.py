import pandas as pd
import numpy as np
import os



def get_dim_size(x, col_shift):
    return -x if col_shift=="right" else x

folder = "results/normal_equations/wordvectors"
for file in os.listdir(folder):

    filepath = os.path.join(folder, file)
    if filepath.endswith("_samples_6000.csv"):
        source, target = file.replace("_samples_6000.csv", "").split("_")
        total_dim = None
        if source == "word2vec":
            total_dim = 300
        elif source == "glove100":
            total_dim = 100
        elif source == "glove300":
            total_dim = 300


        def get_dim_size(x, col_shift):
            return total_dim - x if col_shift == "right" else x

        print(f"working on {filepath}")
        df = pd.read_csv(filepath)
        df['dim_size'] = df.apply(lambda row: get_dim_size(row['dim'], row['shift']), axis=1)
        df.sort_values(by=["shift","dim_size"], inplace=True)
        df=df[["shift", "dim_size", "test_acc", "test_cossim", "X_Y_cossim"]]
        df.to_csv(f"{filepath.replace(".csv", "")}_sorted.csv",index=False)
