import os
import pandas as pd
import torch
from sklearn.decomposition import PCA
from glob import glob

def merge_datasets_with_tabula():
    feature_path = "./baml/ADBench/adbench/image_result/"
    sample_path = os.path.join(feature_path, "sample_level_cluster")
    output_path = os.path.join(feature_path, "merged")
    os.makedirs(output_path, exist_ok=True)

    feature_files = glob(os.path.join(feature_path, "dataset_feature_*_seed1.csv"))

    for feature_file in feature_files:
        dataset_name = os.path.basename(feature_file).replace("dataset_feature_", "").replace("_seed1.csv", "")
        sample_file = os.path.join(sample_path, f"{dataset_name}_sample_scores.csv")

        if not os.path.exists(sample_file):
            print(f"No sample files found.：{sample_file}")
            continue

        df_feat = pd.read_csv(feature_file)
        df_sample = pd.read_csv(sample_file)

        df_feat["sample_id"] = [f"{dataset_name}_{i}" for i in range(len(df_feat))]
        df_sample["sample_id"] = [f"{dataset_name}_{i}" for i in range(len(df_sample))]

        try:
            df_merged = pd.merge(df_feat, df_sample, on="sample_id", how="inner")
            df_merged["dataset"] = dataset_name

            out_path = os.path.join(output_path, f"merged_{dataset_name}.csv")
            df_merged.to_csv(out_path, index=False)
            print(f"Merge and save: {dataset_name}, sample count = {len(df_merged)}, file: {out_path}")

        except Exception as e:
            print(f"Merge failed: {dataset_name}, error message: {e}")

if __name__ == "__main__":
    merge_datasets_with_tabula()