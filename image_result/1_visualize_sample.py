import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils.multiclass import type_of_target

# Automatically determine the type and calculate mutual information (for y_true and model scores)
def compute_mutual_info(df, feature_cols, target_cols):
    mi_results = {}
    for target in target_cols:
        target_type = type_of_target(df[target])
        try:
            if target_type in ["binary", "multiclass"]:
                mi = mutual_info_classif(df[feature_cols], df[target], discrete_features=False)
            else:
                mi = mutual_info_regression(df[feature_cols], df[target])
            mi_results[target] = mi
        except Exception as e:
            print(f"Skip {target} due to exception: {e}")
    mi_df = pd.DataFrame(mi_results, index=feature_cols)
    return mi_df

# Calculate the absolute value of Pearson correlation
def compute_pearson_abs(df, feature_cols, target_cols):
    corr = df[feature_cols + target_cols].corr(method='pearson')
    return corr.loc[feature_cols, target_cols].abs()

# drawing function
def plot_heatmap(data, title):
    rename_map = {
                'y_score_odin': 'ODIN',
                'y_score_gradnorm': 'GradNorm',
                'y_score_icl': 'ICL'
                }
    data = data.rename(columns=rename_map)
    # Delete the y_true column
    data = data.drop(columns=[col for col in data.columns if col.startswith("y_true")], errors="ignore")

    # Identify and average NUDFT/PCA
    nudft_rows = [r for r in data.index if r.startswith("NUDFT_")]
    pca_rows = [r for r in data.index if r.startswith("PCA_")]

    if nudft_rows:
        nudft_avg = data.loc[nudft_rows].mean().to_frame().T
        nudft_avg.index = ["Average_NUDFT"]
        data = pd.concat([nudft_avg, data])

    if pca_rows:
        pca_avg = data.loc[pca_rows].mean().to_frame().T
        pca_avg.index = ["Average_PCA"]
        data = pd.concat([pca_avg, data])
    
    # Save data as CSV
    csv_dir = r"./baml/ADBench/adbench/csv_result"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{title.replace(' ', '_')}.csv")
    data.to_csv(csv_path)

    plt.figure(figsize=(12, 6))
    sns.heatmap(data, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
    # plt.title(title)
    plt.tight_layout()

    save_dir = r"./baml/ADBench/adbench/image_result"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Main function encapsulation
def main_visualize_sample_level(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()

    feature_cols = [col for col in df.columns if col.startswith("PCA_") or col.startswith("NUDFT_")]
    target_cols = [
                    col for col in df.columns 
                    if col not in feature_cols 
                    and col not in ['dataset', 'sample_id']
                    and (pd.api.types.is_numeric_dtype(df[col]) or col == 'PCA')
                    ]

    mi_df = compute_mutual_info(df, feature_cols, target_cols)
    plot_heatmap(mi_df, "Mutual Information between Features and Model Outputs / y_true")

    pearson_df = compute_pearson_abs(df, feature_cols, target_cols)
    plot_heatmap(pearson_df, "Absolute Pearson Correlation between Features and Model Outputs / y_true")

def aggregate_and_plot_heatmap(folder_path):
    mi_list = []
    pearson_list = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file_name)).dropna()
            feature_cols = [col for col in df.columns if col.startswith("PCA_") or col.startswith("NUDFT_")]
            target_cols = [
                            col for col in df.columns 
                            if col not in feature_cols 
                            and col not in ['dataset', 'sample_id']
                            and (pd.api.types.is_numeric_dtype(df[col]) or col == 'PCA')
                            ]

            mi_df = compute_mutual_info(df, feature_cols, target_cols)
            pearson_df = compute_pearson_abs(df, feature_cols, target_cols)

            mi_list.append(mi_df)
            pearson_list.append(pearson_df)

    mi_mean = pd.concat(mi_list).groupby(level=0).mean()
    pearson_mean = pd.concat(pearson_list).groupby(level=0).mean()

    plot_heatmap(mi_mean, "local Mutual Information")
    plot_heatmap(pearson_mean, "local Pearson Correlation")

def analyze_top_feature_model_pairs(csv_folder, score_type="Mutual_Information"):
    """
    Analyse the heatmap data (Mutual Information or Pearson) for each anomaly type to identify:
    1. The most relevant features for each model;
    2. The most relevant models for each feature;

    Parameters:
    csv_folder: The folder path where the heatmap CSV files are stored;
    score_type: ‘Mutual_Information’ or ‘Pearson’;
    """
    top_feature_for_model = []
    top_model_for_feature = []

    # 映射文件名 -> 简洁的异常类型标签
    label_map = {
        "cluster Pearson Correlation": "Cluster",
        "dependency Pearson Correlation": "Dependency",
        "global Pearson Correlation": "Global",
        "local Pearson Correlation": "Local",
        "none Pearson Correlation": "None",
        "cluster Mutual Information": "Cluster",
        "dependency Mutual Information": "Dependency",
        "global Mutual Information": "Global",
        "local Mutual Information": "Local",
        "none Mutual Information": "None"
    }


    for file in os.listdir(csv_folder):
        if score_type in file and file.endswith(".csv"):
            # Automatically extract anomaly types (e.g., extract ‘Cluster’ from ‘cluster_Pearson_Correlation.csv’)
            anomaly_type_raw = Path(file).stem.replace(f"_{score_type}", "").split("_")[0].capitalize()
            anomaly_type = label_map.get(anomaly_type_raw, anomaly_type_raw)


            df = pd.read_csv(os.path.join(csv_folder, file), index_col=0)

            # Top features (rows) for each model (column)
            for model in df.columns:
                try:
                    series = pd.to_numeric(df[model], errors='coerce')  # Forced conversion to numeric values, invalid values become NaN
                    if series.isnull().all():
                        continue  # If all values are NaN, skip.
                    best_feat = series.idxmax()
                    best_score = series.max()
                    top_feature_for_model.append({
                        "Anomaly_Type": anomaly_type,
                        "Model": model,
                        "Top_Feature": best_feat,
                        score_type: best_score
                    })
                except Exception as e:
                    print(f"Skip model {model} due to exception: {e}")


            # Top models (columns) for each feature (row)
            # Top models (columns) for each feature (row)
            for feature in df.index:
                try:
                    row = pd.to_numeric(df.loc[feature], errors='coerce')
                    if row.isnull().all():
                        continue
                    best_model = row.idxmax()
                    best_score = row.max()
                    top_model_for_feature.append({
                        "Anomaly_Type": anomaly_type,
                        "Feature": feature,
                        "Top_Model": best_model,
                        score_type: best_score
                    })
                except Exception as e:
                    print(f"Skip feature {feature} due to anomaly: {e}")


    df1 = pd.DataFrame(top_feature_for_model)
    df2 = pd.DataFrame(top_model_for_feature)

    # 保存结果
    df1.to_csv(os.path.join(csv_folder, f"Top_Feature_for_Model_{score_type}.csv"), index=False)
    df2.to_csv(os.path.join(csv_folder, f"Top_Model_for_Feature_{score_type}.csv"), index=False)

    print(f"Analysis complete! Save to：{csv_folder}")
    print(df1.head())
    print(df2.head())
    df1 = df1[df1["Model"] != score_type]  # Clean up unexpected events
    df2 = df2[df2["Top_Model"] != score_type]

def plot_top_feature_per_model_barplot(df, score_col="Mutual_Information", output_dir="./", save_name="Top_Feature_Per_Model.png"):
    """
    Plot a bar chart: the score (MI or Pearson) corresponding to the top feature for each model under different anomaly types.
    
    Parameters:
        df: DataFrame containing the columns [‘Model’, ‘Anomaly_Type’, score_col]
        score_col: ‘Mutual_Information’ or ‘Pearson’
        output_dir: Folder where images are saved
        save_name: File name for saving images
    """
    df = df[df["Model"] != score_col]  # Remove rows where Model == 'Pearson'

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="Model",
        y=score_col,
        hue="Anomaly_Type"
    )
    # plt.title(f"Top Feature per Model across Anomaly Types ({score_col})")
    plt.ylabel(score_col)
    plt.xticks(rotation=45)
    plt.legend(title="Anomaly Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"图已保存：{save_path}")

# 示例调用
if __name__ == "__main__":
    path = "./baml/ADBench/adbench/result/local/local_none_dataset/merged"
    csv_path = r"./baml/ADBench/adbench/csv_result"
    # main_visualize_sample_level(path)
    aggregate_and_plot_heatmap(path)
    analyze_top_feature_model_pairs(csv_path, score_type="Mutual_Information")
    analyze_top_feature_model_pairs(csv_path, score_type="Pearson")
    """  
    df_mi = pd.read_csv(os.path.join(csv_path, "Top_Feature_for_Model_Mutual_Information.csv"))
    plot_top_feature_per_model_barplot(
        df=df_mi,
        score_col="Mutual_Information",
        output_dir="./baml/ADBench/adbench/image_result",
        save_name="Top_Feature_Per_Model_MI.png"
    )
    df_pearson = pd.read_csv(os.path.join(csv_path, "Top_Feature_for_Model_Pearson.csv"))
    plot_top_feature_per_model_barplot(
        df=df_pearson,
        score_col="Pearson",
        output_dir="./baml/ADBench/adbench/image_result",
        save_name="Top_Feature_Per_Model_Pearson.png"
    )
    """