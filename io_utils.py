import pandas as pd
import os

def read_csv(file_path):
    return pd.read_csv(file_path)

def save_csv(df, file_path):
    df.to_csv(file_path, index=False)

def ensure_directory_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def save_results(df, folder_path, video_name, filter_name, suffix):
    save_folder = folder_path + "preprocessed_data/"
    ensure_directory_exists(save_folder)

    save_csv(df, save_folder + video_name + "_" + filter_name + suffix)
    
    print(f"Complete {suffix.replace('_', ' ')}. The file has been saved in: {folder_path}\n")
