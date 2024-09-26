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


def create_saving_folder_for_confusion_matrix(folder_path):
    save_folder = os.path.join(folder_path, "confusion_matrix_picture")
    ensure_directory_exists(save_folder)
    return save_folder


def prepare_training_dataframe(folder_path: str, video_name: str, filter_name: str, class_amount: int) -> pd.DataFrame:
    file_path = f"{folder_path}combined_preprocessed_data/{video_name}_{filter_name}_preprocessed_result"

    if class_amount == 4:
        df = pd.read_csv(f"{file_path}.csv")
    elif class_amount == 3:
        df = pd.read_csv(f"{file_path}_half_bite_chase.csv")

        # Renumber behavior types for 3-class classification
        # Original indices: bite (0), chase (1), display (2), normal (3)
        # Renumbered to: high aggression (0), moderate aggression (1), low aggression (2)
        df['BehaviorType'] = df['BehaviorType'].replace({0: 0, 1: 0, 2: 1, 3: 2})
    elif class_amount == 2:
        # df = pd.read_csv(f"{file_path}_2cat.csv")  # Composition: 20 bite, 20 chase, 20 display, 60 normal
        df = pd.read_csv(f"{file_path}_2cat_BCN.csv")  # Composition: 30 bite, 30 chase, 60 normal

        # Renumber behavior types for 2-class classification
        # Original indices: bite (0), chase (1), display (2), normal (3)
        # Renumbered to: abnormal (0), normal (1)
        df['BehaviorType'] = df['BehaviorType'].replace({0: 0, 1: 0, 2: 0, 3: 1})
    else:
        raise ValueError("Invalid class_amount provided.")
    return df
