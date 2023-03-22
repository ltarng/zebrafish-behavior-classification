import pandas as pd


def combine_cleaned_csv_files(folder_path, video_names, filter_name):
    # Append the CSV files
    resource_file_path = folder_path + "cleaned_data/"
    df1 = pd.read_csv(resource_file_path + video_names[0] + "_" + filter_name + "_filtered.csv")
    df2 = pd.read_csv(resource_file_path + video_names[1] + "_" + filter_name + "_filtered.csv")

    # Add a new comlumn in two original data, and save them in a temp dataframe
    temp_df_1 = df1.copy()
    temp_df_1['VideoName'] = video_names[0]
    temp_df_2 = df2.copy()
    temp_df_2['VideoName'] = video_names[1]

    # Combine two file
    combined_df = pd.concat([temp_df_1, temp_df_2], ignore_index=True)
    print(combined_df)

    # Change 'VideoName' from the last column to first column
    cols = combined_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]  
    combined_df = combined_df[cols]
    print(combined_df)

    # Save the combine data in other folder
    save_folder = folder_path + "cleaned_data/"
    combined_df.to_csv(save_folder + "Combined" + "_" + filter_name + "_filtered.csv", index = False)
    print("Successfully combined two csv file.")


def combine_annotation_files(folder_path, video_names):
    # Append the CSV files
    resource_file_path = folder_path + "annotation_information_data/"
    df1 = pd.read_csv(resource_file_path + video_names[0] + "_annotation_information.csv")
    df2 = pd.read_csv(resource_file_path + video_names[1] + "_annotation_information.csv")

    # Add a new comlumn in two original data, and save them in a temp dataframe
    temp_df_1 = df1.copy()
    temp_df_1['VideoName'] = video_names[0]
    temp_df_2 = df2.copy()
    temp_df_2['VideoName'] = video_names[1]

    # Combine two file
    combined_df = pd.concat([temp_df_1, temp_df_2], ignore_index=True)
    print(combined_df)

    # Change 'VideoName' from the last column to first column
    cols = combined_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]  
    combined_df = combined_df[cols]
    print(combined_df)

    # Save the combine data in other folder
    combined_df.to_csv(resource_file_path + "Combined" + "_annotation_information.csv", index = False)
    print("Successfully combined two csv file.")