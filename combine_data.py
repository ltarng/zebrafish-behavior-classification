import pandas as pd


def combine_preprocessed_files(folder_path, video_names, filter_name):
    combined_df = pd.DataFrame()
    for index in range(0, len(video_names)):
        df = pd.read_csv(folder_path + video_names[index] + "_" + filter_name + "_preprocessed_result.csv")

        # Add a new comlumn 'VideoName' in original data, and save the result in a temp dataframe
        temp_df = df.copy()
        temp_df['VideoName'] = video_names[index]

        # Combine the file into a combined file
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

    # Change 'VideoName' from the last column to first column
    cols = combined_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]  
    combined_df = combined_df[cols]
    print(combined_df)

    # Save the combine data in other folder
    combined_df.to_csv(folder_path + "Combined" + "_" + filter_name + "_preprocessed_result.csv", index = False)
    print("Successfully combined two csv files.")
