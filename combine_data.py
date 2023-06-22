import pandas as pd


def combine_preprocessed_files(folder_path, video_names, filter_name, filename_tail):
    resource_folder = folder_path + "preprocessed_data/"
    combined_df = pd.DataFrame()

    # Combine files and save the result in "combined_df"
    for index in range(0, len(video_names)):
        df = pd.read_csv(resource_folder + video_names[index] + "_" + filter_name + filename_tail)

        # Create and add the new comlumn 'VideoName' to resource dataframe, save the result in temp_df
        temp_df = df.copy()
        temp_df['VideoName'] = video_names[index]

        # Combine combine_df and temp_df, and update(override) the result in combined_df
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

    # Change the columns order, move 'VideoName' from the last to the first
    cols = combined_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]  
    combined_df = combined_df[cols]
    print(combined_df)

    # Save the combined data in main folder
    combined_df.to_csv(folder_path + "Combined" + "_" + filter_name + filename_tail, index = False)
    print("Successfully combined two csv files.")


def combine_preprocessed_main(folder_path, video_names, filter_name):
    combine_preprocessed_files(folder_path, video_names, filter_name, "_preprocessed_result.csv")
    combine_preprocessed_files(folder_path, video_names, filter_name, "_preprocessed_result_std.csv")
    combine_preprocessed_files(folder_path, video_names, filter_name, "_preprocessed_result_nor.csv")
