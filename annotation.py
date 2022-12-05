import pandas as pd


def auto_annotation(folder_path, video_name, filter_name):
    # Read annotation information file
    anno_resource_folder = folder_path + "annotation_information/"
    anno_info_file_path = anno_resource_folder + video_name + "_annotation_information.csv"
    anno_df = pd.read_csv(anno_info_file_path)

    # Read cleaned trajectory file
    resource_path = folder_path + "cleaned_data/"
    df = pd.read_csv(resource_path + video_name + "_" + filter_name + "_filtered.csv")

    # Set default value in new clolumns "Behavior"
    df['Behavior'] = 0

    # Automatic annotate the trajectory data
    for index in range(0, len(anno_df.index)):
        behavior_type = anno_df['BehaviorType'].iloc[index]
        start_frame, end_frame = anno_df['StartFrame'].iloc[index], anno_df['EndFrame'].iloc[index]

        # Make behavior dictionary
        behavior_dict = {'lost': -1, 'normal': 1, 'display': 2, 'circle': 3, 'chase': 4, 'bite': 5}
        
        # Annotate a behavior
        if behavior_type in behavior_dict.keys():
            df['Behavior'].iloc[start_frame:end_frame] = behavior_dict[behavior_type]
        else:
            df['Behavior'].iloc[start_frame:end_frame] = 100

    # Save the annoatated data
    df.to_csv(video_name + "_" + filter_name + "_filtered" + "_annotated.csv", index = False)
    print("Complete annotation.\n")


def add_behavior_tag_number(df):
    # Set default value in a new clolumn "BehaviorNum"
    df['BehaviorNum'] = 0
    temp_df = df['BehaviorNum'].copy()  # use a temporary variable to prevent SettingWithCopyWarning problem

    # Automatic annotate the trajectory data
    for index in range(0, len(df.index)):
        behavior_type = df['BehaviorType'].iloc[index]

        if behavior_type == 'normal':
            temp_df.iloc[index] = 1
        elif behavior_type == 'display':
            temp_df.iloc[index] = 2
        elif behavior_type == 'circle':
            temp_df.iloc[index] = 3
        elif behavior_type == 'chase':
            temp_df.iloc[index] = 4
        elif behavior_type == 'bite':
            temp_df.iloc[index] = 5
        else:
            temp_df.iloc[index] = 100
    # Remeber to save the result from the temporary variable
    df['BehaviorNum'] = temp_df.copy()


# def add_behavior_tag_number(df):  # short version?
#     # Set default value in a new clolumn "BehaviorNum"
#     df['BehaviorNum'] = 0
#     temp_df = df['BehaviorNum'].copy()  # use a temporary variable to prevent  SettingWithCopyWarning  problem

#     # Automatic annotate the trajectory data
#     for index in range(0, len(df.index)):
#         behavior_type = df['BehaviorType'].iloc[index]

#         # Make behavior dictionary
#         behavior_dict = {'normal': 1, 'display': 2, 'circle': 3, 'chase': 4, 'bite': 5}
        
#         # Annotate a behavior
#         if behavior_type in behavior_dict.keys():
#             temp_df.iloc[index] = behavior_dict[behavior_type]
#         else:
#             temp_df.iloc[index] = 100

#     # Remeber to save the result from the temporary variable
#     df['BehaviorNum'] = temp_df.copy()


def sort_annotation_information(folder_path, video_name):
    # Reading annotation information file
    resource_path = folder_path + "annotation_information/"
    file_path = resource_path + video_name + "_annotation_information.csv"
    anno_df = pd.read_csv(file_path)

    # Remove rows which Behavior is 'lost'
    anno_df = anno_df[anno_df.BehaviorType != "lost"]

    # Add behavior tag number
    add_behavior_tag_number(anno_df)

    # Sort annoatation information file by
    anno_df.sort_values(by=["BehaviorType"], inplace=True)
    anno_df.to_csv(resource_path + video_name + "_annotation_information_sorted.csv", index = False)
    print("Complete sorting.")
    print("The file had been saved in: " + resource_path + "\n")
