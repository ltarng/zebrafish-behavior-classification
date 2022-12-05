import pandas as pd


def auto_annotation(folder_path, video_name, filter_name):
    # Read annotation information file
    anno_resource_folder = folder_path + "annotation_information/"
    anno_info_file_path = anno_resource_folder + video_name + '_annotation_information.csv'
    anno_df = pd.read_csv(anno_info_file_path)

    # Read cleaned trajectory file
    resource_path = folder_path + "cleaned_data/"
    df = pd.read_csv(resource_path + video_name + "_" + filter_name + '_filtered.csv')

    # Set default value in a new clolumn "Behavior"
    df['Behavior'] = 0

    # Automatic annotate the trajectory data
    for index in range(0, len(anno_df.index)):
        behavior_type = anno_df['BehaviorType'].iloc[index]
        start_frame, end_frame = anno_df['StartFrame'].iloc[index], anno_df['EndFrame'].iloc[index]

        if behavior_type == 'lost':
            df['Behavior'].iloc[start_frame:end_frame] = -1
        elif behavior_type == 'normal':
            df['Behavior'].iloc[start_frame:end_frame] = 1
        elif behavior_type == 'display':
            df['Behavior'].iloc[start_frame:end_frame] = 2
        elif behavior_type == 'circle':
            df['Behavior'].iloc[start_frame:end_frame] = 3
        elif behavior_type == 'chase':
            df['Behavior'].iloc[start_frame:end_frame] = 4
        elif behavior_type == 'bite':
            df['Behavior'].iloc[start_frame:end_frame] = 5
        else:
            df['Behavior'].iloc[start_frame:end_frame] = 100

    # Save the annoatated data
    df.to_csv(video_name + "_" + filter_name + "_filtered" + "_annotated.csv", index = False)
    print("Complete annotation.")


def add_behavior_numbertag(df):
    # Set default value for new column
    df['BehaviorNum'] = 0
    # Add behavior number tag line by line
    for index in range(0, len(df.index)):
        behavior = df["BehaviorType"].iloc[index]
        if behavior == 'normal':
            df['BehaviorNum'].iloc[index] = 1
        elif behavior == 'display':
            df['BehaviorNum'].iloc[index] = 2
        elif behavior == 'circle':
            df['BehaviorNum'].iloc[index] = 3
        elif behavior == 'chase':
            df['BehaviorNum'].iloc[index] = 4
        elif behavior == 'bite':
            df['BehaviorNum'].iloc[index] = 5
        else:
            df['BehaviorNum'].iloc[index] = 100


def sort_annotation_information(folder_path, video_name):
    # Reading annotation information file
    resource_path = folder_path + "annotation_information/"
    file_path = resource_path + video_name + '_annotation_information.csv'
    anno_df = pd.read_csv(file_path)

    # Remove rows which Behavior is 'lost'
    anno_df = anno_df[anno_df.BehaviorType != "lost"]

    # Add behavior number tag in a new column
    add_behavior_numbertag(anno_df)

    # Sort annoatation information file by
    anno_df.sort_values(by=["BehaviorType"], inplace=True)
    anno_df.to_csv(resource_path + video_name + "_annotation_information_sorted.csv", index = False)
    print("The file had been sorted.")
    print("The file had been saved in: " + resource_path)
    print("\n")
