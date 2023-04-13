import pandas as pd
import os


def auto_annotate(folder_path, video_name, filter_name):
    # Read annotation information data
    anno_resource_folder = folder_path + "annotation_information_data/"
    anno_info_file_path = anno_resource_folder + video_name + "_annotation_information.csv"
    anno_df = pd.read_csv(anno_info_file_path)

    # Read cleaned trajectory file
    resource_path = folder_path + "cleaned_data/"
    df = pd.read_csv(resource_path + video_name + "_" + filter_name + "_filtered.csv")

    # Set default value in new clolumns "Behavior"
    df['BehaviorType'] = 0

    # Automatic annotate the trajectory data
    for index in range(0, len(anno_df.index)):
        behavior_type = anno_df['BehaviorName'].iloc[index]
        start_frame, end_frame = anno_df['StartFrame'].iloc[index], anno_df['EndFrame'].iloc[index]

        # Make behavior dictionary
        behavior_dict = {'normal': 1, 'display': 2, 'circle': 3, 'chase': 4, 'bite': 5}
        
        # Annotate a behavior
        if behavior_type in behavior_dict.keys():
            df['BehaviorType'].iloc[start_frame:end_frame] = behavior_dict[behavior_type]
        else:
            df['BehaviorType'].iloc[start_frame:end_frame] = 100
            print("Wrong behavior name!" + df['BehaviorType'].iloc[start_frame:end_frame])

    # Setting save file path. If folder is not exist, create a new one
    save_folder = folder_path + "annotated_data/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save the annoatated data
    df.to_csv(save_folder + video_name + "_" + filter_name + "_filtered" + "_annotated.csv", index = False)
    print("Complete annotation.\n")


def sort_annotation_information(folder_path, video_name):
    # Reading annotation information file
    anno_resource_folder = folder_path + "annotation_information_data/"
    anno_info_file_path = anno_resource_folder + video_name + "_annotation_information.csv"
    anno_df = pd.read_csv(anno_info_file_path)

    # Remove rows which Behavior is 'circle'
    anno_df = anno_df[anno_df.BehaviorName != "circle"]

    # Set default value in new clolumns "Behavior"
    anno_df['BehaviorType'] = 0

    # Convert behavior type from text to integer
    for index in range(0, len(anno_df.index)):
        behavior_name = anno_df['BehaviorName'].iloc[index]

        # Make behavior dictionary
        behavior_dict = {'normal': 1, 'display': 2, 'circle': 3, 'chase': 4, 'bite': 5}
        
        # Annotate a behavior
        if behavior_name in behavior_dict.keys():
            anno_df['BehaviorType'].iloc[index] = behavior_dict[behavior_name]
            # print(behavior_name, behavior_dict[behavior_name])
        else:
            anno_df['BehaviorType'].iloc[index] = 100
            print("Wrong behavior name!" + anno_df['BehaviorType'].iloc[index])

    # Sort annoatation information file by
    anno_df.sort_values(by=["BehaviorType"], inplace=True)
    anno_df.to_csv(anno_resource_folder + video_name + "_annotation_information_sorted.csv", index = False)
    print("Complete sorting. The file had been saved in: " + anno_resource_folder + "\n")