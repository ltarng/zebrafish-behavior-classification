import pandas as pd


""" PARAMETER SETTING """
folder_path = "D:/Google Cloud (60747050S)/Research/Trajectory Analysis/"
video_name = "1-14"
filter_name = "mean"

# Read resource file
resource_path = folder_path + "cleaned_data/"
df = pd.read_csv(resource_path + video_name + "_" + filter_name + '_filtered.csv')

anno_resource_path = folder_path + "annotation_information/"
anno_info_file_path = anno_resource_path + video_name + '_annotation_information.csv'
anno_df = pd.read_csv(anno_info_file_path)

# Set default value in a new clolumn "Behavior"
df['Behavior'] = 0

for index in range(0, len(anno_df.index)):
    # Get annotation information from annotation file
    behavior = anno_df['BehaviorType'].iloc[index]
    start_frame, end_frame = anno_df['StartFrame'].iloc[index], anno_df['EndFrame'].iloc[index]

    # Make behavior dictionary
    behavior_dict = {'lost': -1, 'normal': 1, 'display': 2, 'circle': 3, 'chase': 4, 'bite': 5}
    
    print(behavior in behavior_dict.keys())

    # Annotate a behavior
    # df['Behavior'].iloc[start_frame:end_frame] = behavior_dict[behavior]
