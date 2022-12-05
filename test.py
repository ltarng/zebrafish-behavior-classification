import pandas as pd


""" GENERAL PARAMETER SETTING """
folder_path = "D:/Google Cloud (60747050S)/Research/Trajectory Analysis/"
video_name = '1-14'
filter_name = "mean"  # filter option: mean, median, kalman


# Read annotation information file
anno_resource_folder = folder_path + "annotation_information/"
anno_info_file_path = anno_resource_folder + video_name + "_annotation_information.csv"
anno_df = pd.read_csv(anno_info_file_path)


# Set default value in new clolumns "BehaviorNum"
df = anno_df
df['BehaviorNum'] = 0

# temp
temp_df = df['BehaviorNum'].copy()

# Automatic annotate the trajectory data
for index in range(0, len(anno_df.index)):
    behavior_type = df['BehaviorType'].iloc[index]

    # Make behavior dictionary
    behavior_dict = {'lost': -1, 'normal': 1, 'display': 2, 'circle': 3, 'chase': 4, 'bite': 5}


    # Annotate a behavior
    if behavior_type in behavior_dict.keys():
        temp_df.iloc[index] = behavior_dict[behavior_type]
    else:
        temp_df.iloc[index] = 100

df['BehaviorNum'] = temp_df

# Save the annoatated data
df.to_csv(video_name + "_" + filter_name + "_filtered" + "_annotated.csv", index = False)
print("Complete annotation.\n")
