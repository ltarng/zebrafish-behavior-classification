import numpy as np
import pandas as pd
from dtaidistance import dtw_ndim
import math
from progress.bar import IncrementalBar
import os


def calculate_interframe_vector(p0, p1):
    x_shift = p1[0] - p0[0]
    y_shift = p1[1] - p0[1]
    return x_shift, y_shift


def caculate_direction(start_frame, end_frame, traj_df):
    fish0_mean_x = round(traj_df[start_frame:end_frame, 8].mean(), 2)
    fish0_mean_y = round(traj_df[start_frame:end_frame, 9].mean(), 2)
    fish1_mean_x = round(traj_df[start_frame:end_frame, 10].mean(), 2)
    fish1_mean_y = round(traj_df[start_frame:end_frame, 11].mean(), 2)
    return fish0_mean_x, fish0_mean_y, fish1_mean_x, fish1_mean_y


def getPoint(df_fish_x, df_fish_y, index):
    return [df_fish_x.iloc[index], df_fish_y.iloc[index]]


def caculate_interframe_distance(p1, p2):
    return round(math.dist(p1, p2), 2)


def calculate_semifinished_result(folder_path, video_name, filter_name):  # Calculate distance and direction between frames
    # Read trajectory data
    resource_folder = folder_path + "annotated_data/"
    file_path = resource_folder + video_name + "_" + filter_name + "_filtered_annotated.csv"
    df = pd.read_csv(file_path)

    # Create a new column, and set default values with 0
    df['Fish0_interframe_movement_dist'] = 0
    df['Fish1_interframe_movement_dist'] = 0
    df['Fish0_interframe_moving_direction_x'] = 0
    df['Fish0_interframe_moving_direction_y'] = 0
    df["Fish1_interframe_moving_direction_x"] = 0
    df["Fish1_interframe_moving_direction_y"] = 0

    # Use temporary variable to prevent SettingWithCopyWarning problem
    temp_df_fish0_dist = df['Fish0_interframe_movement_dist'].copy()
    temp_df_fish1_dist = df['Fish1_interframe_movement_dist'].copy()
    temp_df_fish0_direction_x = df['Fish0_interframe_moving_direction_x'].copy()
    temp_df_fish0_direction_y = df['Fish0_interframe_moving_direction_y'].copy()
    temp_df_fish1_direction_x = df["Fish1_interframe_moving_direction_x"].copy()
    temp_df_fish1_direction_y = df["Fish1_interframe_moving_direction_y"].copy()

    # Calculate moving distance in the same trajectory interval between two trajectories
    with IncrementalBar(video_name + ' - Progress of Basic Caculation', max=len(df.index)) as bar:  # with a progress bar
        for index in range(0, len(df.index)-1):
            # calculate the distance between: frame n to frame n+1 
            temp_df_fish0_dist.iloc[index] = caculate_interframe_distance(getPoint(df['Fish0_x'], df['Fish0_y'], index), getPoint(df['Fish0_x'], df['Fish0_y'], index+1))
            temp_df_fish1_dist.iloc[index] = caculate_interframe_distance(getPoint(df['Fish1_x'], df['Fish1_y'], index), getPoint(df['Fish1_x'], df['Fish1_y'], index+1))

            # calculate the moving direction from frame n to frame n+1
            temp_df_fish0_direction_x.iloc[index], temp_df_fish0_direction_y.iloc[index] = calculate_interframe_vector(getPoint(df['Fish0_x'], df['Fish0_y'], index), 
                                                                                                                       getPoint(df['Fish0_x'], df['Fish0_y'], index+1))
            temp_df_fish1_direction_x.iloc[index], temp_df_fish1_direction_y.iloc[index] = calculate_interframe_vector(getPoint(df['Fish1_x'], df['Fish1_y'], index), 
                                                                                                                       getPoint(df['Fish1_x'], df['Fish1_y'], index+1))
            bar.next()

    # Remeber to save the result from the temporary variable
    df['Fish0_interframe_movement_dist'] = temp_df_fish0_dist.copy()
    df['Fish1_interframe_movement_dist'] = temp_df_fish1_dist.copy()
    df['Fish0_interframe_moving_direction_x'] = temp_df_fish0_direction_x.copy()
    df['Fish0_interframe_moving_direction_y'] = temp_df_fish0_direction_y.copy()
    df["Fish1_interframe_moving_direction_x"] = temp_df_fish1_direction_x.copy()
    df["Fish1_interframe_moving_direction_y"] = temp_df_fish1_direction_y.copy()

    # Setting save file path. If folder is not exist, create a new one
    save_folder = folder_path + "preprocessed_data/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save the basic information (basic result) into a csv file
    df.to_csv(save_folder + video_name + "_" + filter_name + "_basic_result.csv", index = False)
    print("Complete distance calculation. The file had been saved in: " + folder_path + "\n")


def dtw_same_interval(start_frame, end_frame, traj_df):
    traj1 = np.column_stack((traj_df[start_frame:end_frame, 1], traj_df[start_frame:end_frame, 2]))
    traj2 = np.column_stack((traj_df[start_frame:end_frame, 3], traj_df[start_frame:end_frame, 4]))
    dtw_distance = round(dtw_ndim.distance(traj1, traj2), 2)
    return dtw_distance


def caculate_avg_velocity(start_frame, end_frame, traj_df):
    avg_dist_fish0 = round(traj_df[start_frame:end_frame, 6].mean(), 2)
    avg_dist_fish1 = round(traj_df[start_frame:end_frame, 7].mean(), 2)
    return avg_dist_fish0, avg_dist_fish1


def calculate_movement_length(start_frame, end_frame, traj_df):
    movement_length_fish0 = traj_df[start_frame:end_frame, 6].sum()
    movement_length_fish1 = traj_df[start_frame:end_frame, 7].sum()
    return movement_length_fish0, movement_length_fish1


def calculate_final_result(folder_path, video_name, filter_name):
    # Read trajectory data
    resource_folder = folder_path + "preprocessed_data/"
    file_path = resource_folder + video_name + "_" + filter_name + "_basic_result.csv"
    traj_coordinate_df = np.genfromtxt(file_path, delimiter=",", dtype=int, skip_header=1)

    # Reading annotation file
    anno_resource_folder = folder_path + "annotation_information_data/"
    anno_file_path = anno_resource_folder + video_name + "_annotation_information_sorted.csv"
    anno_df = pd.read_csv(anno_file_path)

    # Add a new column name 'DTW_distance' in anno_df, and set default values with 0
    anno_df['DTW_distance'] = 0
    anno_df['Fish0_avg_velocity'] = 0
    anno_df['Fish1_avg_velocity'] = 0
    anno_df['Fish0_movement_length'] = 0
    anno_df['Fish1_movement_length'] = 0
    anno_df['movement_length_differnece'] = 0
    anno_df['Fish0_moving_direction_x'] = 0
    anno_df['Fish0_moving_direction_y'] = 0
    anno_df["Fish1_moving_direction_x"] = 0
    anno_df["Fish1_moving_direction_y"] = 0

    # use a temporary variable to prevent SettingWithCopyWarning problem
    temp_df_dtw = anno_df['DTW_distance'].copy()  
    temp_df_fish0_avgv = anno_df['Fish0_avg_velocity'].copy()
    temp_df_fish1_avgv = anno_df['Fish1_avg_velocity'].copy()
    temp_df_fish0_movement_length = anno_df['Fish0_movement_length'].copy()
    temp_df_fish1_movement_length = anno_df['Fish1_movement_length'].copy()
    temp_movementl_diff_df = anno_df['movement_length_differnece'].copy()
    temp_df_direction_fish0_x = anno_df['Fish0_moving_direction_x'].copy()
    temp_df_direction_fish0_y = anno_df['Fish0_moving_direction_y'].copy()
    temp_df_direction_fish1_x = anno_df["Fish1_moving_direction_x"].copy()
    temp_df_direction_fish1_y = anno_df["Fish1_moving_direction_y"].copy()

    # Calculate some features in the same trajectory interval between two trajectories
    with IncrementalBar(video_name + ' - Progress of Final Caculation', max=len(anno_df.index)) as bar:  # with a progress bar
        for index in range(0, len(anno_df.index)):
            # get a line of interval information from annotation data
            start_frame, end_frame = anno_df['StartFrame'].iloc[index], anno_df['EndFrame'].iloc[index]

            # calculate DTW in the same interval (compare the trajectory of fish 0 and fish 1)
            temp_df_dtw.iloc[index] = dtw_same_interval(start_frame, end_frame, traj_coordinate_df)

            # calculate average velocity in each trajectory
            temp_df_fish0_avgv.iloc[index],  temp_df_fish1_avgv.iloc[index] = caculate_avg_velocity(start_frame, end_frame, traj_coordinate_df)

            # calculate movement length in each trajectory
            movement_length_fish0, movement_length_fish1 = calculate_movement_length(start_frame, end_frame, traj_coordinate_df)
            temp_df_fish0_movement_length.iloc[index] = movement_length_fish0
            temp_df_fish1_movement_length.iloc[index] = movement_length_fish1

            # calculate movement length difference between two trajectories
            temp_movementl_diff_df.iloc[index] = movement_length_fish0 - movement_length_fish1

            # calculate a direction feature in a trajectory
            fish0_direction_x, fish0_direction_y, fish1_direction_x, fish1_direction_y = caculate_direction(start_frame, end_frame, traj_coordinate_df)
            temp_df_direction_fish0_x.iloc[index], temp_df_direction_fish0_y.iloc[index] = fish0_direction_x, fish0_direction_y
            temp_df_direction_fish1_x.iloc[index], temp_df_direction_fish1_y.iloc[index] = fish1_direction_x, fish1_direction_y

            bar.next()

    # Remeber to save the result from the temporary variable
    anno_df["DTW_distance"] = temp_df_dtw.copy()
    anno_df['Fish0_avg_velocity'] = temp_df_fish0_avgv.copy()
    anno_df['Fish1_avg_velocity'] = temp_df_fish1_avgv.copy()
    anno_df['Fish0_movement_length'] = temp_df_fish0_movement_length.copy()
    anno_df['Fish1_movement_length'] = temp_df_fish1_movement_length.copy()
    anno_df['movement_length_differnece'] = temp_movementl_diff_df.copy()
    anno_df['Fish0_moving_direction_x'] = temp_df_direction_fish0_x.copy()
    anno_df['Fish0_moving_direction_y'] = temp_df_direction_fish0_y.copy()
    anno_df["Fish1_moving_direction_x"] = temp_df_direction_fish1_x.copy()
    anno_df["Fish1_moving_direction_y"] = temp_df_direction_fish1_y.copy()

    # Save the result in a new csv file
    anno_df.to_csv(resource_folder + video_name + "_" + filter_name + "_preprocessed_result.csv", index = False)
    print("Complete DTW, movement length, movement length difference and average velocity calculation.")
    print("The file had been saved in: " + folder_path)
    print("\n")
