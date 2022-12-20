import numpy as np
import pandas as pd
from dtaidistance import dtw_ndim
import math


def getPoint(df_fish_x, df_fish_y, index):
    return [df_fish_x.iloc[index], df_fish_y.iloc[index]]


def distance(p1, p2):
    return round(math.dist(p1, p2), 2)


def calculate_distance_between_frames(folder_path, video_name, filter_name, ifPrintDistResult):
    # Read trajectory data
    resource_folder = folder_path + "training_data/"
    file_path = resource_folder + video_name + "_" + filter_name + "_filtered_annotated.csv"
    df = pd.read_csv(file_path)

    # Add a new column name 'FishN_velocity' in anno_df, and set default values with 0
    df['Fish0_velocity'] = 0
    df['Fish1_velocity'] = 0

    # use temporary variable to prevent SettingWithCopyWarning problem
    temp_df0 = df['Fish0_velocity'].copy()
    temp_df1 = df['Fish1_velocity'].copy()

    # Calculate DTW  in the same trajectory interval between two trajectories
    for index in range(0, len(df.index)-1):
        # calculate the distance between: frame n to frame n+1 
        fish0_dist = distance(getPoint(df['Fish0_x'], df['Fish0_y'], index), getPoint(df['Fish0_x'], df['Fish0_y'], index+1))
        fish1_dist = distance(getPoint(df['Fish1_x'], df['Fish1_y'], index), getPoint(df['Fish1_x'], df['Fish1_y'], index+1))

        temp_df0.iloc[index] = fish0_dist
        temp_df1.iloc[index] = fish1_dist

        if ifPrintDistResult:
            print("From frame " + str(index) + " to " + str(index+1))
            print("Fish0 distance: " + str(fish0_dist) + " Fish1 distance: " + str(fish1_dist))

    # Remeber to save the result from the temporary variable
    df['Fish0_velocity'] = temp_df0.copy()
    df['Fish1_velocity'] = temp_df1.copy()

    # Save the result into a csv file
    df.to_csv(video_name + "_" + filter_name + "_velocity_result.csv", index = False)
    print("Complete velocity calculation.")
    print("The file had been saved in: " + folder_path)
    print("\n")


def avg_velocity(start_frame, end_frame, traj_df):
    avg_dist_fish0 = round(traj_df[start_frame:end_frame, 6].mean(), 2)
    avg_dist_fish1 = round(traj_df[start_frame:end_frame, 7].mean(), 2)
    return avg_dist_fish0, avg_dist_fish1


def dtw_same_interval(start_frame, end_frame, traj_df):
    traj1 = np.column_stack((traj_df[start_frame:end_frame, 1], traj_df[start_frame:end_frame, 2]))
    traj2 = np.column_stack((traj_df[start_frame:end_frame, 3], traj_df[start_frame:end_frame, 4]))
    dtw_distance = round(dtw_ndim.distance(traj1, traj2), 2)
    return dtw_distance


def calculate_main(folder_path, video_name, filter_name):
    # Read trajectory data
    file_path = folder_path + video_name + "_" + filter_name + "_velocity_result.csv"
    traj_coordinate_df = np.genfromtxt(file_path, delimiter=",", dtype=int, skip_header=1)

    # Reading annotation file
    anno_resource_folder = folder_path + "annotation_information_data/"
    anno_file_path = anno_resource_folder + video_name + "_annotation_information_sorted.csv"
    anno_df = pd.read_csv(anno_file_path)

    # Add a new column name 'DTW_distance' in anno_df, and set default values with 0
    anno_df['DTW_distance'] = 0
    anno_df['avg_velocity_fish0'] = 0
    anno_df['avg_velocity_fish1'] = 0

    # use a temporary variable to prevent SettingWithCopyWarning problem
    temp_dtw_df = anno_df['DTW_distance'].copy()  
    temp_avgv_fish0_df = anno_df['avg_velocity_fish0'].copy()
    temp_avgv_fish1_df = anno_df['avg_velocity_fish1'].copy()

    # Calculate DTW  in the same trajectory interval between two trajectories
    for index in range(0, len(anno_df.index)):
        # get a line of interval information from annotation data
        start_frame, end_frame = anno_df['StartFrame'].iloc[index], anno_df['EndFrame'].iloc[index]

        # calculate DTW in the same interval (compare the trajectory of fish 0 and fish 1)
        dtw_distance = dtw_same_interval(start_frame, end_frame, traj_coordinate_df)
        temp_dtw_df.iloc[index] = dtw_distance

        # calculate average velocity in a trajectory
        avg_velocity_fish0, avg_velocity_fish1 = avg_velocity(start_frame, end_frame, traj_coordinate_df)
        temp_avgv_fish0_df.iloc[index] = avg_velocity_fish0
        temp_avgv_fish1_df.iloc[index] = avg_velocity_fish1

    # Remeber to save the result from the temporary variable
    anno_df['avg_velocity_fish0'] = temp_avgv_fish0_df.copy()
    anno_df['avg_velocity_fish1'] = temp_avgv_fish1_df.copy()
    anno_df["DTW_distance"] = temp_dtw_df.copy()

    # Save the DTW result
    anno_df.to_csv(video_name + "_" + filter_name + "_calculate_result.csv", index = False)
    print("Complete DTW and average distance calculation.")
    print("The file had been saved in: " + folder_path)
    print("\n")
