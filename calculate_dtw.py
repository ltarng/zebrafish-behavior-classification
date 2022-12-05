import numpy as np
import pandas as pd
from dtaidistance import dtw_ndim


def dtw_same_interval(start_frame, end_frame, traj_df):
    traj1 = np.column_stack((traj_df[start_frame:end_frame, 1], traj_df[start_frame:end_frame, 2]))
    traj2 = np.column_stack((traj_df[start_frame:end_frame, 3], traj_df[start_frame:end_frame, 4]))
    distance = round(dtw_ndim.distance(traj1, traj2), 2)
    return distance


def calculate_dtw(folder_path, video_name, filter_name, ifPrintResult):
    # Read trajectory data
    file_path = folder_path + video_name + "_" + filter_name + "_filtered_" + "annotated.csv"
    traj_coordinate_df = np.genfromtxt(file_path, delimiter=",", dtype=int, skip_header=1)

    # Reading annotation file
    resource_path = folder_path + "annotation_information/"
    anno_file_path = resource_path + video_name + "_annotation_information_sorted.csv"
    anno_df = pd.read_csv(anno_file_path)


    # Add a new column name 'DTW_distance' in anno_df, and set default values with 0
    anno_df['DTW_distance'] = 0
    temp_df = anno_df['DTW_distance'].copy()  # use a temporary variable to prevent SettingWithCopyWarning problem

    # Calculate DTW  in the same trajectory interval between two trajectories
    for index in range(0, len(anno_df.index)):
        # get information from annotation data
        behavior_type = anno_df['BehaviorType'].iloc[index]
        start_frame, end_frame = anno_df['StartFrame'].iloc[index], anno_df['EndFrame'].iloc[index]

        # calculate DTW
        dtw_distance = dtw_same_interval(start_frame, end_frame, traj_coordinate_df)
        temp_df.iloc[index] = dtw_distance

        if ifPrintResult:
            print("\nBehavior: " + behavior_type)
            print( "From frame " + str(start_frame) + " to " + str(end_frame))
            print(dtw_distance)
    # Remeber to save the result from the temporary variable
    anno_df["DTW_distance"] = temp_df.copy()

    # Save the DTW result
    anno_df.to_csv(video_name + "_" + filter_name + "_filtered_DTW.csv", index = False)
    print("Complete DTW calculation.")
    print("The file had been saved in: " + folder_path)
    print("\n")
