import numpy as np
import pandas as pd
from dtaidistance import dtw_ndim


def dtw_same_trajectory(start_frame, end_frame, traj_df):
    traj1 = np.column_stack((traj_df[start_frame:end_frame, 1], traj_df[start_frame:end_frame, 2]))
    traj2 = np.column_stack((traj_df[start_frame:end_frame, 3], traj_df[start_frame:end_frame, 4]))
    distance = round(dtw_ndim.distance(traj1, traj2), 2)
    return distance


def print_dtw_result(behavior, start_frame, end_frame, dtw_distance):
    print("\nBehavior: " + behavior)
    print( "From frame " + str(start_frame) + " to " + str(end_frame))
    print(dtw_distance)


def calculate_dtw(folder_path, video_name, filter_name, ifPrintResult):
    # Read trajectory data
    file_path = folder_path + video_name + '_' + filter_name + '_filtered_' + 'annotated.csv'
    traj_df = np.genfromtxt(file_path, delimiter=",", dtype=int, skip_header=1)

    # Reading annotation file
    resource_path = folder_path + "annotation_information/"
    anno_file_path = resource_path + video_name + "_annotation_information_sorted.csv"
    anno_df = pd.read_csv(anno_file_path)

    # set default values for a new column name 'DTW_distance'
    anno_df['DTW_distance'] = 0

    # Calculate DTW  in the same trajectory interval between two trajectories
    for index in range(0, len(anno_df.index)):
        # get information from annotation data
        behavior    = anno_df["BehaviorType"].iloc[index]
        start_frame = anno_df["StartFrame"].iloc[index]
        end_frame   = anno_df["EndFrame"].iloc[index]

        # calculate DTW
        dtw_distance = dtw_same_trajectory(start_frame, end_frame, traj_df)
        anno_df['DTW_distance'].iloc[index] = dtw_distance

        if ifPrintResult:
            print_dtw_result(behavior, start_frame, end_frame, dtw_distance)

    # Save the DTW result
    anno_df.to_csv(video_name + "_" + filter_name + "_filtered_DTW.csv", index = False)
    print("Complete DTW calculation.")
    print("The file had been saved in: " + folder_path)
    print("\n")
