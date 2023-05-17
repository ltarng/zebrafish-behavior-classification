import numpy as np
import pandas as pd
from dtaidistance import dtw_ndim
import math
from progress.bar import IncrementalBar
import os


def calculate_dtw_ver2(start_frame, end_frame, traj_df_f0_x, traj_df_f0_y, traj_df_f1_x, traj_df_f1_y):
    traj1 = np.column_stack((traj_df_f0_x[start_frame:end_frame], traj_df_f0_y[start_frame:end_frame]))
    traj2 = np.column_stack((traj_df_f1_x[start_frame:end_frame], traj_df_f1_y[start_frame:end_frame]))
    dtw_distance = dtw_ndim.distance(traj1, traj2)
    return round(dtw_distance, 2)


def caculate_avg_velocity(start_frame, end_frame, traj_df):
    avg_dist_fish0 = round(traj_df[start_frame:end_frame, 6].mean(), 2)
    avg_dist_fish1 = round(traj_df[start_frame:end_frame, 7].mean(), 2)
    return avg_dist_fish0, avg_dist_fish1


def caculate_avg_velocity_ver2(start_frame, end_frame, traj_df):
    avg_dist = round(traj_df[start_frame:end_frame].mean(), 2)
    return round(avg_dist, 2)


def calculate_movement_length(start_frame, end_frame, traj_df):
    movement_length_fish0 = traj_df[start_frame:end_frame, 6].sum()
    movement_length_fish1 = traj_df[start_frame:end_frame, 7].sum()
    return movement_length_fish0, movement_length_fish1


def calculate_movement_length_ver2(start_frame, end_frame, traj_df):
    movement_length = traj_df[start_frame:end_frame].sum()
    return round(movement_length, 2)


def calculate_direction_ver2(start_frame, end_frame, traj_df_x, traj_df_y):  # Under construction
    fish_mean_shift_x = traj_df_x[start_frame:end_frame].mean()
    fish_mean_shift_y = traj_df_y[start_frame:end_frame].mean()
    return round(fish_mean_shift_x, 2), round(fish_mean_shift_y, 2)


def calculate_same_direction_ratio_ver0(start_frame, end_frame, df_fish0_x_shift, df_fish0_y_shift, df_fish1_x_shift, df_fish1_y_shift):
    ''' Some lack of logic have to be solve: under (+,-) and (-,+) situation, the angle difference is 0 < x < 180. '''
    duration_time = end_frame - start_frame + 1
    same_direction_frames = 0
    for index in range(start_frame, end_frame):  # If x-axis is same direction and y-axis is same direction, counter+=1
        print("F0_sx, F0_sy, F1_sx, F1_sy: ", df_fish0_x_shift.iloc[index], df_fish1_x_shift.iloc[index], df_fish0_y_shift.iloc[index], df_fish1_y_shift.iloc[index])
        if df_fish0_x_shift.iloc[index]*df_fish1_x_shift.iloc[index] == 0 and df_fish0_y_shift.iloc[index]*df_fish1_y_shift.iloc[index] == 0:
            # Not in same direction. Angle difference is right angle or a line (one fish didn't move).
            continue
        elif df_fish0_x_shift.iloc[index]*df_fish1_x_shift.iloc[index] >= 0 and df_fish0_y_shift.iloc[index]*df_fish1_y_shift.iloc[index] >= 0:
            print("Is same direction!")
            same_direction_frames += 1
        elif (df_fish0_x_shift.iloc[index]*df_fish1_x_shift.iloc[index])*(df_fish0_y_shift.iloc[index]*df_fish1_y_shift.iloc[index] < 0) < 0:  # (+*-) or (-*+)
            angle_diff = 91
            if angle_diff < 90:  # In the same direction
                print("Is same direction!")
                same_direction_frames += 1
            else:  # Angle difference < 90, not in the same direction
                continue
        else:  # x-axis opposite or y-axis opposite
            # df_fish0_x_shift.iloc[index]*df_fish1_x_shift.iloc[index] < 0 or df_fish0_y_shift.iloc[index]*df_fish1_y_shift.iloc[index] < 0
            # Not in same direction
            continue
    same_direction_ratio = same_direction_frames/duration_time
    return round(same_direction_ratio, 2)


def caculate_angle_between_vectors(a, b):
    # Transform list to numpy array
    v1 = np.array(a)
    v2 = np.array(b)

    # Calculate module
    module_v1 = np.sqrt(v1.dot(v1))
    module_v2 = np.sqrt(v2.dot(v2))

    # Calculate dot product and cosine value 
    dot_value = v1.dot(v2)
    cosine_theta = dot_value / (module_v1*module_v2)

    # Calculate radian value and conversion to angle value
    radian = np.arccos(cosine_theta)
    angle = radian*180 / np.pi

    return round(angle, 2)


def calculate_same_direction_ratio(start_frame, end_frame, df_fish0_x_shift, df_fish0_y_shift, df_fish1_x_shift, df_fish1_y_shift):
    duration_time = end_frame - start_frame + 1
    same_direction_frames = 0
    for index in range(start_frame, end_frame):  # If x-axis is same direction and y-axis is same direction, counter+=1
        # print("F0_sx, F0_sy, F1_sx, F1_sy: ", df_fish0_x_shift.iloc[index], df_fish1_x_shift.iloc[index], df_fish0_y_shift.iloc[index], df_fish1_y_shift.iloc[index])
        vector_angle = caculate_angle_between_vectors([df_fish0_x_shift.iloc[index], df_fish0_y_shift.iloc[index]], 
                                                      [df_fish1_x_shift.iloc[index], df_fish1_y_shift.iloc[index]])
        # print("Vector Angle: ", vector_angle)
        if vector_angle < 90 and vector_angle > 0:
            same_direction_frames += 1
            # print("angle is 0~90 degree")
        else:
            # print("angle is 0 or >= 90 degree")
            continue
        
    same_direction_ratio = same_direction_frames/duration_time
    print(same_direction_ratio)
    return round(same_direction_ratio, 2)


# def calculate_same_direction_ratio_ver0(start_frame, end_frame, df_fish0_x_shift, df_fish0_y_shift, df_fish1_x_shift, df_fish1_y_shift):
#     ''' Some lack of logic have to be solve: under (+,-) and (-,+) situation, the angle difference is 0 < x < 180. '''
#     duration_time = end_frame - start_frame + 1
#     same_direction_frames = 0
#     for index in range(start_frame, end_frame):  # If x-axis is same direction and y-axis is same direction, counter+=1
#         print("F0_sx, F0_sy, F1_sx, F1_sy: ", df_fish0_x_shift.iloc[index], df_fish1_x_shift.iloc[index], df_fish0_y_shift.iloc[index], df_fish1_y_shift.iloc[index])
#         if df_fish0_x_shift.iloc[index]*df_fish1_x_shift.iloc[index] == 0 and df_fish0_y_shift.iloc[index]*df_fish1_y_shift.iloc[index] == 0:
#             # Not in same direction. Angle difference is right angle or a line (one fish didn't move).
#             continue
#         elif df_fish0_x_shift.iloc[index]*df_fish1_x_shift.iloc[index] >= 0 and df_fish0_y_shift.iloc[index]*df_fish1_y_shift.iloc[index] >= 0:
#             print("Is same direction!")
#             same_direction_frames += 1
#         elif (df_fish0_x_shift.iloc[index]*df_fish1_x_shift.iloc[index])*(df_fish0_y_shift.iloc[index]*df_fish1_y_shift.iloc[index] < 0) < 0:  # (+*-) or (-*+)
#             angle_diff = 91
#             if angle_diff < 90:  # In the same direction
#                 print("Is same direction!")
#                 same_direction_frames += 1
#             else:  # Angle difference < 90, not in the same direction
#                 continue
#         else:  # x-axis opposite or y-axis opposite
#             # df_fish0_x_shift.iloc[index]*df_fish1_x_shift.iloc[index] < 0 or df_fish0_y_shift.iloc[index]*df_fish1_y_shift.iloc[index] < 0
#             # Not in same direction
#             continue
#     same_direction_ratio = same_direction_frames/duration_time
#     return round(same_direction_ratio, 2)


def calculate_final_result(folder_path, video_name, filter_name):
    # Read trajectory data
    resource_folder = folder_path + "preprocessed_data/"
    file_path = resource_folder + video_name + "_" + filter_name + "_basic_result.csv"
    # basic_data_df = np.genfromtxt(file_path, delimiter=",", dtype=int, skip_header=1)
    basic_data_df = pd.read_csv(file_path)

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
    anno_df['same_direction_ratio'] = 0

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
    temp_df_same_direction_ratio = anno_df['same_direction_ratio'].copy()

    # Calculate some features in the same trajectory interval between two trajectories
    with IncrementalBar(video_name + ' - Progress of Final Caculation', max=len(anno_df.index)) as bar:  # with a progress bar
        for index in range(0, len(anno_df.index)):
            # get a line of interval information from annotation data
            start_frame, end_frame = anno_df['StartFrame'].iloc[index], anno_df['EndFrame'].iloc[index]

            # calculate DTW in the same interval (compare the trajectory of fish 0 and fish 1)
            temp_df_dtw.iloc[index] = calculate_dtw_ver2(start_frame, end_frame, basic_data_df['Fish0_x'], basic_data_df['Fish0_y'], basic_data_df['Fish1_x'], basic_data_df['Fish1_y'])

            # calculate average velocity in each trajectory
            temp_df_fish0_avgv.iloc[index] = caculate_avg_velocity_ver2(start_frame, end_frame, basic_data_df['Fish0_interframe_movement_dist'])
            temp_df_fish1_avgv.iloc[index] = caculate_avg_velocity_ver2(start_frame, end_frame, basic_data_df['Fish1_interframe_movement_dist'])

            # calculate movement length in each trajectory
            movement_length_fish0 = calculate_movement_length_ver2(start_frame, end_frame, basic_data_df['Fish0_interframe_movement_dist'])
            movement_length_fish1 = calculate_movement_length_ver2(start_frame, end_frame, basic_data_df['Fish1_interframe_movement_dist'])
            temp_df_fish0_movement_length.iloc[index] = movement_length_fish0
            temp_df_fish1_movement_length.iloc[index] = movement_length_fish1

            # calculate movement length difference between two trajectories
            temp_movementl_diff_df.iloc[index] = movement_length_fish0 - movement_length_fish1

            # calculate a direction feature in a trajectory
            fish0_direction_x, fish0_direction_y = calculate_direction_ver2(start_frame, end_frame, basic_data_df['Fish0_interframe_moving_direction_x'], basic_data_df['Fish0_interframe_moving_direction_y'])
            fish1_direction_x, fish1_direction_y = calculate_direction_ver2(start_frame, end_frame, basic_data_df['Fish1_interframe_moving_direction_x'], basic_data_df['Fish1_interframe_moving_direction_y'])
            temp_df_direction_fish0_x.iloc[index], temp_df_direction_fish0_y.iloc[index] = fish0_direction_x, fish0_direction_y
            temp_df_direction_fish1_x.iloc[index], temp_df_direction_fish1_y.iloc[index] = fish1_direction_x, fish1_direction_y
            
            # Calculate the ratio when fish swim in same direction in a trajectory 
            same_direction_ratio = calculate_same_direction_ratio(start_frame, end_frame, 
                                                                  basic_data_df['Fish0_interframe_moving_direction_x'], basic_data_df['Fish0_interframe_moving_direction_y'],
                                                                  basic_data_df['Fish1_interframe_moving_direction_x'], basic_data_df['Fish1_interframe_moving_direction_y'])
            temp_df_same_direction_ratio.iloc[index] = same_direction_ratio

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
    anno_df['same_direction_ratio'] = temp_df_same_direction_ratio.copy()

    print(anno_df['same_direction_ratio'])

    # Save the result in a new csv file
    anno_df.to_csv(resource_folder + video_name + "_" + filter_name + "_preprocessed_TEST.csv", index = False)
    print("Complete DTW, movement length, movement length difference and average velocity calculation.")
    print("The file had been saved in: " + folder_path)
    print("\n")


def main():
    # BASIC SETTING
    folder_path = "D:/Google Cloud (60747050S)/Research/Trajectory Analysis/"
    video_names = ['1-14', '1-22_2nd']
    filter_name = "mean"

    for index in range(0, len(video_names)):
        calculate_final_result(folder_path, video_names[index], filter_name)


if __name__ == '__main__':
    main()
