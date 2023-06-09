import numpy as np
import pandas as pd
from dtaidistance import dtw_ndim
import math
from progress.bar import IncrementalBar
import os


def getPoint(df_fish_x, df_fish_y, index):
    return [df_fish_x.iloc[index], df_fish_y.iloc[index]]


def caculate_interframe_distance(p1, p2):
    return round(math.dist(p1, p2), 2)


def calculate_interframe_vector(p0, p1):
    x_shift = p1[0] - p0[0]
    y_shift = p1[1] - p0[1]
    return x_shift, y_shift


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


def calculate_dtw(start_frame, end_frame, traj_df_f0_x, traj_df_f0_y, traj_df_f1_x, traj_df_f1_y):
    traj1 = np.column_stack((traj_df_f0_x[start_frame:end_frame-1], traj_df_f0_y[start_frame:end_frame-1]))
    traj2 = np.column_stack((traj_df_f1_x[start_frame:end_frame-1], traj_df_f1_y[start_frame:end_frame-1]))
    dtw_distance = dtw_ndim.distance(traj1, traj2)
    return round(dtw_distance, 2)


def caculate_avg_velocity(start_frame, end_frame, traj_df):
    avg_dist = round(traj_df[start_frame:end_frame-1].mean(), 2)
    return round(avg_dist, 2)


def get_min_max(start_frame, end_frame, traj_df):
    min_dist = min(traj_df[start_frame:end_frame-1])
    max_dist = max(traj_df[start_frame:end_frame-1])
    return min_dist, max_dist


def calculate_movement_length(start_frame, end_frame, traj_df):
    movement_length = traj_df[start_frame:end_frame-1].sum()
    return round(movement_length, 2)


def calculate_direction(start_frame, end_frame, traj_df_x, traj_df_y):  # Under construction
    fish_mean_shift_x = traj_df_x[start_frame:end_frame-1].sum()
    fish_mean_shift_y = traj_df_y[start_frame:end_frame-1].sum()
    return fish_mean_shift_x, fish_mean_shift_y


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
    angle = radian*180/np.pi

    return round(angle, 2)


def calculate_same_direction_ratio(start_frame, end_frame, df_fish0_x_shift, df_fish0_y_shift, df_fish1_x_shift, df_fish1_y_shift):
    duration_time = end_frame - start_frame + 1
    same_direction_frames = 0
    vector_angle_sum = 0
    for index in range(start_frame, end_frame):  # If x-axis is same direction and y-axis is same direction, counter+=1
        vector_angle = caculate_angle_between_vectors([df_fish0_x_shift.iloc[index], df_fish0_y_shift.iloc[index]], 
                                                      [df_fish1_x_shift.iloc[index], df_fish1_y_shift.iloc[index]])
        if vector_angle < 90 and vector_angle > 0:
            # Angle degree is 0~90
            same_direction_frames += 1
        else:
            # Angle value is >=90 or equal to zero
            continue

        vector_angle_sum = vector_angle_sum + vector_angle

    same_direction_ratio = same_direction_frames/duration_time
    avg_vector_angle = vector_angle_sum/duration_time
    return round(same_direction_ratio, 2), round(avg_vector_angle, 2)


def calculate_final_result(folder_path, video_name, filter_name):
    # Read trajectory data
    resource_folder = folder_path + "preprocessed_data/"
    file_path = resource_folder + video_name + "_" + filter_name + "_basic_result.csv"
    basic_data_df = pd.read_csv(file_path)

    # Reading annotation file
    anno_resource_folder = folder_path + "annotation_information_data/"
    anno_file_path = anno_resource_folder + video_name + "_annotation_information_sorted.csv"
    anno_df = pd.read_csv(anno_file_path)

    # Add a new column name 'DTW_distance' in anno_df, and set default values with 0
    anno_df['DTW_distance'] = 0
    anno_df['Fish0_avg_velocity'] = 0
    anno_df['Fish1_avg_velocity'] = 0
    anno_df['Fish0_min_velocity'] = 0
    anno_df['Fish1_min_velocity'] = 0
    anno_df['Fish0_max_velocity'] = 0
    anno_df['Fish1_max_velocity'] = 0
    anno_df['Fish0_movement_length'] = 0
    anno_df['Fish1_movement_length'] = 0
    anno_df['movement_length_differnece'] = 0
    anno_df['Fish0_moving_direction_x'] = 0
    anno_df['Fish0_moving_direction_y'] = 0
    anno_df["Fish1_moving_direction_x"] = 0
    anno_df["Fish1_moving_direction_y"] = 0
    anno_df['same_direction_ratio'] = 0
    anno_df['avg_vector_angle'] = 0

    # use a temporary variable to prevent SettingWithCopyWarning problem
    temp_df_dtw = anno_df['DTW_distance'].copy()  
    temp_df_fish0_avgv = anno_df['Fish0_avg_velocity'].copy()
    temp_df_fish1_avgv = anno_df['Fish1_avg_velocity'].copy()
    temp_df_fish0_minv = anno_df['Fish0_min_velocity'].copy()
    temp_df_fish1_minv = anno_df['Fish1_min_velocity'].copy()
    temp_df_fish0_maxv = anno_df['Fish0_max_velocity'].copy()
    temp_df_fish1_maxv = anno_df['Fish1_max_velocity'].copy()
    temp_df_fish0_movement_length = anno_df['Fish0_movement_length'].copy()
    temp_df_fish1_movement_length = anno_df['Fish1_movement_length'].copy()
    temp_movementl_diff_df = anno_df['movement_length_differnece'].copy()
    temp_df_direction_fish0_x = anno_df['Fish0_moving_direction_x'].copy()
    temp_df_direction_fish0_y = anno_df['Fish0_moving_direction_y'].copy()
    temp_df_direction_fish1_x = anno_df["Fish1_moving_direction_x"].copy()
    temp_df_direction_fish1_y = anno_df["Fish1_moving_direction_y"].copy()
    temp_df_same_direction_ratio = anno_df['same_direction_ratio'].copy()
    temp_df_avg_vector_angle = anno_df['avg_vector_angle'].copy()

    # Calculate some features in the same trajectory interval between two trajectories
    with IncrementalBar(video_name + ' - Progress of Final Caculation', max=len(anno_df.index)) as bar:  # with a progress bar
        for index in range(0, len(anno_df.index)):
            # get a line of interval information from annotation data
            start_frame, end_frame = anno_df['StartFrame'].iloc[index], anno_df['EndFrame'].iloc[index]

            # calculate DTW in the same interval (compare the trajectory of fish 0 and fish 1)
            temp_df_dtw.iloc[index] = calculate_dtw(start_frame, end_frame, basic_data_df['Fish0_x'], basic_data_df['Fish0_y'], basic_data_df['Fish1_x'], basic_data_df['Fish1_y'])

            # calculate average velocity in each trajectory
            temp_df_fish0_avgv.iloc[index] = caculate_avg_velocity(start_frame, end_frame, basic_data_df['Fish0_interframe_movement_dist'])
            temp_df_fish1_avgv.iloc[index] = caculate_avg_velocity(start_frame, end_frame, basic_data_df['Fish1_interframe_movement_dist'])

            # get the minimun and miximum velocity in each trajectory
            temp_df_fish0_minv.iloc[index], temp_df_fish0_maxv.iloc[index] = get_min_max(start_frame, end_frame, basic_data_df['Fish0_interframe_movement_dist'])
            temp_df_fish1_minv.iloc[index], temp_df_fish1_maxv.iloc[index] = get_min_max(start_frame, end_frame, basic_data_df['Fish1_interframe_movement_dist'])

            # calculate movement length in each trajectory
            movement_length_fish0 = calculate_movement_length(start_frame, end_frame, basic_data_df['Fish0_interframe_movement_dist'])
            movement_length_fish1 = calculate_movement_length(start_frame, end_frame, basic_data_df['Fish1_interframe_movement_dist'])
            temp_df_fish0_movement_length.iloc[index] = movement_length_fish0
            temp_df_fish1_movement_length.iloc[index] = movement_length_fish1

            # calculate movement length difference between two trajectories
            temp_movementl_diff_df.iloc[index] = round(movement_length_fish0 - movement_length_fish1, 2)

            # calculate a direction feature in a trajectory
            fish0_direction_x, fish0_direction_y = calculate_direction(start_frame, end_frame, basic_data_df['Fish0_interframe_moving_direction_x'], basic_data_df['Fish0_interframe_moving_direction_y'])
            fish1_direction_x, fish1_direction_y = calculate_direction(start_frame, end_frame, basic_data_df['Fish1_interframe_moving_direction_x'], basic_data_df['Fish1_interframe_moving_direction_y'])
            temp_df_direction_fish0_x.iloc[index], temp_df_direction_fish0_y.iloc[index] = fish0_direction_x, fish0_direction_y
            temp_df_direction_fish1_x.iloc[index], temp_df_direction_fish1_y.iloc[index] = fish1_direction_x, fish1_direction_y

            # Calculate the ratio when fish swim in same direction in a trajectory 
            same_direction_ratio, avg_vector_angle = calculate_same_direction_ratio(start_frame, end_frame, 
                                                                  basic_data_df['Fish0_interframe_moving_direction_x'], basic_data_df['Fish0_interframe_moving_direction_y'],
                                                                  basic_data_df['Fish1_interframe_moving_direction_x'], basic_data_df['Fish1_interframe_moving_direction_y'])
            temp_df_same_direction_ratio.iloc[index] = same_direction_ratio
            temp_df_avg_vector_angle.iloc[index] = avg_vector_angle

            bar.next()

    # Remeber to save the result from the temporary variable
    anno_df["DTW_distance"] = temp_df_dtw.copy()
    anno_df['Fish0_avg_velocity'] = temp_df_fish0_avgv.copy()
    anno_df['Fish1_avg_velocity'] = temp_df_fish1_avgv.copy()
    anno_df['Fish0_min_velocity'] = temp_df_fish0_minv.copy()
    anno_df['Fish1_min_velocity'] = temp_df_fish1_minv.copy()
    anno_df['Fish0_max_velocity'] = temp_df_fish0_maxv.copy()
    anno_df['Fish1_max_velocity'] = temp_df_fish1_maxv.copy()
    anno_df['Fish0_movement_length'] = temp_df_fish0_movement_length.copy()
    anno_df['Fish1_movement_length'] = temp_df_fish1_movement_length.copy()
    anno_df['movement_length_differnece'] = temp_movementl_diff_df.copy()
    anno_df['Fish0_moving_direction_x'] = temp_df_direction_fish0_x.copy()
    anno_df['Fish0_moving_direction_y'] = temp_df_direction_fish0_y.copy()
    anno_df["Fish1_moving_direction_x"] = temp_df_direction_fish1_x.copy()
    anno_df["Fish1_moving_direction_y"] = temp_df_direction_fish1_y.copy()
    anno_df['same_direction_ratio'] = temp_df_same_direction_ratio.copy()
    anno_df['avg_vector_angle'] = temp_df_avg_vector_angle.copy()

    # Save the result in a new csv file
    anno_df.to_csv(resource_folder + video_name + "_" + filter_name + "_preprocessed_result.csv", index = False)
    print("Complete DTW, movement length, movement length difference and average velocity calculation.")
    print("The file had been saved in: " + folder_path)
    print("\n")
