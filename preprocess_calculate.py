import numpy as np
import pandas as pd
from dtaidistance import dtw_ndim
import math
from progress.bar import IncrementalBar
import os
from sklearn.preprocessing import MinMaxScaler


def getPoint(df_fish_x, df_fish_y, index):
    return [df_fish_x.iloc[index], df_fish_y.iloc[index]]


def calculate_interframe_distance(p0, p1):
    return round(math.dist(p0, p1), 2)


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
            temp_df_fish0_dist.iloc[index] = calculate_interframe_distance(getPoint(df['Fish0_x'], df['Fish0_y'], index), getPoint(df['Fish0_x'], df['Fish0_y'], index+1))
            temp_df_fish1_dist.iloc[index] = calculate_interframe_distance(getPoint(df['Fish1_x'], df['Fish1_y'], index), getPoint(df['Fish1_x'], df['Fish1_y'], index+1))

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


def calculate_avg_velocity(start_frame, end_frame, traj_df):  # average moving interframe distance
    avg_dist = round(traj_df[start_frame:end_frame-1].mean(), 2)
    return round(avg_dist, 2)


def get_min_max(start_frame, end_frame, traj_df):
    min_value = min(traj_df[start_frame:end_frame-1])
    max_value = max(traj_df[start_frame:end_frame-1])
    return min_value, max_value


def calculate_movement_length(start_frame, end_frame, traj_df):  # total moving disance
    movement_length = traj_df[start_frame:end_frame-1].sum()
    return round(movement_length, 2)


def calculate_direction(start_frame, end_frame, traj_df_x, traj_df_y):  # summation of moving vectors
    fish_mean_shift_x = traj_df_x[start_frame:end_frame-1].sum()
    fish_mean_shift_y = traj_df_y[start_frame:end_frame-1].sum()
    return fish_mean_shift_x, fish_mean_shift_y


def calculate_angle_between_vectors(a, b):  # an angle between two interframe moving vectors
    # Transform list to numpy array
    v1, v2 = np.array(a), np.array(b)

    # Calculate module
    module_v1 = math.sqrt(v1.dot(v1))
    module_v2 = math.sqrt(v2.dot(v2))

    # Calculate dot product and cosine value 
    dot_value = v1.dot(v2)
    module_v1v2 = module_v1*module_v2
    if module_v1v2 == 0:
        cosine_theta = 0
    else:
        cosine_theta = dot_value / module_v1v2
    
    # cosine_theta may out of range by calculating problem. It always should be in [-1.0, 1.0].
    if cosine_theta > 1:
        cosine_theta = 1
    elif cosine_theta < -1:
        cosine_theta = -1
    else:
        cosine_theta = round(cosine_theta, 4)

    # Calculate radian value
    radian = math.acos(cosine_theta)

    # Convert radian into angle. Dual situation if radian equal to zero
    if radian > 0:
        angle = radian*180/math.pi
    else:
        angle = 0

    return round(angle, 2)


def getVectorAnglesFeature(df_vector_angles):
    min_angle = min(df_vector_angles)
    max_angle = max(df_vector_angles)
    avg_angle = round(df_vector_angles.mean(), 2)
    return min_angle, max_angle, avg_angle


def calculate_vector_angles(start_frame, end_frame, df_fish0_x_shift, df_fish0_y_shift, df_fish1_x_shift, df_fish1_y_shift):  # angles between two vectors
    vector_angles = []
    for index in range(start_frame, end_frame):
        vector_angle = calculate_angle_between_vectors([df_fish0_x_shift.iloc[index], df_fish0_y_shift.iloc[index]], 
                                                      [df_fish1_x_shift.iloc[index], df_fish1_y_shift.iloc[index]])
        vector_angles.append(vector_angle)
    df = pd.DataFrame(vector_angles, columns=['direction_vector_angle'])
    return df


def calculate_same_direction_ratio(df_vector_angles):  # SDR (Same Direction Ratio)
    duration_time = len(df_vector_angles.index)
    same_direction_frames = 0
    for index in range(duration_time):  # If x-axis is same direction and y-axis is same direction, counter+=1
        vector_angle = df_vector_angles.iloc[index]
        if vector_angle < 90 and vector_angle > 0:
            # Angle degree is 0~90
            same_direction_frames += 1
        else:
            # Angle value is >=90 or equal to zero
            continue

    same_direction_ratio = same_direction_frames/duration_time
    return round(same_direction_ratio, 2)


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
    anno_df['max_vector_angle'] = 0
    anno_df['min_vector_angle'] = 0
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
    temp_df_min_vector_angle = anno_df['min_vector_angle'].copy()
    temp_df_max_vector_angle = anno_df['max_vector_angle'].copy()
    temp_df_avg_vector_angle = anno_df['avg_vector_angle'].copy()

    # Calculate some features in the same trajectory interval between two trajectories
    with IncrementalBar(video_name + ' - Progress of Final Caculation', max=len(anno_df.index)) as bar:  # with a progress bar
        for index in range(0, len(anno_df.index)):
            # get a line of interval information from annotation data
            start_frame, end_frame = anno_df['StartFrame'].iloc[index], anno_df['EndFrame'].iloc[index]

            # calculate DTW in the same interval (compare the trajectory of fish 0 and fish 1)
            temp_df_dtw.iloc[index] = calculate_dtw(start_frame, end_frame, basic_data_df['Fish0_x'], basic_data_df['Fish0_y'], basic_data_df['Fish1_x'], basic_data_df['Fish1_y'])

            # calculate average velocity in each trajectory
            temp_df_fish0_avgv.iloc[index] = calculate_avg_velocity(start_frame, end_frame, basic_data_df['Fish0_interframe_movement_dist'])
            temp_df_fish1_avgv.iloc[index] = calculate_avg_velocity(start_frame, end_frame, basic_data_df['Fish1_interframe_movement_dist'])

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
            df_vector_angles = calculate_vector_angles(start_frame, end_frame, 
                                                        basic_data_df['Fish0_interframe_moving_direction_x'], basic_data_df['Fish0_interframe_moving_direction_y'],
                                                        basic_data_df['Fish1_interframe_moving_direction_x'], basic_data_df['Fish1_interframe_moving_direction_y'])
            # min_vector_angle, max_vector_angle = min(df_vector_angles['direction_vector_angle']), max(df_vector_angles['direction_vector_angle'])
            min_vector_angle, max_vector_angle, avg_vector_angle = getVectorAnglesFeature(df_vector_angles['direction_vector_angle'])
            same_direction_ratio = calculate_same_direction_ratio(df_vector_angles['direction_vector_angle'])
            temp_df_avg_vector_angle.iloc[index], temp_df_same_direction_ratio.iloc[index] = avg_vector_angle, same_direction_ratio
            temp_df_min_vector_angle.iloc[index], temp_df_max_vector_angle.iloc[index] = min_vector_angle, max_vector_angle

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
    anno_df['min_vector_angle'] = temp_df_min_vector_angle.copy()
    anno_df['max_vector_angle'] = temp_df_max_vector_angle.copy()
    anno_df['avg_vector_angle'] = temp_df_avg_vector_angle.copy()

    # Save the result in a new csv file
    anno_df.to_csv(resource_folder + video_name + "_" + filter_name + "_preprocessed_result.csv", index = False)
    print("Complete calculation. The file had been saved in: " + folder_path)
    print("\n")


def normalize_preprocessed_data(folder_path, video_name, filter_name):
    resource_folder = folder_path + "preprocessed_data/"
    df = pd.read_csv(resource_folder + video_name + '_' + filter_name + '_preprocessed_result.csv')

    scaler = MinMaxScaler()
    start_col, end_col = 4, 22  # be aware of this range if you change the amount of features
    df.iloc[:,start_col:end_col] = scaler.fit_transform(df.iloc[:,start_col:end_col].to_numpy())
    
    df.to_csv(resource_folder + video_name + "_" + filter_name + "_preprocessed_result_nor.csv", index = False)
    print("Complete Normalization. The file had been saved in: " + folder_path)
    print("\n")
