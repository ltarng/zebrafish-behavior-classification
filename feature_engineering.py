import numpy as np
import pandas as pd
import math
from dtaidistance import dtw_ndim


def extract_coordinates(df_fish_x, df_fish_y, index):
    return [df_fish_x.iloc[index], df_fish_y.iloc[index]]


def calculate_distance_between_frames(p0, p1):
    return round(math.dist(p0, p1), 2)


def calculate_vector_between_frames(p0, p1):
    x_shift = p1[0] - p0[0]
    y_shift = p1[1] - p0[1]
    return x_shift, y_shift


def calculate_distance_and_vector(temp_columns, index, df, fish_prefix):
    temp_columns[fish_prefix + 'interframe_movement_dist'].iloc[index] = calculate_distance_between_frames(
        extract_coordinates(df[fish_prefix + 'x'], df[fish_prefix + 'y'], index), 
        extract_coordinates(df[fish_prefix + 'x'], df[fish_prefix + 'y'], index+1)
    )
    temp_columns[fish_prefix + 'interframe_moving_direction_x'].iloc[index], temp_columns[fish_prefix + 'interframe_moving_direction_y'].iloc[index] = calculate_vector_between_frames(
        extract_coordinates(df[fish_prefix + 'x'], df[fish_prefix + 'y'], index), 
        extract_coordinates(df[fish_prefix + 'x'], df[fish_prefix + 'y'], index+1)
    )


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