import math

def extract_coordinates(df_fish_x, df_fish_y, index):
    return [df_fish_x.iloc[index], df_fish_y.iloc[index]]

def compute_distance_between_frames(p0, p1):
    return round(math.dist(p0, p1), 2)

def compute_vector_between_frames(p0, p1):
    x_shift = p1[0] - p0[0]
    y_shift = p1[1] - p0[1]
    return x_shift, y_shift

def calculate_distance_and_vector(temp_columns, index, df, fish_prefix):
    temp_columns[fish_prefix + 'interframe_movement_dist'].iloc[index] = compute_distance_between_frames(
        extract_coordinates(df[fish_prefix + 'x'], df[fish_prefix + 'y'], index), 
        extract_coordinates(df[fish_prefix + 'x'], df[fish_prefix + 'y'], index+1)
    )
    temp_columns[fish_prefix + 'interframe_moving_direction_x'].iloc[index], temp_columns[fish_prefix + 'interframe_moving_direction_y'].iloc[index] = compute_vector_between_frames(
        extract_coordinates(df[fish_prefix + 'x'], df[fish_prefix + 'y'], index), 
        extract_coordinates(df[fish_prefix + 'x'], df[fish_prefix + 'y'], index+1)
    )