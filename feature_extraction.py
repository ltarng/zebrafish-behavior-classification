import numpy as np
import pandas as pd

FEATURE_MAP = {
    "dtw": lambda df: np.vstack(df['DTW_distance'].to_numpy()),
    "velocity": lambda df: np.column_stack((df['Fish0_avg_velocity'], df['Fish1_avg_velocity'])),
    "min_max_velocity": lambda df: np.column_stack((df['Fish0_max_velocity'], df['Fish1_max_velocity'], df['Fish0_min_velocity'], df['Fish1_min_velocity'])),
    "movement_length": lambda df: np.column_stack((df['Fish0_movement_length'], df['Fish1_movement_length'])),
    "movement_length_difference": lambda df: np.vstack(df['movement_length_difference'].to_numpy()),
    "direction": lambda df: np.column_stack((df['Fish0_moving_direction_x'], df['Fish0_moving_direction_y'], df['Fish1_moving_direction_x'], df['Fish1_moving_direction_y'])),
    "same_direction_ratio": lambda df: np.vstack(df['same_direction_ratio'].to_numpy()),
    "avg_vector_angle": lambda df: np.vstack(df['avg_vector_angle'].to_numpy()),
    "min_max_vector_angle": lambda df: np.column_stack((df['min_vector_angle'], df['max_vector_angle'])),
    "dtw_velocities_direction_sdr_angles_length": lambda df: np.column_stack((
        df['Fish0_avg_velocity'], df['Fish1_avg_velocity'], df['DTW_distance'],
        df['Fish0_movement_length'], df['Fish1_movement_length'],
        df['Fish0_max_velocity'], df['Fish1_max_velocity'], df['Fish0_min_velocity'], df['Fish1_min_velocity'],
        df['Fish0_moving_direction_x'], df['Fish0_moving_direction_y'], df['Fish1_moving_direction_x'], df['Fish1_moving_direction_y'],
        df['same_direction_ratio'], df['avg_vector_angle'], df['min_vector_angle'], df['max_vector_angle']
    )),
    "dtw_velocities_direction_sdr_partangles_length": lambda df: np.column_stack((
        df['Fish0_avg_velocity'], df['Fish1_avg_velocity'], df['DTW_distance'], 
        df['Fish0_movement_length'], df['Fish1_movement_length'], 
        df['Fish0_max_velocity'], df['Fish1_max_velocity'], df['Fish0_min_velocity'], df['Fish1_min_velocity'], 
        df['Fish0_moving_direction_x'], df['Fish0_moving_direction_y'], df['Fish1_moving_direction_x'], df['Fish1_moving_direction_y'], 
        df['same_direction_ratio'], df['min_vector_angle'], df['max_vector_angle']
    ))
}

def getFeaturesData(feature: str, df: pd.DataFrame) -> np.ndarray:
    if feature in FEATURE_MAP:
        return FEATURE_MAP[feature](df)
    else:
        raise ValueError(f"Error: feature name {feature} does not exist.")
