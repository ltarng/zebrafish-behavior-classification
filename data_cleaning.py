import plot_trajectory

import numpy as np
import pandas as pd
import scipy.ndimage as sn
import scipy.signal as ss
from pykalman import KalmanFilter


def mean_filter(df, window_len):
    # declare a dataframe format variable
    df_filtered = pd.DataFrame()

    df_filtered['Fish0_x'] = sn.uniform_filter(df['Fish0_x'], size = window_len)
    df_filtered['Fish0_y'] = sn.uniform_filter(df['Fish0_y'], size = window_len)
    df_filtered['Fish1_x'] = sn.uniform_filter(df['Fish1_x'], size = window_len)
    df_filtered['Fish1_y'] = sn.uniform_filter(df['Fish1_y'], size = window_len)
    return df_filtered


def median_filter(df, window_len):
    # declare a dataframe format variable
    df_filtered = pd.DataFrame()

    df_filtered['Fish0_x'] = ss.medfilt(df['Fish0_x'], kernel_size = window_len)
    df_filtered['Fish0_y'] = ss.medfilt(df['Fish0_y'], kernel_size = window_len)
    df_filtered['Fish1_x'] = ss.medfilt(df['Fish1_x'], kernel_size = window_len)
    df_filtered['Fish1_y'] = ss.medfilt(df['Fish1_y'], kernel_size = window_len)
    return df_filtered


def kalman_filter(df):
    # Reference: https://stackoverflow.com/questions/43377626/how-to-use-kalman-filter-in-python-for-location-data
    data = df.to_numpy()  # transform dataframe into numpy format

    """START OF SETTING PARAMETERS OF KALMAN FILTER"""
    measurements_fish0 = np.column_stack([data[:, 1], data[:, 2]])
    measurements_fish1 = np.column_stack([data[:, 3], data[:, 4]])

    initial_state_mean_fish0 = [measurements_fish0[0,0],
                                0,
                                measurements_fish0[0,1],
                                0]
    initial_state_mean_fish1 = [measurements_fish1[0,0],
                                0,
                                measurements_fish1[0,1],
                                0]

    transition_matrix = [[1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0],
                        [0, 0, 1, 0]]

    kf0 = KalmanFilter(transition_matrices = transition_matrix,
                    observation_matrices = observation_matrix,
                    initial_state_mean = initial_state_mean_fish0)
    kf1 = KalmanFilter(transition_matrices = transition_matrix,
                    observation_matrices = observation_matrix,
                    initial_state_mean = initial_state_mean_fish1)

    kf0 = kf1.em(measurements_fish0, n_iter=5)
    kf1 = kf1.em(measurements_fish1, n_iter=5)
    """END OF SETTING PARAMETERS OF KALMAN FILTER"""


    # EXECUTE KALMAN FILTER
    (smoothed_state_means_fish0, smoothed_state_covariances_fish0) = kf0.smooth(measurements_fish0)
    (smoothed_state_means_fish1, smoothed_state_covariances_fish1) = kf1.smooth(measurements_fish1)


    # TRANSFORM RESULT INTO DATAFRAME FORMAT
    # Get specific column from numpy result, and round the values, and join them
    array1 = np.vstack([np.round(smoothed_state_means_fish0[:, 0]), np.round(smoothed_state_means_fish0[:, 2])])
    array2 = np.vstack([np.round(smoothed_state_means_fish1[:, 0]), np.round(smoothed_state_means_fish1[:, 2])])
    array  = np.vstack([array1, array2])
    array = array.T  # Transpose array
    # Transform numpy array into dataframe
    df_filtered = pd.DataFrame(array, columns=["Fish0_x", "Fish0_y", "Fish1_x", "Fish1_y"])

    return df_filtered


def data_cleaning(folder_path, video_name, filter_name, ifPlotTraj):
    save_path = folder_path + "cleaned_data/"
    
    # Read from resource
    raw_data_path = "D:/Google Cloud (60747050S)/Research/FMTResult/fighting/"
    df = pd.read_csv(raw_data_path + video_name + '_tracked.csv')

    # Apply filter and save data
    window_len = 3  # window size should be odd number
    df_filtered = pd.DataFrame()
    if filter_name == "mean":
        df_filtered = mean_filter(df, window_len)
    elif filter_name == "median":
        df_filtered = median_filter(df, window_len)
    elif filter_name == "kalman":
        df_filtered = kalman_filter(df)

    df_filtered.to_csv(save_path + video_name + "_" + filter_name + "_filtered.csv", index_label="FrameIndex")
    
    # System messages
    print("Complete data cleaning by " + filter_name + " filter.")
    print("The file had been saved in: " + save_path + "\n")
    # print("\n")

    if ifPlotTraj == True:
        Plot.plot_trajectory(video_name, df, df_filtered)
