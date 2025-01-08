from progress.bar import IncrementalBar
from sklearn.preprocessing import MinMaxScaler
from io_utils import read_csv, save_results
import feature_engineering as feature_engr


def assign_temp_columns(df, column_names):
    # Create temporary columns to prevent SettingWithCopyWarning
    temp_columns = {}
    for column in column_names:
        temp_columns[column] = df[column].copy()
    return temp_columns


def save_temp_columns_back(df, temp_columns):
    for column, temp_column in temp_columns.items():
        df[column] = temp_column.copy()


def calculate_semifinished_result(folder_path, video_name, filter_name):  # Calculate distance and direction between frames
    # Read trajectory data
    df = read_csv(folder_path + "annotated_data/" + video_name + "_" + filter_name + "_filtered_annotated.csv")

    # Initialize and assign temporary columns
    column_names = ['Fish0_interframe_movement_dist', 'Fish1_interframe_movement_dist',
                    'Fish0_interframe_moving_direction_x', 'Fish0_interframe_moving_direction_y',
                    "Fish1_interframe_moving_direction_x", "Fish1_interframe_moving_direction_y"]
    df[column_names] = 0
    
    temp_columns = assign_temp_columns(df, column_names)

    # Calculate moving distance in the same trajectory interval between two trajectories
    with IncrementalBar(video_name + ' - Progress of Basic Caculation', max=len(df.index)) as bar:  # with a progress bar
        for index in range(0, len(df.index)-1):
            feature_engr.calculate_distance_and_vector(temp_columns, index, df, 'Fish0_')
            feature_engr.calculate_distance_and_vector(temp_columns, index, df, 'Fish1_')
            bar.next()

    # Save results
    save_temp_columns_back(df, temp_columns)
    save_results(df, folder_path, video_name, filter_name, "_basic_result.csv")


def calculate_final_result(folder_path, video_name, filter_name):
    basic_data_df = read_csv(folder_path + "preprocessed_data/" + video_name + "_" + filter_name + "_basic_result.csv")
    anno_df = read_csv(folder_path + "annotation_information_data/" + video_name + "_annotation_information_sorted.csv")

    # Initialize and assign temporary columns
    feature_names = ['DTW_distance', 'Fish0_avg_velocity', 'Fish1_avg_velocity',
               'Fish0_min_velocity', 'Fish1_min_velocity', 'Fish0_max_velocity', 'Fish1_max_velocity',
               'Fish0_movement_length', 'Fish1_movement_length', 'movement_length_difference',
               'Fish0_moving_direction_x', 'Fish0_moving_direction_y', "Fish1_moving_direction_x", "Fish1_moving_direction_y",
               'same_direction_ratio', 'min_vector_angle', 'max_vector_angle', 'avg_vector_angle']
    anno_df[feature_names] = 0

    temp_columns = assign_temp_columns(anno_df, feature_names)

    # Calculate some features in the same trajectory interval between two trajectories
    with IncrementalBar(video_name + ' - Progress of Final Caculation', max=len(anno_df.index)) as bar:  # with a progress bar
        for index in range(0, len(anno_df.index)):
            # get a line of interval information from annotation data
            start_frame, end_frame = anno_df['StartFrame'].iloc[index], anno_df['EndFrame'].iloc[index]

            # calculate DTW in the same interval (compare the trajectory of fish 0 and fish 1)
            temp_columns['DTW_distance'].iloc[index] = feature_engr.calculate_dtw(
                start_frame, end_frame, basic_data_df['Fish0_x'], basic_data_df['Fish0_y'], basic_data_df['Fish1_x'], basic_data_df['Fish1_y']
            )

            # Average, min and Max velocity
            for fish in ['Fish0', 'Fish1']:
                temp_columns[f'{fish}_avg_velocity'].iloc[index] = feature_engr.calculate_avg_velocity(
                    start_frame, end_frame, basic_data_df[f'{fish}_interframe_movement_dist']
                )
                temp_columns[f'{fish}_min_velocity'].iloc[index], temp_columns[f'{fish}_max_velocity'].iloc[index] = feature_engr.get_min_max(
                    start_frame, end_frame, basic_data_df[f'{fish}_interframe_movement_dist']
                )

            # Movement length
            for fish in ['Fish0', 'Fish1']:
                temp_columns[f'{fish}_movement_length'].iloc[index] = feature_engr.calculate_movement_length(
                    start_frame, end_frame, basic_data_df[f'{fish}_interframe_movement_dist']
                )
            
            # Movement Length Difference between two trajectories
            temp_columns['movement_length_difference'].iloc[index] = round(
                temp_columns['Fish0_movement_length'].iloc[index] - temp_columns['Fish1_movement_length'].iloc[index], 2
            )

            # Moving direction
            for fish in ['Fish0', 'Fish1']:
                temp_columns[f'{fish}_moving_direction_x'].iloc[index], temp_columns[f'{fish}_moving_direction_y'].iloc[index] = feature_engr.calculate_direction(
                    start_frame, end_frame, basic_data_df[f'{fish}_interframe_moving_direction_x'], basic_data_df[f'{fish}_interframe_moving_direction_y']
                )

            # Vector angles and same direction ratio
            df_vector_angles = feature_engr.calculate_vector_angles(start_frame, end_frame, 
                basic_data_df['Fish0_interframe_moving_direction_x'], basic_data_df['Fish0_interframe_moving_direction_y'],
                basic_data_df['Fish1_interframe_moving_direction_x'], basic_data_df['Fish1_interframe_moving_direction_y']
            )
            temp_columns['min_vector_angle'].iloc[index], temp_columns['max_vector_angle'].iloc[index], temp_columns['avg_vector_angle'].iloc[index] = feature_engr.getVectorAnglesFeature(df_vector_angles['direction_vector_angle'])
            temp_columns['same_direction_ratio'].iloc[index] = feature_engr.calculate_same_direction_ratio(df_vector_angles['direction_vector_angle'])

            bar.next()

    save_temp_columns_back(anno_df, temp_columns)
    save_results(anno_df, folder_path, video_name, filter_name, "_preprocessed_result.csv")


def normalize_and_save(folder_path, video_name, filter_name):
    df = read_csv(folder_path + "preprocessed_data/" + video_name + "_" + filter_name + "_preprocessed_result.csv")

    scaler = MinMaxScaler()
    start_col, end_col = 4, 22  # be aware of this range if you change the amount of features
    df.iloc[:,start_col:end_col] = scaler.fit_transform(df.iloc[:,start_col:end_col].to_numpy())
    
    save_results(df, folder_path, video_name, filter_name, "_preprocessed_result_nor.csv")
