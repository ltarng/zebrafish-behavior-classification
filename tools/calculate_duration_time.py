import pandas as pd
import matplotlib.pyplot as plt


def calculate_duration_time(folder_path, video_name, filter_name):
    # Read annotation information file
    anno_resource_folder = folder_path + "annotation_information_data/"
    anno_info_file_path = anno_resource_folder + video_name + "_annotation_information.csv"
    anno_df = pd.read_csv(anno_info_file_path)

    # Make behavior dictionary
    behavior_dict = {'normal': 1, 'display': 2, 'circle': 3, 'chase': 4, 'bite': 5}

    # declare a new dataframe
    cols = ['BehaviorName', 'DurationTime']
    df = pd.DataFrame(columns = cols)

    # Plot and save the trajectories as png files
    for index in range(0, len(anno_df.index)):
        # Get trajectory annotation information
        behavior_type = anno_df['BehaviorName'].iloc[index]
        start_frame, end_frame = anno_df['StartFrame'].iloc[index], anno_df['EndFrame'].iloc[index]

        # Plot a trajectory
        if behavior_type in behavior_dict.keys():
            new_row = [behavior_type, end_frame - start_frame]
            df.loc[len(df)] = new_row
    
    return df


if __name__ == '__main__':
    # Setting source file and trajectory type
    folder_path = 'D:/Google Cloud (60747050S)/Research/Trajectory Analysis/'
    trajectory_type = "fighting"
    video_names = ['1-14', '1-22_2nd']
    filter_name = "mean"

    dt_df = pd.DataFrame()
    for index in range(0, len(video_names)):
        # Plot and save the graph of trajectories
        if trajectory_type == "fighting":
            dt_df = calculate_duration_time(folder_path, video_names[index], filter_name)
    
    plt.title("Duration Time of Trajectories")
    plt.xlabel("Duration Time of a Trajectory (sec)")
    plt.ylabel("Trajectory Amount")
    plt.hist(dt_df['DurationTime'], bins='auto')
    plt.show()
