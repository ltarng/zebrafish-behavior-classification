import matplotlib.pyplot as plt
import pandas as pd


def trajectory_plotting_setting(trajectory_type):
    # Setting interval duration
    interval_duration = 15  # seconds

    # FPS
    if(trajectory_type == "mirror_biting"):
        fps = 30
    elif(trajectory_type == "novel_tank"):
        fps = 15

    # Number of intervals
    if(interval_duration == 10):
        if(trajectory_type == "mirror_biting"):
            number_of_intervals = 90
        elif(trajectory_type == "novel_tank"):
            number_of_intervals = 135
    elif(interval_duration == 15):
        if(trajectory_type == "mirror_biting"):
            number_of_intervals = 60
        elif(trajectory_type == "novel_tank"):
            number_of_intervals = 90

    # Calculate number of interval frames
    interval_frames = interval_duration * fps

    return number_of_intervals, interval_frames


def size_of_border_setting(trajectory_type):
    if(trajectory_type == "mirror_biting"):
        xlim = [660, 1100]
        ylim = [260, 510]
    elif(trajectory_type == "novel_tank"):
        xlim = [230, 1800]
        ylim = [290, 980]
    elif(trajectory_type == "fighting"):
        xlim = [230, 1800]
        ylim = [290, 980]
    
    return xlim, ylim


def one_trajectory_to_images(trajectory_type, df):
    # Setting header name of output file
    header_output_filename = "10sec_mirror_S4_"

    # Save trajectory pictures
    number_of_intervals, interval_frames = trajectory_plotting_setting(trajectory_type)
    for i in range(number_of_intervals):
        plt.figure()
        # Frame str(interval_frames*i) to str(interval_frames*(i+1))
        x = df['Fish0_x'].iloc[interval_frames*i:interval_frames*(i+1)]
        y = df['Fish0_y'].iloc[interval_frames*i:interval_frames*(i+1)]
        plt.plot(x, y, lw = 1)

        # Set fixed size of border
        xaxis_range, yaxis_range = size_of_border_setting(trajectory_type)
        plt.xlim(xaxis_range)
        plt.ylim(yaxis_range)
        
        # Hide border, ticks and labels of axis
        plt.axis('off')

        # Save image of a trajectory
        output_filename = header_output_filename + str(interval_frames*i) + "-" + str(interval_frames*(i+1)) + ".png"
        save_path = "D:/Trajectory(image)/" + output_filename
        plt.savefig(save_path, dpi=200)


def show_all_trajectory(trajectory_type, filename, df):
    # Show all trajectories in a figure
    number_of_intervals, interval_frames = trajectory_plotting_setting(trajectory_type)
    plt.figure()
    plt.title("Movement Trajectory | " + filename)

    for i in range(number_of_intervals):
        # Frame str(interval_frames*i) to str(interval_frames*(i+1))
        x = df['Fish0_x'].iloc[interval_frames*i:interval_frames*(i+1)]
        y = df['Fish0_y'].iloc[interval_frames*i:interval_frames*(i+1)]
        plt.plot(x, y, lw = 1)

    # Set fixed size of border
    xaxis_range, yaxis_range = size_of_border_setting(trajectory_type)
    plt.xlim(xaxis_range)
    plt.ylim(yaxis_range)

    plt.show()


def getTrajectoryAnnotationInfo(anno_df, index):
    # Automatic annotate the trajectory data
    behavior_type = anno_df['BehaviorType'].iloc[index]
    start_frame, end_frame = anno_df['StartFrame'].iloc[index], anno_df['EndFrame'].iloc[index]

    return behavior_type, start_frame, end_frame


def plot_one_fight_trajecory(df, behavior_type, start_frame, end_frame):  # Only finish one trajectory
    # Setting header name of output file
    header_output_filename = behavior_type + "_fighting_1-14_"

    # Save trajectory pictures
    plt.figure()

    # Plot trajectory of fish 0
    fish0_x = df['Fish0_x'].iloc[start_frame:end_frame]
    fish0_y = df['Fish0_y'].iloc[start_frame:end_frame]
    plt.plot(fish0_x, fish0_y, lw = 1)

    # Plot trajectory of fish 1
    fish1_x = df['Fish1_x'].iloc[start_frame:end_frame]
    fish1_y = df['Fish1_y'].iloc[start_frame:end_frame]
    plt.plot(fish1_x, fish1_y, lw = 1)

    # Set fixed size of border
    xaxis_range, yaxis_range = size_of_border_setting(trajectory_type)
    plt.xlim(xaxis_range)
    plt.ylim(yaxis_range)
        
    # Hide border, ticks and labels of axis
    plt.axis('off')

    # Save image of a trajectory
    output_filename = header_output_filename + str(start_frame) + "-" + str(end_frame) + ".png"
    save_path = "D:/Trajectory(image)/" + output_filename
    plt.savefig(save_path, dpi=200)


def plot_fight_trajectory_to_images(df):  # unfinished, it just copy from the ONE_TRAJECTORY
    # Basic setting
    folder_path = "D:/Google Cloud (60747050S)/Research/Trajectory Analysis/"
    video_name = "1-14"

    # Read annotation information file
    anno_resource_folder = folder_path + "annotation_information_data/"
    anno_info_file_path = anno_resource_folder + video_name + "_annotation_information.csv"
    anno_df = pd.read_csv(anno_info_file_path)

    # Make behavior dictionary
    behavior_dict = {'normal': 1, 'display': 2, 'circle': 3, 'chase': 4, 'bite': 5}

    for index in range(0, len(anno_df.index)):
        behavior_type, start_frame, end_frame = getTrajectoryAnnotationInfo(anno_df, index)
        if behavior_type in behavior_dict.keys():
            plot_one_fight_trajecory(df, behavior_type, start_frame, end_frame)


if __name__ == '__main__':
    # Setting source file and trajectory type
    folder_path = 'D:/Google Cloud (60747050S)/Research/Trajectory Analysis/cleaned_data/'
    trajectory_type = "fighting"
    video_name = "1-14"
    filter_name = "mean"
    filename = video_name + "_" + filter_name + "_filtered.csv"

    # Read source file
    file_path = folder_path + filename
    print(file_path)
    df = pd.read_csv(file_path)

    # Execution options
    ifShowTraj = False

    # Execution
    if trajectory_type == "fighting":
        plot_fight_trajectory_to_images(df)
    else:
        one_trajectory_to_images(trajectory_type, df)

    # Plot all trajectory in the browser
    if ifShowTraj:
        show_all_trajectory(trajectory_type, filename, df)
