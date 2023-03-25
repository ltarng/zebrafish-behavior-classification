import matplotlib.pyplot as plt
import pandas as pd
import os
from progress.bar import IncrementalBar


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

    interval_frames = interval_duration * fps  # Calculate number of interval frames
    return number_of_intervals, interval_frames, interval_duration


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


def one_trajectory_to_images(folder_path, trajectory_type, video_name, filter_name):
    # Read source data
    file_path = folder_path + 'cleaned_data/' + video_name + "_" + filter_name + "_filtered.csv"
    print(file_path)
    df = pd.read_csv(file_path)

    # Save trajectory pictures
    number_of_intervals, interval_frames, interval_duration = trajectory_plotting_setting(trajectory_type)
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
        header_output_filename = str(interval_duration) + " sec_" + "_" + video_name + "_"
        output_filename = header_output_filename + str(interval_frames*i) + "-" + str(interval_frames*(i+1)) + ".png"
        save_path = "D:/Trajectory(image)/" + output_filename
        plt.savefig(save_path, dpi=200)
        plt.close()


def show_all_trajectory(folder_path, trajectory_type, filter_name, video_name):
    # Read source file
    file_path = folder_path + 'cleaned_data/' + video_name + "_" + filter_name + "_filtered.csv"
    print(file_path)
    df = pd.read_csv(file_path)

    # Show all trajectories in a figure
    number_of_intervals, interval_frames = trajectory_plotting_setting(trajectory_type)
    plt.figure()
    plt.title("Movement Trajectory | " + video_name + "_" + filter_name)

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
    plt.close()


def plot_fight_trajecory(traj_type, video_name, df, behavior_type, start_frame, end_frame):
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

    # Create folder for saving pictures
    save_folder = "D:/Trajectory(image)/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Save images of a  trajectory
    header_output_filename = behavior_type + "_" + traj_type + "_" + video_name + "_"
    save_path = save_folder + header_output_filename + str(start_frame) + "-" + str(end_frame) + ".png"
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_fight_trajectory_to_images(folder_path, traj_type, video_name, filter_name):  # unfinished, it just copy from the ONE_TRAJECTORY
    # Read source file
    file_path = folder_path + 'cleaned_data/' + video_name + "_" + filter_name + "_filtered.csv"
    df = pd.read_csv(file_path)

    # Read annotation information file
    anno_resource_folder = folder_path + "annotation_information_data/"
    anno_info_file_path = anno_resource_folder + video_name + "_annotation_information.csv"
    anno_df = pd.read_csv(anno_info_file_path)

    # Make behavior dictionary
    behavior_dict = {'normal': 1, 'display': 2, 'circle': 3, 'chase': 4, 'bite': 5}

    # Plot and save the trajectories as png files
    progress_bar_title = "Progress of Saving Trajectory Images (" + video_name+ "): "
    with IncrementalBar(progress_bar_title, max=len(anno_df.index)) as bar:  # Execute with progress bar
        for index in range(0, len(anno_df.index)):
            # Get trajectory annotation information
            behavior_type = anno_df['BehaviorType'].iloc[index]
            start_frame, end_frame = anno_df['StartFrame'].iloc[index], anno_df['EndFrame'].iloc[index]

            # Plot a trajectory
            if behavior_type in behavior_dict.keys():
                plot_fight_trajecory(traj_type, video_name, df, behavior_type, start_frame, end_frame)
            bar.next()


if __name__ == '__main__':
    # Setting source file and trajectory type
    folder_path = 'D:/Google Cloud (60747050S)/Research/Trajectory Analysis/'
    trajectory_type = "fighting"
    video_names = ['1-14', '1-22_2nd']
    filter_name = "mean"

    # Execution options
    ifShowTraj = False

    for index in range(0, len(video_names)):
        # Plot and save the graph of trajectories
        if trajectory_type == "fighting":
            plot_fight_trajectory_to_images(folder_path, trajectory_type, video_names[index], filter_name)
        else:
            one_trajectory_to_images(folder_path, trajectory_type, video_names[index], filter_name)

        # Plot all trajectory in the browser
        if ifShowTraj:
            show_all_trajectory(folder_path, trajectory_type, filter_name, video_names[index])
