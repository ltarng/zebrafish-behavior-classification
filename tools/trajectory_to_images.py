import matplotlib.pyplot as plt
import numpy as np
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
    
    return xlim, ylim


# Setting header name of output file
header_output_filename = "10sec_mirror_S4_"


# Setting source file and trajectory type

# trajectory_type = "novel_tank"
# filename = trajectory_type + "/4-16_tracked.csv"

trajectory_type = "mirror_biting"
filename = trajectory_type + "/mirror_S4_right_cleaned.csv"


# Read source file
folder_path = 'D:/Google Cloud (60747050S)/Research/FMTResult/'
file_path = folder_path + filename
df = pd.read_csv(file_path)


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


# Show all trajectories in a figure
# number_of_intervals, interval_frames = trajectory_plotting_setting(trajectory_type)
# plt.figure()
# plt.title("Movement Trajectory | " + filename)

# for i in range(number_of_intervals):
#     # Frame str(interval_frames*i) to str(interval_frames*(i+1))
#     x = df['Fish0_x'].iloc[interval_frames*i:interval_frames*(i+1)]
#     y = df['Fish0_y'].iloc[interval_frames*i:interval_frames*(i+1)]
#     plt.plot(x, y, lw = 1)

# # Set fixed size of border
# xaxis_range, yaxis_range = size_of_border_setting(trajectory_type)
# plt.xlim(xaxis_range)
# plt.ylim(yaxis_range)

# plt.show()
