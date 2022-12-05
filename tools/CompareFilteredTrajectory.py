import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers


def plot_trajectory(video_name, df, df_filtered):
    """ PARAMETER SETTING FOR PLOTTING POINTS """
    interval_seconds = 15
    fps = 30
    interval_frames = interval_seconds * fps
    number_of_intervals = 10

    fig = go.Figure()

    # unfiltered data
    for i in range(number_of_intervals):
        title = "Fish 0 - Frame " + str(interval_frames*i) + " to " + str(interval_frames*(i+1))
        fig.add_trace(go.Scatter(
            x = df['Fish0_x'].iloc[interval_frames*i:interval_frames*(i+1)],
            y = df['Fish0_y'].iloc[interval_frames*i:interval_frames*(i+1)],
            marker = dict(size=3, symbol='square'),
            mode = "lines+markers",
            # mode = "markers",
            name = title,
        ))
    for i in range(number_of_intervals):
        title = "Fish 1 - Frame " + str(interval_frames*i) + " to " + str(interval_frames*(i+1))
        fig.add_trace(go.Scatter(
            x = df['Fish1_x'].iloc[interval_frames*i:interval_frames*(i+1)],
            y = df['Fish1_y'].iloc[interval_frames*i:interval_frames*(i+1)],
            marker = dict(size=3, symbol='diamond'),
            mode = "lines+markers",
            # mode = "markers",
            name = title,
        ))

    # filtered data
    for i in range(number_of_intervals):
        title = "Fish 0 (filtered) - Frame " + str(interval_frames*i) + " to " + str(interval_frames*(i+1))
        fig.add_trace(go.Scatter(
            x = df_filtered['Fish0_x'].iloc[interval_frames*i:interval_frames*(i+1)],
            y = df_filtered['Fish0_y'].iloc[interval_frames*i:interval_frames*(i+1)],
            marker = dict(size=3, symbol='cross'),
            mode = "lines+markers",
            #   mode = "markers",
            name = title,
        ))
    for i in range(number_of_intervals):
        title = "Fish 1 (filtered) - Frame " + str(interval_frames*i) + " to " + str(interval_frames*(i+1))
        fig.add_trace(go.Scatter(
            x = df_filtered['Fish1_x'].iloc[interval_frames*i:interval_frames*(i+1)],
            y = df_filtered['Fish1_y'].iloc[interval_frames*i:interval_frames*(i+1)],
            marker = dict(size=3, symbol='x'),
            mode = "lines+markers",
            #   mode = "markers",
            name = title,
        ))

    # set name of title, x-axis and y-axis
    fig.update_layout(title="Movement Trajectory | " + video_name,
                    xaxis_title="X-axis",
                    yaxis_title="Y-axis")
    fig.show()


def plot_filtered_traj_for_compare(folder_path, video_name, filter_name):
    # Read unfiltered trajectory data
    df = pd.read_csv(folder_path + video_name + '_tracked.csv')

    # Read filtered trajectory data
    folder_path_filtered = folder_path + "cleaned_data/"
    df_filtered = pd.read_csv(folder_path_filtered + video_name + '_' + filter_name + '_filtered.csv')

    # Plot both filtered and unfiltered trajectory data in the same figure
    plot_trajectory(video_name, df, df_filtered)
