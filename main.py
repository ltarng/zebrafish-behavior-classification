import annotation
import calculate_dtw
import data_cleaning
import svm_model


""" GENERAL PARAMETER SETTING"""
folder_path = "D:/Google Cloud (60747050S)/Research/Trajectory Analysis/"
video_name = '1-14'
filter_name = "mean"


if __name__ == '__main__':
    # Data cleaning step
    ifPlotTraj = False
    data_cleaning.data_cleaning(folder_path, video_name, filter_name, ifPlotTraj)
    
    # annotation step
    annotation.auto_annotation(folder_path, video_name, filter_name)
    annotation.sort_annotation_information(folder_path, video_name)

    # Analysis step
    ifPrintDTWResult = False
    calculate_dtw.calculate_dtw(folder_path, video_name, filter_name, ifPrintDTWResult)
    svm_model.svm(folder_path, video_name, filter_name)
    