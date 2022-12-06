import annotate_training_data
import calculate_dtw
import data_cleaning
import svm_model


""" GENERAL PARAMETER SETTING """
folder_path = "D:/Google Cloud (60747050S)/Research/Trajectory Analysis/"
video_name = '1-14'
filter_name = "kalman"  # filter option: mean, median, kalman


def main():
    """ EXECUTION OPTION SETTING """
    ifDoDataCleaning = False
    ifDoAnnotate = False
    ifDoAnalysis = True

    # Data cleaning step
    if ifDoDataCleaning:
        ifPlotTraj = False
        data_cleaning.data_cleaning(folder_path, video_name, filter_name, ifPlotTraj)
    
    # Training data annotation step
    if ifDoAnnotate:
        annotate_training_data.auto_annotate(folder_path, video_name, filter_name)
        annotate_training_data.sort_annotation_information(folder_path, video_name)

    # Analysis step
    if ifDoAnalysis:
        ifPrintDTWResult = False
        calculate_dtw.dtw_main(folder_path, video_name, filter_name, ifPrintDTWResult)
        svm_model.svm(folder_path, video_name, filter_name)


if __name__ == '__main__':
    main()
    