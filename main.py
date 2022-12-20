import annotate_training_data
import preprocess_calculate
import data_cleaning
import svm_model


""" GENERAL PARAMETER SETTING """
folder_path = "D:/Google Cloud (60747050S)/Research/Trajectory Analysis/"
video_name = '1-14'
filter_name = "mean"  # filter option: mean, median, kalman


def main():
    """ EXECUTION OPTION SETTING """
    ifDoDataCleaning = False
    ifDoPreprocess = False
    ifDoAnnotate = False
    ifDoAnalysis = True

    # Data cleaning step
    if ifDoDataCleaning:
        ifPlotTraj = False
        data_cleaning.data_cleaning(folder_path, video_name, filter_name, ifPlotTraj)

    if ifDoPreprocess:
        ifPrintDistResult = False
        preprocess_calculate.calculate_distance_between_frames(folder_path, video_name, filter_name, ifPrintDistResult)
    
    # Training data annotation step
    if ifDoAnnotate:
        annotate_training_data.auto_annotate(folder_path, video_name, filter_name)
        annotate_training_data.sort_annotation_information(folder_path, video_name)

    # Analysis step
    if ifDoAnalysis:
        preprocess_calculate.calculate_main(folder_path, video_name, filter_name)
        svm_model.svm(folder_path, video_name, filter_name)
        # svm_model.random_forest(folder_path, video_name, filter_name)


if __name__ == '__main__':
    main()
    