import annotate_training_data
import preprocess_calculate
import data_cleaning
import ml_model


""" GENERAL PARAMETER SETTING """
folder_path = "D:/Google Cloud (60747050S)/Research/Trajectory Analysis/"
video_name = '1-14'
filter_name = "mean"  # filter option: mean, median, kalman


def main():
    """ EXECUTION OPTION SETTING """
    ifDoDataCleaning = False
    ifDoAnnotate = False
    ifDoPreprocess = False
    ifDoAnalysis = True

    # Data cleaning step
    if ifDoDataCleaning:
        ifPlotTraj = False
        data_cleaning.data_cleaning(folder_path, video_name, filter_name, ifPlotTraj)
    
    # Calculate some file with manual annotation information
    if ifDoAnnotate:
        annotate_training_data.auto_annotate(folder_path, video_name, filter_name)
        annotate_training_data.sort_annotation_information(folder_path, video_name)

    if ifDoPreprocess:
        preprocess_calculate.calculate_semifinished_result(folder_path, video_name, filter_name)
        preprocess_calculate.calculate_final_result(folder_path, video_name, filter_name)

    # Analysis step
    if ifDoAnalysis:
        model_name = "SVM"
        # model_name = "RandomForest"

        # feature = "dtw"
        # feature = "velocity"
        # feature = "movement_length"
        # feature = "movement_length_difference"
        # feature = "movement_length_features"
        feature = "all"

        ml_model.machine_learning_main(folder_path, video_name, filter_name, model_name, feature)


if __name__ == '__main__':
    main()
    