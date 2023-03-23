import annotate_training_data
import preprocess_calculate
import data_cleaning
import ml_model
import combine_data


""" GENERAL PARAMETER SETTING """
folder_path = "D:/Google Cloud (60747050S)/Research/Trajectory Analysis/"
video_names = ['1-14', '1-22_2nd']
filter_name = "mean"  # filter option: mean, median, kalman


def main():
    """ EXECUTION OPTION SETTING """
    ifDoDataCleaning = False
    ifDoAnnotate = False
    ifDoPreprocess = False
    ifDoCombine = True
    ifDoAnalysis = True

    # Data cleaning step
    if ifDoDataCleaning:
        ifPlotTraj = False
        for index in range(0, len(video_names)):
            data_cleaning.data_cleaning(folder_path, video_names[index], filter_name, ifPlotTraj)

    # Calculate some file with manual annotation information
    if ifDoAnnotate:
        for index in range(0, len(video_names)):
            annotate_training_data.auto_annotate(folder_path, video_names[index], filter_name)
            annotate_training_data.sort_annotation_information(folder_path, video_names[index])

    if ifDoPreprocess:
        for index in range(0, len(video_names)):
            preprocess_calculate.calculate_semifinished_result(folder_path, video_names[index], filter_name)
            preprocess_calculate.calculate_final_result(folder_path, video_names[index], filter_name)

    # # Combine cleaned data from different video
    if ifDoCombine:
        combine_data.combine_preprocessed_files(folder_path, video_names, filter_name)

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

        # ml_model.machine_learning_main(folder_path, video_name, filter_name, model_name, feature)
        ml_model.machine_learning_main(folder_path, "Combined", filter_name, model_name, feature)


if __name__ == '__main__':
    main()
    