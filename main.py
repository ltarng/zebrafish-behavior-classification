import annotate_training_data
import preprocess_calculate
import data_cleaning
import ml_model
import combine_data
import os


""" GENERAL PARAMETER SETTING """
folder_path = os.getcwd().replace('\\', '/') + '/'  # Get current working direction
video_names = ['1-14', '1-22_2nd']
# Filter option: mean, median, kalman | use raw data: nofilter
# filter_name = "mean"
# filter_name = "median"
# filter_name = "kalman"
filter_name = "nofilter"  # use raw data


def main():
    """ EXECUTION OPTION SETTING """
    ifDoDataCleaning = False
    ifDoAnnotate = False
    ifDoPreprocess = False
    ifDoAnalysis = True

    print(folder_path)
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

    # Calculate the features for classification and standarlize
    if ifDoPreprocess:
        ifSkipSemi = False

        for index in range(0, len(video_names)):
            if not ifSkipSemi:
                preprocess_calculate.calculate_semifinished_result(folder_path, video_names[index], filter_name)
            preprocess_calculate.calculate_final_result(folder_path, video_names[index], filter_name)
            preprocess_calculate.normalize_preprocessed_data(folder_path, video_names[index], filter_name)

        # Combine cleaned data from different video
        combine_data.combine_preprocessed_main(folder_path, video_names, filter_name)

    # Analysis step
    if ifDoAnalysis:
        ifDoTuning = False
        ifDoTraining = True

        class_num = 2  # option: 2 (normal/abnormal), 3 (high/moderate/low aggression), 4 (bite/chase/display/normal)

        # model option: 'SVM', 'RandomForest', 'XGBoost'
        model_name = "SVM"  
        # model_name = "RandomForest"
        # model_name = "XGBoost"

        feature = "dtw_velocities_direction_sdr_partangles_length"
        # feature = "dtw_velocities_direction_sdr_angles_length"

        if ifDoTuning:
            ml_model.hyperparameter_tuning(folder_path, "Combined", filter_name, model_name, feature, class_num)  # non-normalization data
            # ml_model.hyperparameter_tuning(folder_path, "Combined_nor", filter_name, model_name, feature, class_num)  # using normalization data

        if ifDoTraining:
            # Usual use
            ml_model.machine_learning_main_cv_ver(folder_path, "Combined", filter_name, model_name, feature, class_num)  # non-normalization data
            # ml_model.machine_learning_main_cv_ver(folder_path, "Combined_nor", filter_name, model_name, feature, class_num)  # using normalized data


if __name__ == '__main__':
    main()
    