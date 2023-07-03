import annotate_training_data
import preprocess_calculate
import data_cleaning
import ml_model
import combine_data
import CNN_model
import LSTM_model
import image_CNN



""" GENERAL PARAMETER SETTING """
folder_path = "D:/Google Cloud (60747050S)/Research/Trajectory Analysis/"
video_names = ['1-14', '1-22_2nd']
filter_name = "mean"  # filter option: mean, median, kalman
# filter_name = "median"  # filter option: mean, median, kalman
# filter_name = "kalman"  # filter option: mean, median, kalman
# filter_name = "nofilter"  # use raw data


def main():
    """ EXECUTION OPTION SETTING """
    ifDoDataCleaning = False
    ifDoAnnotate = False
    ifDoPreprocess = False
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

    # Calculate the features for classification and standarlize
    if ifDoPreprocess:
        ifSkipSemi = True

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

        class_num = 4

        model_name = "SVM"
        # model_name = "RandomForest"
        # model_name = "XGBoost"

        # feature = "dtw"  # not bad
        # feature = "movement_length"  # bad
        # feature = "velocity" # not bad
        # feature = "min_max_velocity"  # Good
        # feature = "direction"  # ok
        # feature = "min_max_vector_angle"  # fine
        # feature = "same_direction_ratio"  # bad
        # feature = "avg_vector_angle"  # fine

        # feature = "dtw_velocity_related_direction_sdr"  # SVM: 61%, 140 sec | RF: 64%, 16 sec | XGBoost: 63%, 15 sec

        # feature = "dtw_velocity_related_direction_sdr_vecangle"  # SVM: 59%, 456 sec | RF: 64% | XGBoost: 65% (median), 8 sec
        # feature = "dtw_velocity_related_direction_sdr_length"  # SVM: 60%, 1386 sec | RF: 65% | XGBoost: 62%
        # feature = "dtw_velocity_related_direction_sdr_vecangle_length"  # SVM: 60%, 1126 sec| RF: 66%, 36 sec | XGBoost: 64%

        # feature = "dtw_velocities_direction_sdr_partangles_length"  # "dtw_dist_speed_direction_sdr_partangles_length"
        feature = "dtw_velocities_direction_sdr_angles_length"

        # 
        if ifDoTuning:
            ml_model.hyperparameter_tuning(folder_path, "Combined", filter_name, model_name, feature, class_num)

        if ifDoTraining:
            # Traditional ML - Test
            # ml_model.machine_learning_main(folder_path, "Combined", filter_name, model_name, feature)  # only test one time
            # ml_model.machine_learning_cross_validation_test(folder_path, "Combined", filter_name, model_name, feature)  # observe the result in every round

            # Usual use
            ml_model.machine_learning_main_cv_ver(folder_path, "Combined", filter_name, model_name, feature, class_num)  # 10-fold corss validation

            # Analysis the correlation of features and classes
            # Compare the features and the impoart features highligh by RF and XGBoost

            ## NN ML, under construction
            # CNN_model.deep_learning_main(folder_path, "Combined", filter_name, "1D-CNN", feature, class_num)
            # CNN_model.deep_learning_main_cv_ver(folder_path, "Combined", filter_name, "1D-CNN", feature, class_num)  # Under construction
            # LSTM_model.lstm_main(folder_path, "Combined", filter_name, "LSTM", feature, class_num)


if __name__ == '__main__':
    main()
    