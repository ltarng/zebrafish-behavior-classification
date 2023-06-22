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
filter_name = "median"  # filter option: mean, median, kalman


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
        ifNotSkipSemi = False
        ifDoStd = True

        for index in range(0, len(video_names)):
            if ifNotSkipSemi:
                preprocess_calculate.calculate_semifinished_result(folder_path, video_names[index], filter_name)
            preprocess_calculate.calculate_final_result(folder_path, video_names[index], filter_name)
            preprocess_calculate.standarlize_preprocessed_data(folder_path, video_names[index], filter_name)

        # Combine cleaned data from different video
        combine_data.combine_preprocessed_main(folder_path, video_names, filter_name)


    # Analysis step
    if ifDoAnalysis:
        # model_name = "SVM"
        model_name = "RandomForest"
        # model_name = "XGBoost"

        # feature = "dtw"  # not bad
        # feature = "velocity" # not bad
        # feature = "min_max_velocity"  # not bad
        # feature = "direction"  # ok
        # feature = "same_direction_ratio"  # fine
        # feature = "movement_length"  # bad
        # feature = "avg_vector_angle"  # bad
        feature = "min_max_vector_angle"

        # feature = "dtw_velocity_related_direction_sdr"  # SVM: 61%, 140 sec | RF: 64%, 16 sec | XGBoost: 63%, 15 sec

        # feature = "dtw_velocity_related_direction_sdr_vecangle"  # SVM: 59%, 456 sec | RF: 64% | XGBoost: 65% (median), 8 sec
        # feature = "dtw_velocity_related_direction_sdr_length"  # SVM: 60%, 1386 sec | RF: 65% | XGBoost: 62%
        # feature = "dtw_velocity_related_direction_sdr_vecangle_length"  # SVM: 60%, 1126 sec| RF: 66%, 36 sec | XGBoost: 64%

        ## Traditional ML - Test
        # ml_model.machine_learning_main(folder_path, "Combined", filter_name, model_name, feature)  # only test one time
        # ml_model.machine_learning_cross_validation_test(folder_path, "Combined", filter_name, model_name, feature)  # observe the result in every round

        ## Usual use
        ml_model.machine_learning_main_cv_ver(folder_path, "Combined", filter_name, model_name, feature)  # 10-fold corss validation
        # ml_model.machine_learning_main_cv_3categories(folder_path, "Combined", filter_name, model_name, feature)
        # ml_model.machine_learning_main_cv_std(folder_path, "Combined", filter_name, model_name, feature)

        # Analysis the correlation of features and classes
        # Compare the features and the impoart features highligh by RF and XGBoost


        ## NN ML
        # CNN_model.deep_learning_main(folder_path, "Combined", filter_name, "1D-CNN", feature)
        # CNN_model.deep_learning_main_cv_ver(folder_path, "Combined", filter_name, "1D-CNN", feature)  # Under construction
        # LSTM_model.lstm_main(folder_path, "Combined", filter_name, "LSTM", feature)

        ## Combine bite and chase
        # CNN_model.deep_learning_main_3categories(folder_path, "Combined", filter_name, "1D-CNN", feature)
        # LSTM_model.lstm_main_3categories(folder_path, "Combined", filter_name, "LSTM", feature)


if __name__ == '__main__':
    main()
    