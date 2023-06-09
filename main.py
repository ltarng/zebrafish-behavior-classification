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

    # Calculate the features for classification (analysis step)
    if ifDoPreprocess:
        for index in range(0, len(video_names)):
            preprocess_calculate.calculate_semifinished_result(folder_path, video_names[index], filter_name)
            preprocess_calculate.calculate_final_result(folder_path, video_names[index], filter_name)

        # Combine cleaned data from different video
        combine_data.combine_preprocessed_files(folder_path, video_names, filter_name)

    # Analysis step1
    if ifDoAnalysis:
        model_name = "SVM"
        # model_name = "RandomForest"
        # model_name = "XGBoost"

        # feature = "dtw"  # not bad
        # feature = "velocity" # not bad
        feature = "min_max_velocity"  # not bad
        # feature = "movement_length"  # bad
        # feature = "direction"  # ok
        # feature = "same_direction_ratio"  # fine
        # feature = "avg_vector_angle"  # bad

        # feature = "dtw_velocity_related_direction_sdr"  # SVM: 61%, 140 sec | RF: 64%, 16 sec | XGBoost: 63%, 15 sec

        # feature = "dtw_velocity_related_direction_sdr_vecangle"  # SVM: 60%, 154 sec | RF: 64%, 21 sec | XGBoost: 65% (median), 64%(kalman), 59%(mean), 15 sec
        # feature = "dtw_velocity_related_direction_sdr_length"  # SVM: 60%, 441 sec | RF: 65%, 17 sec | XGBoost: 62%, 16 sec
        # feature = "dtw_velocity_related_direction_sdr_vecangle_length"  # SVM: 63%, 584 sec| RF: 65%, 19 sec | XGBoost: 64%, 15 sec

        ## Test
        # ml_model.machine_learning_main(folder_path, "Combined", filter_name, model_name, feature)
        # ml_model.machine_learning_cross_validation_test(folder_path, "Combined", filter_name, model_name, feature)

        ## Usual use
        ml_model.machine_learning_main_cv_ver(folder_path, "Combined", filter_name, model_name, feature)

        # CNN_model.deep_learning_main(folder_path, "Combined", filter_name, "1D-CNN", feature)
        # CNN_model.deep_learning_main_cv_ver(folder_path, "Combined", filter_name, "1D-CNN", feature)  # Under construction

        # LSTM_model.lstm_main(folder_path, "Combined", filter_name, "LSTM", feature)

        ## Combine bite and chase
        # ml_model.machine_learning_main_cv_3categories(folder_path, "Combined", filter_name, model_name, feature)
        # CNN_model.deep_learning_main_3categories(folder_path, "Combined", filter_name, "1D-CNN", feature)
        # LSTM_model.lstm_main_3categories(folder_path, "Combined", filter_name, "LSTM", feature)

        ## Add new feature? -> overlapping times


if __name__ == '__main__':
    main()
    