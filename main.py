import annotate_training_data
import preprocess_calculate
import data_cleaning
import machine_learning_pipeline
import combine_data
import os


""" GENERAL PARAMETER SETTING """
def get_folder_path():
    """Returns the current folder path with normalized separators."""
    return os.getcwd().replace('\\', '/') + '/'


folder_path = get_folder_path()

# Video data and filter settings
video_names = ['1-14', '1-22_2nd']
filter_name = "median"  # Options: "mean", "median", "kalman", "nofilter"


def data_cleaning_step(if_plot_traj):
    """Executes the data cleaning step for each video."""
    for video in video_names:
        data_cleaning.data_cleaning(folder_path, video, filter_name, if_plot_traj)


def annotation_step():
    """Executes the data annotation step for each video."""
    for video in video_names:
        annotate_training_data.auto_annotate(folder_path, video, filter_name)
        annotate_training_data.sort_annotation_information(folder_path, video)


def preprocessing_step(skip_semi):
    """Preprocessing calculations, feature extraction, and data combination."""
    for video in video_names:
        if not skip_semi:
            preprocess_calculate.calculate_semifinished_result(folder_path, video, filter_name)
        preprocess_calculate.calculate_final_result(folder_path, video, filter_name)
        preprocess_calculate.normalize_and_save(folder_path, video, filter_name)

    # Combine preprocessed data from different videos
    combine_data.combine_preprocessed_main(folder_path, video_names, filter_name)


def analysis_step(do_tuning, do_training, model_name, feature, class_amount):
    """
    Performs model analysis, including tuning and training.

    :param do_tuning: If True, execute hyperparameter tuning.
    :param do_training: If True, execute model training.
    :param model_name: The name of the machine learning model to use.
    :param feature: The feature set to use for the model.
    :param class_amount: The number of classes for classification. 
    """
    if do_tuning:
        machine_learning_pipeline.hyperparameter_tuning(folder_path, "Combined", filter_name, model_name, feature, class_amount)
        # ml_model.hyperparameter_tuning(folder_path, "Combined_nor", filter_name, model_name, feature, class_amount)

    if do_training:
        machine_learning_pipeline.machine_learning_main_cv_ver(folder_path, "Combined", filter_name, model_name, feature, class_amount)
        # ml_model.machine_learning_main_cv_ver(folder_path, "Combined_nor", filter_name, model_name, feature, class_amount)  # Use normalized raw data


def main():
    """Main execution entry point."""
    # Execution control flags
    execute_steps = {
        "data_cleaning": True,
        "annotation": True,
        "preprocessing": True,
        "analysis": True
    }

    print(f"Working directory: {folder_path}")

    # Execute data cleaning step
    if execute_steps["data_cleaning"]:
        data_cleaning_step(if_plot_traj=False)

    # Execute annotation step
    if execute_steps["annotation"]:
        annotation_step()

    # Execute preprocessing step
    if execute_steps["preprocessing"]:
        preprocessing_step(skip_semi=False)

    # Execute analysis step
    if execute_steps["analysis"]:
        analysis_step(
            do_tuning=False, 
            do_training=True, 
            model_name="SVM",  # Options: 'SVM', 'RandomForest', 'XGBoost'
            feature="dtw_velocities_direction_sdr_partangles_length",   # Options: "dtw_velocities_direction_sdr_partangles_length", "dtw_velocities_direction_sdr_angles_length"
            class_amount=2  # option: 2 (normal/abnormal), 3 (high/moderate/low aggression), 4 (bite/chase/display/normal)
        )


if __name__ == '__main__':
    main()
    