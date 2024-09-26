# Zebrafish Behavior Classification System
A system for thesis.

## Pakage List
| Pakage Name | version |
|--------:|:---------|
| Python | 3.10.7 |
| pandas | 2.0.3 |
| numpy  | 1.23.5 |
| scipy  | 1.11.1 |
| pykalman| 0.9.5 |
| dtaidistance| 2.3.10 |
| scikit-learn| 1.2.0 |
| matplotlib | 3.7.1 |
| plotly | 5.15.0 |
| seaborn | 0.12.2 |
| progress | 1.6 |
| tensorflow | 2.12.0 |
| tensorboard | 2.12.3 |
| opencv-python | 4.8.0.74 |
| xgboost | 1.7.6 |

## Parameters May Need to Modify by Manual
### main.py
#### 1. General Setting
- ```video_names```: A list of video names (e.g. ```['1-14', '1-22_2nd']```).
- ```filter_name```: Specifies the filter type to apply. Options include ```"mean"```, ```"median"```, ```"kalman"```, and ```"nofilter"```.


#### 2. System Execution Setting
- ```execute_steps```: A dictionary that controls which of the following steps to run, using boolean flags:
    - ```"data_cleaning"```: Executes data cleaning if set to ```True```.
    - ```"annotation"```: Adds annotations if set to ```True```.
    - ```"preprocessing"```: Performs data preprocessing if set to ```True```.
    - ```"analysis"```: Runs data analysis if set to ```True```.
- Variables within the ```"data_cleaning"``` step:
    - ```if_plot_traj```: Boolean flag to indicate whether to plot graphs comparing trajectories before and after filtering.
- Variables within the ```"preprocessing"``` step:
    - ```skip_semi```: Boolean flag to determine if semi-calculation steps should be skipped.
- Variables within the ```"analysis"``` step:
    - ```do_tuning```: Boolean flag to indicate whether to perform hyperparameter tuning.
    - ```do_training```: Boolean flag to indicate whether to train the model.
    - ```model_name```: Specifies the model to use. Options are ```"SVM"```, ```"RandomForest"```, and ```"XGBoost"```.
    - ```feature```: Specifies the feature combination to use. Options include: 
        - ```"dtw_velocities_direction_sdr_partangles_length"```
        - ```"dtw_velocities_direction_sdr_angles_length"```
    - ```class_amount```: Integer value specifying the number of behavior categories. Options are ```2```, ```3```, and ```4```.

-----

### machine_learning_pipeline.py
- ```k_num```: Integer (e.g., ```5``` or ```10```),  representing the **"k"** value for ***k***-fold cross-validation.

-----

### preprocess_calculate.py *(optional)*
*Modification is only necessary if the amount of features changes.*
#### 1. Function ```normalize_preprocessed_data()```
- ```start_col```: Integer indicating the starting column.
- ```end_col```: Integer indicating the ending column.