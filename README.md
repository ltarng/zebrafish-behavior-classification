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
#### General Setting
- folder_path: a string format (for example: "D:/Research/Trajectory Analysis/")
- video_names: a list format (for example: ['1-14', '1-22_2nd'])
- filter_name: mean, median, kalman, nofilter

#### System Execution Setting
- ifDoDataCleaning: a boolean (for example: True)
- ifDoAnnotate: boolean
- ifDoPreprocess: boolean
- ifDoAnalysis: boolean
- ifPlotTraj: boolean
- ifSkipSemi: boolean
- ifDoTuning: boolean
- ifDoTraining: boolean

- class_num: 3, 4 (number of behavior categories)
- model_name: SVM, RandomForest, XGBoost
- feature: (depends on which features you would like to use, please refer to the function getFeaturesData() in ml_model.py)

### ml_model.py
#### hyperparameter_tuning()
- params: depends on what combinations you would like to try

#### machine_learning_main_cv_ver()
- k_num: integer, such as 5 or 10 ("k" for k-fold cross validation)

### preprocess_calculate.py [only when you change the amount of features]
#### normalize_preprocessed_data()
- start_col: integer
- end_col: integer