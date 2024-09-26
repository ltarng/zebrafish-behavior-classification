import pandas as pd
import numpy as np
import copy as cp
from typing import Tuple
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from time import process_time
from feature_extraction import getFeaturesData
from modeling import getModel
from feature_importance import plot_feature_importance
from evaluation import plot_confusion_matrix, print_classification_report
from io_utils import create_saving_folder_for_confusion_matrix, prepare_training_dataframe

"""Set the number of folds for K-fold cross-validation"""
k_num = 5


PARAM_GRID = {
    "SVM": [{'kernel': ['linear'], 'C': [1, 5, 10, 20, 30, 40, 50]}, 
            {'kernel': ['rbf'], 'gamma': ['scale', 'auto']}],
    "RandomForest": {
        'n_estimators': [500, 600, 700, 800, 900, 1000],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': list(range(6, 16)),
        'criterion': ['gini', 'entropy']
    },
    "XGBoost": {
        'max_depth': [8, 9],
        'learning_rate': [0.1, 0.15, 0.2, 0.25, 0.3],
        'n_estimators': [150, 200, 250, 300],
        'colsample_bytree': [0.3, 0.5, 0.7],
        'gamma': list(range(6))
    }
}


def hyperparameter_tuning(folder_path, video_name, filter_name, model_name, feature, class_amount):
    df = prepare_training_dataframe(folder_path, video_name, filter_name, class_amount)

    # Split data into trainging data and testing data
    X = getFeaturesData(feature, df)
    y = df['BehaviorType']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=54, stratify=y)

    # Define value of parameters for Grid Search
    if model_name in PARAM_GRID:
        params = PARAM_GRID[model_name]
        model = getModel(model_name)[0]
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Do Grid Search, seeking the best combination of parameters
    skfold = StratifiedKFold(n_splits=k_num, random_state=99, shuffle=True)  # For N-fold cross-validation
    clf = GridSearchCV(estimator=model, 
                       param_grid=params, 
                       scoring='accuracy', 
                       cv=skfold.split(X_train, y_train), 
                       verbose=1, 
                       n_jobs=3)
    grid_result = clf.fit(X_train, y_train)
    
    # Show the result of all hyper-parameter combination
    print(f"Best accuracy: {grid_result.best_score_}, best parameter combination: {grid_result.best_params_}")
    for mean, stdev, param in zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score'], grid_result.cv_results_['params']):
        print(f"Mean accuracy: {mean}, standard deviation: {stdev}, parameter combination: {param}")
    print("Best parameters:", grid_result.best_params_)


def cross_val_predict(model, chosen_model, feature, skfold: StratifiedKFold, X: np.array, y: np.array) -> Tuple[np.array, np.array, np.array]:
    # Reference: https://towardsdatascience.com/how-to-plot-a-confusion-matrix-from-a-k-fold-cross-validation-b607317e9874
    model_ = cp.deepcopy(model)
    no_classes = len(np.unique(y))

    pre = model_
    
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes]) 

    for i in range(0, 50):  # Setting how many rounds to do training. For example, range(0, 50) is to observe total training result for 50 rounds
        for train_ndx, test_ndx in skfold.split(X, y):

            train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

            actual_classes = np.append(actual_classes, test_y)
            model_.fit(train_X, train_y)
            predicted_classes = np.append(predicted_classes, model_.predict(test_X))

            try:
                predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
            except:
                predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    # Plot feature importance of the model
    plot_feature_importance(model_, chosen_model, feature)

    return actual_classes, predicted_classes, predicted_proba


def machine_learning_main_cv_ver(folder_path, video_name, filter_name, chosen_model, feature, class_amount):  # cross-validation version
    # Read preprocessed trajectory data
    df = prepare_training_dataframe(folder_path, video_name, filter_name, class_amount)

    # Split data into trainging data and testing data
    X = getFeaturesData(feature, df)
    y = df['BehaviorType']

    # Select model
    model, model_name = getModel(chosen_model)

    # k-fold  cross validation
    skfold = StratifiedKFold(n_splits=k_num, shuffle=True)  # Split data evenly among different behavior data type
    start_time = process_time()
    actual_classes, predicted_classes, _ = cross_val_predict(model, chosen_model, feature, skfold, X, y)
    end_time = process_time()

    # If it is 2 or 3 categories, rename model name
    if class_amount == 3:
        model_name = model_name + "_3categories"
    elif class_amount == 2:
        model_name = model_name + "_2categories"

    # Show the testing result with confusion matrix
    print(model_name, feature)
    print(print_classification_report(actual_classes, predicted_classes))

    # Setting sorted label and print the labels on screen
    if class_amount == 3:
        sorted_labels = ['bite & chase', 'display', 'normal']
    elif class_amount == 2:
        sorted_labels = ['abnormal', 'normal']
    else:  # 4 categories
        sorted_labels = ['bite', 'chase', 'display', 'normal']
    print("Class label 0 ~", class_amount," means: ", sorted_labels)

    # Setting the path and create a folder to save confusion matrix pictures
    save_folder = create_saving_folder_for_confusion_matrix(folder_path)

    # Plot the confusion matrix graph on screen, and save it in png format
    plot_confusion_matrix(actual_classes, predicted_classes, sorted_labels, model_name+"_"+str(k_num)+"cv", filter_name, feature, save_folder)
    print("Execution time: ", end_time - start_time)


def machine_learning_main(folder_path, video_name, filter_name, chosen_model, feature):  # for test
    # Read preprocessed trajectory data
    df = pd.read_csv(folder_path + "combined_preprocessed_data/" + video_name + '_' + filter_name + '_preprocessed_result.csv')

    # Split data into trainging data and testing data
    X = getFeaturesData(feature, df)
    y = df['BehaviorType']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=54, stratify=y)

    # Select model and training
    model, model_name = getModel(chosen_model)
    model.fit(X_train, y_train)

    # Show the testing result with confusion matrix
    predictions = model.predict(X_test)
    print(model_name, feature)
    print(print_classification_report(y_test, predictions))

    # Setting the path for saving confusion matrix pictures
    save_folder = create_saving_folder_for_confusion_matrix(folder_path)
    
    # Plot the confusion matrix graph on screen, and save it in png format
    sorted_labels = ['bite', 'chase', 'display', 'normal']
    plot_confusion_matrix(y_test, predictions, sorted_labels, model_name, filter_name, feature, save_folder)
