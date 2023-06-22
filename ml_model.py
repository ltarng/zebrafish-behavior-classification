import pandas as pd
import numpy as np
import copy as cp
from typing import Tuple
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from time import process_time


def getFeaturesData(feature, df):
    # Single Feature
    if feature == "dtw":
        X = np.vstack( df['DTW_distance'].to_numpy() )  # transform df['DTW_distance'] into a numpy 2D-array
    elif feature == "velocity":
        X = np.column_stack((df['Fish0_avg_velocity'], df['Fish1_avg_velocity']))
    elif feature == "min_max_velocity":
        X = np.column_stack((df['Fish0_max_velocity'], df['Fish1_max_velocity'], df['Fish0_min_velocity'], df['Fish1_min_velocity']))
    elif feature == "movement_length":
        X = np.column_stack((df['Fish0_movement_length'], df['Fish1_movement_length']))
    elif feature == "movement_length_difference":
        X = np.vstack( df['movement_length_differnece'].to_numpy() )
    elif feature == "direction":
        X = np.column_stack((df['Fish0_moving_direction_x'], df['Fish0_moving_direction_y'], df['Fish1_moving_direction_x'], df['Fish1_moving_direction_y']))
    elif feature == "same_direction_ratio":
        X = np.vstack( df['same_direction_ratio'].to_numpy() )
    elif feature == "avg_vector_angle":
        X = np.vstack( df['avg_vector_angle'].to_numpy() )
    elif feature == "min_max_vector_angle":
        X = np.column_stack((df['min_vector_angle'], df['max_vector_angle']))
    # Combined Features
    elif feature == "dtw_velocity_related_direction_sdr":
        X = np.column_stack((df['Fish0_avg_velocity'], df['Fish1_avg_velocity'], df['DTW_distance'], 
                             df['Fish0_max_velocity'], df['Fish1_max_velocity'], df['Fish0_min_velocity'], df['Fish1_min_velocity'], 
                             df['Fish0_moving_direction_x'], df['Fish0_moving_direction_y'], df['Fish1_moving_direction_x'], df['Fish1_moving_direction_y'], 
                             df['same_direction_ratio']))
    elif feature == "dtw_velocity_related_direction_sdr_vecangle":
        X = np.column_stack((df['Fish0_avg_velocity'], df['Fish1_avg_velocity'], df['DTW_distance'], 
                             df['Fish0_max_velocity'], df['Fish1_max_velocity'], df['Fish0_min_velocity'], df['Fish1_min_velocity'], 
                             df['Fish0_moving_direction_x'], df['Fish0_moving_direction_y'], df['Fish1_moving_direction_x'], df['Fish1_moving_direction_y'], 
                             df['same_direction_ratio'], df['avg_vector_angle']))
    elif feature == "dtw_velocity_related_direction_sdr_length":
        X = np.column_stack((df['Fish0_avg_velocity'], df['Fish1_avg_velocity'], df['DTW_distance'], 
                             df['Fish0_movement_length'], df['Fish1_movement_length'], 
                             df['Fish0_max_velocity'], df['Fish1_max_velocity'], df['Fish0_min_velocity'], df['Fish1_min_velocity'], 
                             df['Fish0_moving_direction_x'], df['Fish0_moving_direction_y'], df['Fish1_moving_direction_x'], df['Fish1_moving_direction_y'], 
                             df['same_direction_ratio']))
    elif feature == "dtw_velocity_related_direction_sdr_vecangle_length":
        X = np.column_stack((df['Fish0_avg_velocity'], df['Fish1_avg_velocity'], df['DTW_distance'], 
                             df['Fish0_movement_length'], df['Fish1_movement_length'], 
                             df['Fish0_max_velocity'], df['Fish1_max_velocity'], df['Fish0_min_velocity'], df['Fish1_min_velocity'], 
                             df['Fish0_moving_direction_x'], df['Fish0_moving_direction_y'], df['Fish1_moving_direction_x'], df['Fish1_moving_direction_y'], 
                             df['same_direction_ratio'], df['avg_vector_angle']))
    else:
        print("Error: feature name does not exist.")
    return X


def choose_SVC_kernel_model():
    kernel_num = input("Choose kernel function - 1:linear, 2:poly, 3:rbf, 4:sigmoid \n")
    if kernel_num == '1':
        kernel_name = 'linear'
    elif kernel_num == '2':
        kernel_name = 'poly'
    elif kernel_num == '3':
        kernel_name = 'rbf'
    elif kernel_num == '4':
        kernel_name = 'sigmoid'
    else:
        print("Your option is not in the list. Please choose again.")
        choose_SVC_kernel_model()
    return kernel_name


def getModel(chosen_model):
    # Select model and training
    if chosen_model == "SVM":
        kernel_name = choose_SVC_kernel_model()
        model_name = chosen_model + '-' + kernel_name
        model = SVC(kernel=kernel_name)
    elif chosen_model == "DecisionTree":
        model_name = "Decision Tree"
        model = DecisionTreeClassifier()
    elif chosen_model == "RandomForest":
        model_name = "Random Forest"
        model = RandomForestClassifier(n_estimators=1000)
        # model = RandomForestClassifier(n_estimators=2000)
    elif chosen_model == "XGBoost":
        model_name = "XGBoost"
        # model = xgb.sklearn.XGBClassifier(max_depth=6, n_estimators=100)  # default
        # model = xgb.sklearn.XGBClassifier(max_depth=6, n_estimators=70)
        model = xgb.sklearn.XGBClassifier()
    else:
        print("Wrong model name! Please input 'SVM' or 'RandomForest'.")
    return model, model_name


def cross_val_predict(model, skfold: StratifiedKFold, X: np.array, y: np.array) -> Tuple[np.array, np.array, np.array]:
    # Reference: https://towardsdatascience.com/how-to-plot-a-confusion-matrix-from-a-k-fold-cross-validation-b607317e9874
    model_ = cp.deepcopy(model)
    no_classes = len(np.unique(y))

    pre = model_
    
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes]) 

    for train_ndx, test_ndx in skfold.split(X, y):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        actual_classes = np.append(actual_classes, test_y)
        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))

        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    print(model_)
    # # xgb.plot_importance(model_)
    # importances = model_.feature_importances_
    # print('每個特徵重要程度: ', importances)
    # feature_names = [f"feature {i}" for i in range(X.shape[1])]
    # std = np.std([tree.feature_importances_ for tree in model_.estimators_], axis=0)

    # forest_importances = pd.Series(importances, index=feature_names)

    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=std, ax=ax)
    # ax.set_title("Feature importances using MDI")
    # ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    # plt.show()

    return actual_classes, predicted_classes, predicted_proba


def create_saving_folder(folder_path):
    # Setting the path for saving confusion matrix pictures
    save_folder = folder_path + "confusion_matrix_picture/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    return save_folder


def plot_confusion_matrix(actual_classes, predicted_classes, sorted_labels, model_name, filter_name, feature, save_folder):
    cm = confusion_matrix(actual_classes, predicted_classes)
    sns.heatmap(cm, square=True, annot=True, cmap='Blues', xticklabels=sorted_labels, yticklabels=sorted_labels)
    # sns.despine(left=False, right=False, top=False, bottom=False)

    # Text part
    plt.title('Confusion Matrix of ' + model_name + ' Classifier' + "\n" + 
              'Filter Name: ' + filter_name + "\n" +
              'Used Feature: ' + feature)
    plt.xlabel("Predicted Value")
    plt.ylabel("True Value")
    plt.tight_layout()

    plt.savefig(save_folder + model_name + "_" + filter_name + "_" + feature + "_confusion_matrix.png")
    plt.show()


def machine_learning_main(folder_path, video_name, filter_name, chosen_model, feature):
    # Read preprocessed trajectory data
    df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_preprocessed_result.csv')

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
    print(classification_report(y_test, predictions))

    # Setting the path for saving confusion matrix pictures
    save_folder = folder_path + "confusion_matrix_picture/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Plot the confusion matrix graph on screen, and save it in png format
    sorted_labels = ['bite', 'chase', 'display', 'normal']
    plot_confusion_matrix(y_test, predictions, sorted_labels, model_name, filter_name, feature, save_folder)


def machine_learning_main_cv_ver(folder_path, video_name, filter_name, chosen_model, feature):
    # Read preprocessed trajectory data
    df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_preprocessed_result.csv')

    # Split data into trainging data and testing data
    X = getFeaturesData(feature, df)
    y = df['BehaviorType']

    # Select model
    model, model_name = getModel(chosen_model)

    # 10-fold  cross validation
    skfold = StratifiedKFold(n_splits=10, random_state=99, shuffle=True)  # Split data evenly among different behavior data type
    start_time = process_time()
    actual_classes, predicted_classes, _ = cross_val_predict(model, skfold, X, y)
    end_time = process_time()

    # Show the testing result with confusion matrix
    print(model_name, feature)
    print(classification_report(actual_classes, predicted_classes))
    print("Class number meaning - 0:bite, 1:chase, 2:display, 3:normal")

    # # Setting the path and create a folder to save confusion matrix pictures
    # save_folder = create_saving_folder(folder_path)

    # # Plot the confusion matrix graph on screen, and save it in png format
    # sorted_labels = ['bite', 'chase', 'display', 'normal']
    # plot_confusion_matrix(actual_classes, predicted_classes, sorted_labels, model_name, filter_name, feature, save_folder)
    # print("Execution time: ", end_time - start_time)


def machine_learning_main_cv_std(folder_path, video_name, filter_name, chosen_model, feature):
    # Read preprocessed trajectory data
    df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_preprocessed_result_std.csv')

    # Split data into trainging data and testing data
    X = getFeaturesData(feature, df)
    y = df['BehaviorType']

    # Select model
    model, model_name = getModel(chosen_model)

    # 10-fold  cross validation
    skfold = StratifiedKFold(n_splits=10, random_state=99, shuffle=True)  # Split data evenly among different behavior data type
    start_time = process_time()
    actual_classes, predicted_classes, _ = cross_val_predict(model, skfold, X, y)
    end_time = process_time()

    # Show the testing result with confusion matrix
    print(model_name, feature)
    print(classification_report(actual_classes, predicted_classes))
    print("Class number meaning - 0:bite, 1:chase, 2:display, 3:normal")


def machine_learning_cross_validation_test(folder_path, video_name, filter_name, chosen_model, feature):
    # Read preprocessed trajectory data
    df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_preprocessed_result.csv')

    # Split data into trainging data and testing data
    X = getFeaturesData(feature, df)
    y = df['BehaviorType']

    # Select model and training
    model = getModel(chosen_model)

    skfold = StratifiedKFold(n_splits=10, random_state=99, shuffle=True)

    cv_results = cross_val_score(model, X, y, cv=skfold, scoring='accuracy', verbose=10)
    print("Mean accuracy: ", cv_results.mean(), ", standard deviation", cv_results.std())


def machine_learning_main_cv_3categories(folder_path, video_name, filter_name, chosen_model, feature):
    # Read preprocessed trajectory data
    df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_preprocessed_result_half_bite_chase.csv')

    # Split data into trainging data and testing data
    X = getFeaturesData(feature, df)
    y = df['BehaviorType']

    # Combine bite and chase, renumber all number of class type
    for index in range(0, len(df['BehaviorType'].index)):
        if df['BehaviorType'].iloc[index] == 0 or df['BehaviorType'].iloc[index] == 1:  # bite and chase
            df['BehaviorType'].iloc[index] = 0
        elif df['BehaviorType'].iloc[index] == 2:  # display
            df['BehaviorType'].iloc[index] = 1
        elif df['BehaviorType'].iloc[index] == 3:  # normal
            df['BehaviorType'].iloc[index] = 2
        else:
            print("Index of BehaviorType is not 0, 1, 2 or 3.")

    # Select model
    model, model_name = getModel(chosen_model)

    # 10-fold  cross validation
    skfold = StratifiedKFold(n_splits=5, random_state=99, shuffle=True)  # Split data evenly among different behavior data type
    start_time = process_time()
    actual_classes, predicted_classes, _ = cross_val_predict(model, skfold, X, y)
    end_time = process_time()

    # Model name
    model_name = model_name + "_3categories_only"

    # Show the testing result with confusion matrix
    print(model_name, feature)
    print(classification_report(actual_classes, predicted_classes))
    print("Class number meaning - 0:bite&chase, 1:display, 2:normal")

    # Setting the path and create a folder to save confusion matrix pictures
    save_folder = create_saving_folder(folder_path)

    # Plot the confusion matrix graph on screen, and save it in png format
    sorted_labels = ['bite & chase', 'display', 'normal']
    plot_confusion_matrix(actual_classes, predicted_classes, sorted_labels, model_name+"_5cv_", filter_name, feature, save_folder)
    print("Execution time: ", end_time - start_time)
