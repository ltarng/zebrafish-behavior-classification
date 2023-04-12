import pandas as pd
import numpy as np
import copy as cp
from typing import Tuple
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def getFeaturesData(feature, df):
    if feature == "dtw":
        X = np.vstack( df['DTW_distance'].to_numpy() )  # transform df['DTW_distance'] into a numpy 2D-array
    elif feature == "velocity":
        X = np.column_stack((df['Fish0_avg_velocity'], df['Fish1_avg_velocity']))
    elif feature == "dtw_and_velocity":
        X = np.column_stack((df['DTW_distance'], df['Fish0_avg_velocity'], df['Fish1_avg_velocity']))
    elif feature == "movement_length":
        X = np.column_stack((df['Fish0_movement_length'], df['Fish1_movement_length']))
    elif feature == "movement_length_difference":
        X = np.vstack( df['movement_length_differnece'].to_numpy() )
    elif feature == "movement_length_features":
        X = np.column_stack((df['Fish0_movement_length'], df['Fish1_movement_length'],df['movement_length_differnece']))
    else:
        X = np.column_stack((df['Fish0_avg_velocity'], df['Fish1_avg_velocity'], df['Fish0_movement_length'], df['Fish1_movement_length'], df['movement_length_differnece'], df['DTW_distance']))
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


def cross_val_predict(model, skfold: StratifiedKFold, X: np.array, y: np.array) -> Tuple[np.array, np.array, np.array]:
    # Reference: https://towardsdatascience.com/how-to-plot-a-confusion-matrix-from-a-k-fold-cross-validation-b607317e9874
    model_ = cp.deepcopy(model)
    no_classes = len(np.unique(y))
    
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

    return actual_classes, predicted_classes, predicted_proba


def plot_confusion_matrix(actual_classes, predicted_classes, sorted_labels, model_name, feature, save_folder):

    cm = confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)
    sns.heatmap(cm, square=True, annot=True, cmap='Blues', xticklabels=sorted_labels, yticklabels=sorted_labels)
    # sns.despine(left=False, right=False, top=False, bottom=False)

    # Text part
    plt.title('Confusion Matrix of ' + model_name + ' Classifier' +"\n" + 'Used Feature: ' + feature)
    plt.xlabel("Predicted Value")
    plt.ylabel("True Value")
    plt.tight_layout()

    plt.savefig(save_folder + model_name + "_" + feature + "_confusion_matrix.png")
    plt.show()


def machine_learning_main(folder_path, video_name, filter_name, model_name, feature):
    # Read preprocessed trajectory data
    # df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_preprocessed_result.csv')
    df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_preprocessed_result_remove_circle.csv')

    # Split data into trainging data and testing data
    X = getFeaturesData(feature, df)
    y = df['BehaviorType']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=54, stratify=y)

    # Select model and training
    if model_name == "SVM":
        kernel_name = choose_SVC_kernel_model()
        model_name = model_name + '-' + kernel_name
        model = SVC(kernel=kernel_name)
    elif model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=1000)
    else:
        print("Wrong model name! Please input 'SVM' or 'RandomForest'.")

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
    # sorted_labels = ['bite', 'chase', 'circle', 'display', 'normal']
    sorted_labels = ['bite', 'chase', 'display', 'normal']
    plot_confusion_matrix(y_test, predictions, sorted_labels, model_name, feature, save_folder)


def machine_learning_main_cv_ver(folder_path, video_name, filter_name, model_name, feature):
    # Read preprocessed trajectory data
    # df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_preprocessed_result.csv')
    df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_preprocessed_result_remove_circle.csv')

    # Split data into trainging data and testing data
    X = getFeaturesData(feature, df)
    y = df['BehaviorType']

    # Select model and training
    if model_name == "SVM":
        kernel_name = choose_SVC_kernel_model()
        model_name = model_name + '-' + kernel_name
        model = SVC(kernel=kernel_name)
    elif model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=1000)
    else:
        print("Wrong model name! Please input 'SVM' or 'RandomForest'.")

    # 10-fold  cross validation
    skfold = StratifiedKFold(n_splits=10, random_state=99, shuffle=True)  # Split data evenly among different behavior data type
    actual_classes, predicted_classes, _ = cross_val_predict(model, skfold, X, y)

    # Show the testing result with confusion matrix
    print(model_name, feature)
    print(classification_report(actual_classes, predicted_classes))

    # Setting the path for saving confusion matrix pictures
    save_folder = folder_path + "confusion_matrix_picture/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Plot the confusion matrix graph on screen, and save it in png format
    # sorted_labels = ['bite', 'chase', 'circle', 'display', 'normal']
    sorted_labels = ['bite', 'chase', 'display', 'normal']
    plot_confusion_matrix(actual_classes, predicted_classes, sorted_labels, model_name, feature, save_folder)


def machine_learning_cross_validation_test(folder_path, video_name, filter_name, model_name, feature):
    # Read preprocessed trajectory data
    # df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_preprocessed_result.csv')
    df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_preprocessed_result_remove_circle.csv')

    # Split data into trainging data and testing data
    X = getFeaturesData(feature, df)
    y = df['BehaviorType']

    # Select model and training
    if model_name == "SVM":
        kernel_name = choose_SVC_kernel_model()
        model_name = model_name + '-' + kernel_name
        model = SVC(kernel=kernel_name)
    elif model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=1000)
    else:
        print("Wrong model name! Please input 'SVM' or 'RandomForest'.")

    skfold = StratifiedKFold(n_splits=10, random_state=99, shuffle=True)

    cv_results = cross_val_score(model, X, y, cv=skfold, scoring='accuracy', verbose=10)
    print("Mean accuracy: ", cv_results.mean(), ", standard deviation", cv_results.std())
