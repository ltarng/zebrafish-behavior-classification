import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(confusion_matrix, pic_name):
    classes = ['bite', 'chase', 'circle', 'display', 'normal']
    sns.heatmap(confusion_matrix, square=True, annot=True, cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    # sns.despine(left=False, right=False, top=False, bottom=False)

    plt.title('Confusion Matrix of the Classifier')
    plt.xlabel("Predicted Value")
    plt.ylabel("True Value")
    plt.tight_layout()

    plt.savefig(pic_name)
    plt.show()


def svm(folder_path, video_name, filter_name):
    # Read DTW result file
    df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_preprocessed_result.csv')
    # feature = "dtw"
    # feature = "velocity"
    feature = "both"

    # Divide data into trainging data and testing data
    if feature == "dtw":
        X = np.vstack( df['DTW_distance'].to_numpy() )  # transform df['DTW_distance'] into a numpy 2D-array
        pic_name = "SVM_dtw_feature_confusion_matrix.png"
    elif feature == "velocity":
        X = np.column_stack((df['avg_velocity_fish0'], df['avg_velocity_fish1']))
        pic_name = "SVM_velocity_feature_confusion_matrix.png"
    else:
        X = np.column_stack((df['avg_velocity_fish0'], df['avg_velocity_fish1'], df['DTW_distance']))
        pic_name = "SVM_both_feature_confusion_matrix.png"
    y = df['BehaviorType']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=99)

    # Use Support Vector Classifier to build model
    model = SVC()
    model.fit(X_train, y_train)

    # Use test data to test the result of model
    predictions = model.predict(X_test)

    # Load classification_report & confusion_matrix to estimate the performance of model
    cm = confusion_matrix(y_test, predictions)

    # Show the SVM result
    print(classification_report(y_test, predictions))
    plot_confusion_matrix(cm, pic_name)


def random_forest(folder_path, video_name, filter_name):
    # Read DTW result file
    df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_preprocessed_result.csv')
    # feature = "dtw"
    feature = "velocity"
    # feature = "both"

    # Divide data into trainging data and testing data
    if feature == "dtw":
        X = np.vstack( df['DTW_distance'].to_numpy() )  # transform df['DTW_distance'] into a numpy 2D-array
        pic_name = "RF_dtw_feature_confusion_matrix.png"
    elif feature == "velocity":
        X = np.column_stack((df['avg_velocity_fish0'], df['avg_velocity_fish1']))
        pic_name = "RF_velocity_feature_confusion_matrix.png"
    else:
        X = np.column_stack((df['avg_velocity_fish0'], df['avg_velocity_fish1'], df['DTW_distance']))
        pic_name = "RF_both_feature_confusion_matrix.png"
    y = df['BehaviorType']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=99)

    # Use Random Forest Classifier to build model
    model = RandomForestClassifier(n_estimators=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    # Show confusion matrix
    print(classification_report(y_test, predictions))
    plot_confusion_matrix(cm, pic_name)
