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


def machine_learning_main(folder_path, video_name, filter_name, model_name, feature):
    # Read preprocessed trajectory data
    df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_preprocessed_result.csv')

    # Divide data into trainging data and testing data
    if feature == "dtw":
        X = np.vstack( df['DTW_distance'].to_numpy() )  # transform df['DTW_distance'] into a numpy 2D-array
        pic_name = "RF_dtw_feature_confusion_matrix.png"
    elif feature == "velocity":
        X = np.column_stack((df['avg_velocity_fish0'], df['avg_velocity_fish1']))
        pic_name = "RF_velocity_feature_confusion_matrix.png"
    elif feature == "movement_length":
        X = np.column_stack((df['movement_length_fish0'], df['movement_length_fish1']))
        pic_name = "RF_movement_length_feature_confusion_matrix.png"
    elif feature == "movement_length_difference":
        X = np.vstack( df['movement_length_differnece'].to_numpy() )
        pic_name = "RF_movement_length_diff_feature_confusion_matrix.png"
    elif feature == "movement_length_features":
        X = np.column_stack((df['movement_length_fish0'], df['movement_length_fish1'],df['movement_length_differnece']))
        pic_name = "RF_movementlength_all_feature_confusion_matrix.png"
    else:
        X = np.column_stack((df['avg_velocity_fish0'], df['avg_velocity_fish1'], df['movement_length_fish0'], df['movement_length_fish1'], df['movement_length_differnece'], df['DTW_distance']))
        pic_name = "RF_all_feature_confusion_matrix.png"

    # Split training data
    y = df['BehaviorType']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=99)

    # Training
    if model_name == "SVM":
        model = SVC()
    elif model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=1000)
    else:
        print("Wrong model name! Please input 'SVM' or 'RandomForest'.")

    model.fit(X_train, y_train)

    # Testing
    predictions = model.predict(X_test)

    # Show result and confusion matrix graph
    print(classification_report(y_test, predictions))
    cm = confusion_matrix(y_test, predictions)
    plot_confusion_matrix(cm, pic_name)
