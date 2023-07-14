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
from sklearn.model_selection import GridSearchCV

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
    elif feature == "dtw_velocities_direction_sdr_angles_length":
        X = np.column_stack((df['Fish0_avg_velocity'], df['Fish1_avg_velocity'], df['DTW_distance'], 
                             df['Fish0_movement_length'], df['Fish1_movement_length'], 
                             df['Fish0_max_velocity'], df['Fish1_max_velocity'], df['Fish0_min_velocity'], df['Fish1_min_velocity'], 
                             df['Fish0_moving_direction_x'], df['Fish0_moving_direction_y'], df['Fish1_moving_direction_x'], df['Fish1_moving_direction_y'], 
                             df['same_direction_ratio'], df['avg_vector_angle'], df['min_vector_angle'], df['max_vector_angle']))
        feature_names = ['AVG1', 'AVG2, DTW',
                         'Dist1', 'Dist2',
                         'max(D1)', 'max(D2)', 'min(D1)', 'min(D2)',
                         'V1_x', 'V1_y', 'V2_x', 'V2_y',
                         'SDR', 'avg(A)', 'min(A)', 'max(A)']
    elif feature == "dtw_velocities_direction_sdr_partangles_length":
        X = np.column_stack((df['Fish0_avg_velocity'], df['Fish1_avg_velocity'], df['DTW_distance'], 
                             df['Fish0_movement_length'], df['Fish1_movement_length'], 
                             df['Fish0_max_velocity'], df['Fish1_max_velocity'], df['Fish0_min_velocity'], df['Fish1_min_velocity'], 
                             df['Fish0_moving_direction_x'], df['Fish0_moving_direction_y'], df['Fish1_moving_direction_x'], df['Fish1_moving_direction_y'], 
                             df['same_direction_ratio'], df['min_vector_angle'], df['max_vector_angle']))
        feature_names = ['AVG1', 'AVG2, DTW',
                         'Dist1', 'Dist2',
                         'max(D1)', 'max(D2)', 'min(D1)', 'min(D2)',
                         'V1_x', 'V1_y', 'V2_x', 'V2_y',
                         'SDR', 'min(A)', 'max(A)']
    elif feature == "vecangle_related":
        X = np.column_stack(( df['same_direction_ratio'],df['avg_vector_angle'],df['min_vector_angle'], df['max_vector_angle']))
    else:
        print("Error: feature name does not exist.")
    return X


def getModel(chosen_model):
    # Select model and training
    if chosen_model == "SVM":
        kernel_name = "linear"
        model_name = chosen_model + '-' + kernel_name
        model = SVC(C=0.01, kernel=kernel_name)  # not normalization
        # model = SVC(C=20, kernel=kernel_name)  # normalization data
    elif chosen_model == "DecisionTree":
        model_name = "Decision Tree"
        model = DecisionTreeClassifier()
    elif chosen_model == "RandomForest":
        model_name = "Random Forest"
        # model = RandomForestClassifier(max_depth=13, n_estimators=900, criterion="gini", max_features='sqrt')  # 63%|65%(14sec), 78%(14.6sec) | 65%(29.1sec), 78%(26.9sec)
        model = RandomForestClassifier(max_depth=13, n_estimators=900, criterion="gini", max_features='log2')  # best 4cat by GSCV, 65%(14sec), 78%(13.5sec) | 66%(28.6sec), 78%(25.2sec)
        # model = RandomForestClassifier(max_depth=6, n_estimators=900, criterion="gini", max_features='log2')  # normalization for 4cat
    elif chosen_model == "XGBoost":
        model_name = "XGBoost"
        # model = xgb.sklearn.XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=100, colsample_bytree=1)  # default
        model = xgb.sklearn.XGBClassifier(colsample_bytree=0.5, learning_rate=0.3, max_depth=9, n_estimators=200)  # 66%(m), 77%(m), 65%(no), 78%(no)
        # model = xgb.sklearn.XGBClassifier(colsample_bytree=0.3, learning_rate=0.15, max_depth=6, n_estimators=250, gamma=3)  # 65%, 77% | 65%, 78%
    else:
        print("Wrong model name! Please check the variable 'model_name' in main.py.")
    return model, model_name


def hyperparameter_tuning(folder_path, video_name, filter_name, model_name, feature, class_num):
    # Read preprocessed trajectory data
    df = getDF(folder_path, video_name, filter_name, class_num)

    # Split data into trainging data and testing data
    X = getFeaturesData(feature, df)
    y = df['BehaviorType']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=54, stratify=y)

    # Define value of parameters for Grid Search
    if model_name == "SVM":
        params = [{'kernel': ['linear'],
                #    'C': [0.001,0.005,0.01,0.05,0.1,1]},  # non-normalized data
                   'C': [1,5,10,20,30,40,50]},  # nomalized data
                   {'kernel':['rbf'],'gamma':['scale', 'auto']}
                   ]
        model = SVC(random_state=1)
    elif model_name == "RandomForest":
        params = {'n_estimators': [700,750,800,900,1000], 
                  'max_features': ['sqrt', 'log2', None],
                  'max_depth' : [6,9,12],
                  'criterion' :['gini']}
        model = RandomForestClassifier(random_state=1)
    elif model_name == "XGBoost":
        params = {'max_depth': [8,9],
                  'learning_rate': [0.1, 0.15, 0.2, 0.25, 0.3],
                  'n_estimators': [150,200,250,300],
                  'colsample_bytree': [0.3,0.5,0.7],
                  'gamma': [0,1,2,3,4,5],
                  'n_jobs': [-1]}
        model = xgb.sklearn.XGBClassifier(random_state=1)
    else:
        print("Invalid model name.")

    # Do Grid Search, seeking the best combination of parameters
    skfold = StratifiedKFold(n_splits=5, random_state=99, shuffle=True)  # For 5-fold cross-validation
    clf = GridSearchCV(estimator=model, 
                       param_grid=params,
                       scoring='accuracy',
                       cv=skfold.split(X_train, y_train), 
                       verbose=1,
                       n_jobs=3)
    grid_result = clf.fit(X_train, y_train)

    # Show the result of all hyper-parameter combination
    print(f"Best accuracy: {grid_result.best_score_}, best parameter combination: {grid_result.best_params_}")
    # 取得 cross validation 的平均準確率及標準差
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f"Mean accuracy: {mean}, standard deviation: {stdev}, parameter combination: {param}")
    print("Best parameters:", clf.best_params_)


def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.xlabel("Weight")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model_, chosen_model, X):
    # Plot feature importances graph
    feature_names = ['AVG1', 'AVG2, DTW',
                    'Dist1', 'Dist2',
                    'max(D1)', 'max(D2)', 'min(D1)', 'min(D2)',
                    'V1_x', 'V1_y', 'V2_x', 'V2_y',
                    'SDR', 'avg(A)', 'min(A)', 'max(A)']

    if chosen_model == "SVM":
        for i in range(len(model_.coef_)):  # The weight of each features
            f_importances(model_.coef_[i], feature_names)
    elif chosen_model == "XGBoost":
        importances = model_.feature_importances_
        f_importances(importances, feature_names)
    elif chosen_model == "RandomForest":
        importances = model_.feature_importances_
        # f_importances(importances, feature_names)
        # print('Importance of each feature: ', importances)  # Show actual weight of features in command line
        std = np.std([tree.feature_importances_ for tree in model_.estimators_], axis=0)
        forest_importances = pd.Series(importances, index=feature_names)

        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show()
    else:
        print("This model do not exist, please check variable chosen_model: ", chosen_model)


def cross_val_predict(model, chosen_model, skfold: StratifiedKFold, X: np.array, y: np.array) -> Tuple[np.array, np.array, np.array]:
    # Reference: https://towardsdatascience.com/how-to-plot-a-confusion-matrix-from-a-k-fold-cross-validation-b607317e9874
    model_ = cp.deepcopy(model)
    no_classes = len(np.unique(y))

    pre = model_
    
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes]) 

    for i in range(0, 10):
        for train_ndx, test_ndx in skfold.split(X, y):

            train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

            actual_classes = np.append(actual_classes, test_y)
            model_.fit(train_X, train_y)
            predicted_classes = np.append(predicted_classes, model_.predict(test_X))

            try:
                predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
            except:
                predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    plot_feature_importance(model_, chosen_model, X)  # plot feature importance of the model

    return actual_classes, predicted_classes, predicted_proba


def create_saving_folder(folder_path):
    # Setting the path for saving confusion matrix pictures
    save_folder = folder_path + "confusion_matrix_picture/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    return save_folder


def plot_confusion_matrix(actual_classes, predicted_classes, sorted_labels, model_name, filter_name, feature, save_folder):
    cm = confusion_matrix(actual_classes, predicted_classes)
    sns.heatmap(cm, square=True, annot=True, fmt="d", cmap='Blues', xticklabels=sorted_labels, yticklabels=sorted_labels)
    # sns.despine(left=False, right=False, top=False, bottom=False)

    # Text part
    plt.title('Confusion Matrix of ' + model_name + ' Classifier' + "\n" + 
              'Filter Name: ' + filter_name + "\n" +
              'Used Feature: ' + feature)
    plt.xlabel("Predicted Value")
    plt.ylabel("True Value")
    plt.tight_layout()

    plt.savefig(save_folder + model_name + "_" + filter_name + "_confusion_matrix.png")
    plt.show()


def getDF(folder_path, video_name, filter_name, class_num):
    if class_num == 4:
        df = pd.read_csv(folder_path + "combined_preprocessed_data/" + video_name + '_' + filter_name + '_preprocessed_result.csv')
    elif class_num == 3:
        # Read preprocessed trajectory data
        df = pd.read_csv(folder_path + "combined_preprocessed_data/" + video_name + '_' + filter_name + '_preprocessed_result_half_bite_chase.csv')

        # Combine bite and chase, renumber all number of class type
        temp_df_type = df['BehaviorType'].copy()
        for index in range(0, len(df['BehaviorType'].index)):
            if df['BehaviorType'].iloc[index] == 0 or df['BehaviorType'].iloc[index] == 1:  # bite and chase
                temp_df_type.iloc[index] = 0
            elif df['BehaviorType'].iloc[index] == 2:  # display
                temp_df_type.iloc[index] = 1
            elif df['BehaviorType'].iloc[index] == 3:  # normal
                temp_df_type.iloc[index] = 2
            else:
                print("Index of BehaviorType is not 0, 1, 2 or 3.")
        df['BehaviorType'] = temp_df_type.copy()
    else:
        print("Wrong class_num setting.")
    return df


def machine_learning_main_cv_ver(folder_path, video_name, filter_name, chosen_model, feature, class_num):
    # Read preprocessed trajectory data
    df = getDF(folder_path, video_name, filter_name, class_num)

    # Split data into trainging data and testing data
    X = getFeaturesData(feature, df)
    y = df['BehaviorType']

    # Select model
    model, model_name = getModel(chosen_model)

    # k-fold  cross validation
    k_num = 5
    skfold = StratifiedKFold(n_splits=k_num, shuffle=True)  # Split data evenly among different behavior data type
    start_time = process_time()
    actual_classes, predicted_classes, _ = cross_val_predict(model, chosen_model, skfold, X, y)
    end_time = process_time()

    # If it is 3 categories, rename model name
    if class_num == 3:
        model_name = model_name + "_3categories"

    # Show the testing result with confusion matrix
    print(model_name, feature)
    print(classification_report(actual_classes, predicted_classes))

    # Setting sorted label and print the labels on screen
    if class_num == 3:
        sorted_labels = ['bite & chase', 'display', 'normal']
    else:
        sorted_labels = ['bite', 'chase', 'display', 'normal']
    print("Class label 0 ~", class_num," means: ", sorted_labels)

    # Setting the path and create a folder to save confusion matrix pictures
    save_folder = create_saving_folder(folder_path)

    # Plot the confusion matrix graph on screen, and save it in png format
    plot_confusion_matrix(actual_classes, predicted_classes, sorted_labels, model_name+"_"+str(k_num)+"cv", filter_name, feature, save_folder)
    print("Execution time: ", end_time - start_time)


def machine_learning_main(folder_path, video_name, filter_name, chosen_model, feature):
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
    print(classification_report(y_test, predictions))

    # Setting the path for saving confusion matrix pictures
    save_folder = folder_path + "confusion_matrix_picture/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Plot the confusion matrix graph on screen, and save it in png format
    sorted_labels = ['bite', 'chase', 'display', 'normal']
    # plot_confusion_matrix(y_test, predictions, sorted_labels, model_name, filter_name, feature, save_folder)

    features_names = ['AVG_1', 'AVG_2, DTW',
                    'Dist_1', 'Dist_2',
                    'max(D_1)', 'max(D_2)', 'min(D_1)', 'min(D_2)',
                    'V_1_x', 'V_1_y', 'V_2_x', 'V_2_y',
                    'SDR', 'avg(A)', 'min(A)', 'max(A)']

    if chosen_model == "SVM":
        f_importances(model.coef_[0], features_names)
