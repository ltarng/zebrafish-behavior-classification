import numpy as np
import matplotlib.pyplot as plt


def f_importances(coef, names):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.xlabel("Weight")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model_, chosen_model, features):
    feature_names = {
        'dtw_velocities_direction_sdr_partangles_length': ['AVG1', 'AVG2', 'DTW', 'Dist1', 'Dist2',
                                                           'max(D1)', 'max(D2)', 'min(D1)', 'min(D2)',
                                                           'V1_x', 'V1_y', 'V2_x', 'V2_y',
                                                           'SDR', 'min(A)', 'max(A)'],
        'dtw_velocities_direction_sdr_angles_length': ['AVG1', 'AVG2', 'DTW', 'Dist1', 'Dist2',
                                                       'max(D1)', 'max(D2)', 'min(D1)', 'min(D2)',
                                                       'V1_x', 'V1_y', 'V2_x', 'V2_y',
                                                       'SDR', 'avg(A)', 'min(A)', 'max(A)']
    }.get(features)

    if feature_names is None:
        print("Invalid features name.")
        return

    if chosen_model == "SVM":
        for i in range(len(model_.coef_)):
            f_importances(model_.coef_[i], feature_names)
    elif chosen_model in ["XGBoost", "RandomForest"]:
        importances = model_.feature_importances_ if chosen_model == "XGBoost" else model_.feature_importances_
        if chosen_model == "RandomForest":
            std = np.std([tree.feature_importances_ for tree in model_.estimators_], axis=0)
            plt.barh(range(len(feature_names)), importances, yerr=std, align='center')
            plt.title("Feature importances using MDI")
            plt.ylabel("Mean decrease in impurity")
        else:
            f_importances(importances, feature_names)
    else:
        print("This model does not exist, please check variable chosen_model:", chosen_model)
