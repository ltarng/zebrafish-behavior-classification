import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(confusion_matrix):
    sns.heatmap(confusion_matrix, square= True, annot=True, cbar= False)
    plt.xlabel("predicted value")
    plt.ylabel("true value")
    plt.show()


def svm(folder_path, video_name, filter_name):
    # Read DTW result file
    df = pd.read_csv(folder_path + video_name + '_' + filter_name + '_filtered_DTW.csv')

    # SVM model
    # Divide data into trainging data and testing data
    X = df.iloc[:,3:5].values
    y = df['BehaviorType']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=99)

    # Use Support Vector Classifier to build model
    model = SVC()
    model.fit(X_train, y_train)

    # Use test data to test the result of model
    predictions = model.predict(X_test)

    # Load classification_report & confusion_matrix to estimate the performance of model
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    print('\n')
    print(classification_report(y_test, predictions))

    plot_confusion_matrix(cm)
