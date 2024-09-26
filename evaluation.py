import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def plot_confusion_matrix(actual_classes, predicted_classes, sorted_labels, model_name, filter_name, feature, save_folder):
    cm = confusion_matrix(actual_classes, predicted_classes)
    sns.heatmap(cm, square=True, annot=True, fmt="d", cmap='Blues', xticklabels=sorted_labels, yticklabels=sorted_labels)

    plt.title(f'Confusion Matrix of {model_name} Classifier\nFilter Name: {filter_name}\nUsed Feature: {feature}')
    plt.xlabel("Predicted Value")
    plt.ylabel("True Value")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"{model_name}_{filter_name}_confusion_matrix.png"))
    plt.show()

def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))
