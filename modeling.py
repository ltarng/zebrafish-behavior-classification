import xgboost as xgb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


MODEL_MAP = {
    "SVM": lambda: SVC(C=0.01, kernel="linear"),
    "DecisionTree": lambda: DecisionTreeClassifier(),
    "RandomForest": lambda: RandomForestClassifier(max_depth=13, 
                                                   n_estimators=900, 
                                                   criterion="gini", 
                                                   max_features='log2'),
    "XGBoost": lambda: xgb.sklearn.XGBClassifier(colsample_bytree=0.5, 
                                                 learning_rate=0.3, 
                                                 max_depth=9, 
                                                 n_estimators=200)
}


def getModel(chosen_model: str):
    if chosen_model in MODEL_MAP:
        return MODEL_MAP[chosen_model](), chosen_model
    else:
        raise ValueError(f"Error: model name {chosen_model} does not exist.")
