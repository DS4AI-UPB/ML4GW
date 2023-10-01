from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score 
import numpy as np
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

seed = 7
np.random.seed(seed)
n_estimators = 1000

accuracy = []
precision = []
precision_micro = []
precision_macro = []
recall = []
recall_micro = []
recall_macro = []

def getMeasures(y_test, y_pred, binary=False):
    try:
        if binary:            
            accuracy.append(accuracy_score(y_test, y_pred))
            precision.append(precision_score(y_test, y_pred, average='binary'))
            recall.append(recall_score(y_test, y_pred, average='binary'))
        else:
            accuracy.append(accuracy_score(y_test, y_pred))
            precision.append(precision_score(y_test, y_pred, average='weighted'))
            precision_micro.append(precision_score(y_test, y_pred, average='micro'))
            precision_macro.append(precision_score(y_test, y_pred, average='macro'))
            recall.append(recall_score(y_test, y_pred, average='weighted'))
            recall_micro.append(recall_score(y_test, y_pred, average='micro'))
            recall_macro.append(recall_score(y_test, y_pred, average='macro'))
    except:
        print("Din't work")

if __name__ == "__main__":
    fn = sys.argv[1]
    
    df = pd.read_csv(fn, sep=',')
    df = df.set_index(df.columns[1])
    labels = df[df.columns[-1]].unique()

    class2label = {}
    idx = 0
    for label in labels:
        class2label[label] = idx
        idx += 1

    no_classes = idx

    data = pd.DataFrame(df, columns = df.columns[1:])
    data = data.replace({df.columns[-1]: class2label})
    weights = data.groupby('label').count()[data.columns[1]].to_dict()
    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    print(X)

    scaler_std = StandardScaler()
    X = scaler_std.fit_transform(X)

    print(X)

    # print(X)
    # print(y)

    xgb = XGBClassifier(booster='gbtree', tree_method='exact', objective='multi:softmax', max_depth=6, n_estimators=100)

    etc = ExtraTreesClassifier(n_estimators=100, criterion='gini', max_features=None, class_weight=weights, n_jobs=-1)

    rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_features=None, class_weight=weights, n_jobs=-1)

    model = VotingClassifier(estimators=[('xgb', xgb), ('rfc', rfc), ('etc', etc)], voting='hard')
    

    # print(cross_val_score(model, X, y, cv=10))

    for i in range(0, 2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        getMeasures(y_test, y_pred)

    print("Accuracy: ", round(np.array(accuracy).mean(), 2), round(np.array(accuracy).std(), 2))
    print("Precision: ", round(np.array(precision).mean(), 2), round(np.array(precision).std(), 2))
    print("Precision_micro: ", round(np.array(precision_micro).mean(), 2), round(np.array(precision_micro).std(), 2))
    print("Precision_macro: ", round(np.array(precision_macro).mean(), 2), round(np.array(precision_macro).std(), 2))
    print("Recall: ", round(np.array(recall).mean(), 2), round(np.array(recall).std(), 2))
    print("Recall_micro: ", round(np.array(recall_micro).mean(), 2), round(np.array(recall_micro).std(), 2))
    print("Recall_macro: ", round(np.array(recall_macro).mean(), 2), round(np.array(recall_macro).std(), 2))
