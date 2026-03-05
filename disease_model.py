import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score,classification_report

#load dataset.
data =load_breast_cancer()
x= pd.DataFrame(data.data,columns=data.feature_names)
y = data.target

print("Dataset shape:",x.shape)
print("Target Values:",np.unique(y))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

lr = LogisticRegression(max_iter=5000)
lr.fit(x_train_scaled,y_train)
y_pred_lr = lr.predict(x_test_scaled)

print("Logic Regession Accuracy : ",accuracy_score(y_test,y_pred_lr))
print(classification_report(y_test,y_pred_lr))

svm = SVC()
svm.fit(x_train_scaled,y_train)
y_pred_svm = svm.predict(x_test_scaled)

print("SVM Accuracy : ",accuracy_score(y_test,y_pred_svm))
print(classification_report(y_test,y_pred_svm))

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)
y_pred_rf = rf.predict(x_test)

print("Random Forest Accuracy : ",accuracy_score(y_test,y_pred_rf))
print(classification_report(y_test,y_pred_rf))

xgb = XGBClassifier(use_label_encoder=False,eval_matric='logloss')
xgb.fit(x_train,y_train)

y_pred_xgb = xgb.predict(x_test)
print("XGBoost Accuracy : ",accuracy_score(y_test,y_pred_xgb))
print(classification_report(y_test,y_pred_xgb))


