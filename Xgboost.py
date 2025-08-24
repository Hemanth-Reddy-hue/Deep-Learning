from sklearn.datasets import load_iris
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

iris_data=load_iris()

x,y=iris_data.data,iris_data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
clf=XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,random_state=42,use_lable_encoder=False,eval_metrics="mlogloss")
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(f"accuracy_score : {accuracy_score(y_test,y_pred)}")
print(f"classification report")
print(classification_report(y_test,y_pred,target_names=iris_data.target_names))