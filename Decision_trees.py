from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt

iris=load_iris()
x,y=iris.data,iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
clf=DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=42)
clf.fit(x_train,y_train)
print("Accuracy :",clf.score(x_test,y_test))
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
for i in range(5):
    print(f"prediction : {clf.predict([x[i]])} Actual : {[[y[i]]]}")
