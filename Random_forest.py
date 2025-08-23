from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris_data = load_iris()
x, y = iris_data.data, iris_data.target

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Random Forest model
clf = RandomForestClassifier(n_estimators=3, max_depth=3, criterion="entropy", random_state=42)
clf.fit(x_train, y_train)

# Predictions
y_pred = clf.predict(x_test)

# Evaluation
print(f"Final Accuracy Score: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris_data.target_names))
