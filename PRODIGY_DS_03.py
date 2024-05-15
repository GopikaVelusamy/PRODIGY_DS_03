import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score , precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

df=pd.read_csv("C:\\Users\\kavit\\OneDrive\\Desktop\\Prodigy Intern\\bank-additional.csv", sep=';', header=0)
print(df.head())

df = pd.get_dummies(df, drop_first=True)
X = df.drop('y_yes', axis=1)  # Assuming 'y_yes' is the column after dummy encoding for the target variable
y = df['y_yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("precision score:",precision_score(y_test,y_pred))
print("recall score:",recall_score(y_test,y_pred))
print("fi_score:",f1_score(y_test,y_pred))
plt.figure(figsize=(20,10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()

