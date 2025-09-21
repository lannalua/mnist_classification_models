from main import *

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

clf = LogisticRegression(max_iter=1000, n_jobs=-1)
clf.fit(x_train_flat, y_train)
y_pred = clf.predict(x_test_flat)
print("Acc:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
