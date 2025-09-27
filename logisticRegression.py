from main import *

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

clf = LogisticRegression(max_iter=1000, n_jobs=-1)
# C - grid search
# clf.set_params(C=0.25) 
clf.fit(x_train_flat, y_train)
y_pred = clf.predict(x_test_flat)
print("Acc:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
scores = []

for choice in C:
    clf.set_params(C=choice)
    clf.fit(x_train_flat, y_train)
    scores.append(clf.score(x_test_flat, y_train))

print(scores)

