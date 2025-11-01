import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from main import *

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

clf = LogisticRegression(max_iter=1000, n_jobs=-1)
history = clf.fit(x_train_flat, y_train)
y_pred = clf.predict(x_test_flat)


print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\n--- Matriz de Confusão ---")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True'); plt.show()

#Grid Search para achar o melhor C

C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
scores = []

for choice in C:
    clf.set_params(C=choice)
    clf.fit(x_train_flat, y_train)
    scores.append(clf.score(x_test_flat, y_train))

print(scores)

with open("results.txt", "a") as f:
    f.write("Logistic regression \n")

    f.write(f'Accurary inicial: {accuracy_score(y_test, y_pred)}')
    f.write("--- Resultados para diferentes valores de C ---\n")
    for i, choice in enumerate(C):
        f.write(f"C = {choice}: Accuracy = {scores[i]:.4f}\n")
    f.close()


print("Resultados salvos em logistic_regression_results.txt")

