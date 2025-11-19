import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from main import *

import numpy as np
import time
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

start_time = 0
start_time = time.perf_counter()
es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)


def create_model(num_hidden_layers=1, neurons_per_layer=256, dropout_rate=0.3, optimizer='adam', activation='relu'):
  model = Sequential()
  model.add(Flatten(input_shape=(28, 28)))

  for _ in range(num_hidden_layers):
      model.add(Dense(neurons_per_layer, activation=activation))
      model.add(Dropout(float(dropout_rate)))

  model.add(Dense(10, activation='softmax'))

  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy', metrics=['accuracy'])

  return model


keras_clf = KerasClassifier(
    model=create_model,
    epochs=15,
    callbacks=[es],
    verbose=0,
    validation_split=0.1
)


# Definição do Espaço de Busca
search_spaces = {
    'model__num_hidden_layers': Categorical([1, 2]),
    'model__neurons_per_layer': Categorical([128, 256]),
    'model__dropout_rate': Real(0.2, 0.5, prior='uniform'),
    'batch_size': Categorical([32, 64]),
    'model__optimizer': Categorical(['adam', 'rmsprop', 'sgd']),
    'model__activation': Categorical(['relu', 'tanh'])
}

bayes_search = BayesSearchCV(
    estimator=keras_clf,
    search_spaces=search_spaces,
    n_iter=10,
    cv=3,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

bayes_search_result = bayes_search.fit(x_train, y_train_cat)

print(
    f"Melhor score (acurácia média na validação cruzada): {bayes_search_result.best_score_:.4f}")  # type:ignore
print("Melhores parâmetros encontrados:")
print(bayes_search_result.best_params_)  # type:ignore

best_keras_model = bayes_search_result.best_estimator_.model_  # type:ignore
test_loss, test_acc = best_keras_model.evaluate(x_test, y_test_cat, verbose=0)

print(f"\nAcurácia do melhor modelo no conjunto de teste: {test_acc:.4f}")
print(f"Perda do melhor modelo no conjunto de teste: {test_loss:.4f}")

# Refit the best model to get the history object
best_bayes_search_model = bayes_search_result.best_estimator_.model_  # type:ignore
history_best_bayes_search = best_bayes_search_model.fit(x_train, y_train_cat, validation_split=0.1, epochs=12,
                                                        batch_size=bayes_search_result.best_params_[  # type:ignore
                                                            'batch_size'],
                                                        callbacks=[es])
end_time = time.perf_counter()
elapsed_time = (end_time - start_time)/60

train_loss_best = history_best_bayes_search.history['loss']
val_loss_best = history_best_bayes_search.history['val_loss']
train_acc_best = history_best_bayes_search.history['accuracy']
val_acc_best = history_best_bayes_search.history['val_accuracy']

print("\n--- Análise Detalhada do Melhor Modelo MLP no Conjunto de Teste ---")

y_pred_proba = best_keras_model.predict(x_test)
y_pred = np.argmax(y_pred_proba, axis=1)

y_test_labels = np.argmax(y_test_cat, axis=1)

print("\n--- Relatório de Classificação ---")
print(classification_report(y_test_labels, y_pred))

print("\n--- Matriz de Confusão ---")

cm = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True'); plt.show()

print("\n--- Cálculo da Especificidade por Classe ---")
total_samples = np.sum(cm)
num_classes = len(cm)

for i in range(num_classes):
    tp = cm[i, i]
    fp = np.sum(cm[:, i]) - tp
    fn = np.sum(cm[i, :]) - tp
    tn = total_samples - (tp + fp + fn)

    # Fórmula da Especificidade
    specificity = tn / (tn + fp)

    print(f"Especificidade para a Classe {i}: {specificity:.4f}")
    print("-"*15)

with open("results_mlp.txt", "a") as f:
    f.write("MLP - Bayesiana v5: \n")
    f.write("Add: epochs: 15. 'model__dropout_rate': Real(0.2, 0.5, prior='uniform') batch_size: tirei 128\n")
    f.write(f"Time: {elapsed_time} min \n")

    f.write(f"Melhor score: {bayes_search_result.best_score_:.4f}\n") #type:ignore
    f.write(str(bayes_search_result.best_params_))  # type:ignore
    f.write("\n")

    f.write(f"Média treino accuracy: {np.mean(train_acc_best)}\n")
    f.write(f"Média validação accuracy: {np.mean(val_acc_best)}\n")
    f.write(f"Média treino loss: {np.mean(train_loss_best)}\n")
    f.write(f"Média validação loss: {np.mean(val_loss_best)}\n")

    f.write(f"Acurácia no teste: {test_acc:.4f}\n")
    f.write(f"Perda no teste: {test_loss:.4f}\n")
    f.close()

print("Resultados salvos em results_mlp.txt")
