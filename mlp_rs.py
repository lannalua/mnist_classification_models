import seaborn as sns
from main import *
import numpy as np
from keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import Flatten, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

def create_model(neurons_per_layer=(256, 128), dropout_rate=0.3, optimizer='adam', activation='relu'):
  model = Sequential()
  model.add(Flatten(input_shape=(28, 28)))

  for neurons in neurons_per_layer:
      model.add(Dense(neurons, activation=activation))
      model.add(Dropout(dropout_rate))

  model.add(Dense(10, activation='softmax'))

  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy', metrics=['accuracy'])

  return model


param_dist = {
    'model__neurons_per_layer': [(256,), (256, 128)],
    'model__dropout_rate': uniform(0.2, 0.4),
    'batch_size': [32, 64, 128],
    'model__optimizer': ['adam', 'rmsprop', 'sgd'],
    'model__activation': ['relu', 'tanh']
}

keras_clf = KerasClassifier(
    model=create_model,
    epochs=15,
    callbacks=[es],
    verbose=0,
    validation_split=0.1
)

# cv=3 significa 3-fold cross-validation (treina 3 modelos para cada combinação)
random_search = RandomizedSearchCV(
    estimator=keras_clf,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search_result = random_search.fit(x_train, y_train_cat)

print(
    f"Melhor score (acurácia média na validação cruzada): {random_search_result.best_score_:.4f}")
print("Melhores parâmetros encontrados:")
print(random_search_result.best_params_)

best_keras_model = random_search_result.best_estimator_.model_ #type: ignore
test_loss, test_acc = best_keras_model.evaluate(x_test, y_test_cat, verbose=0)

print(f"\nAcurácia do melhor modelo no conjunto de teste: {test_acc:.4f}")
print(f"Perda do melhor modelo no conjunto de teste: {test_loss:.4f}")

# Refit the best model to get the history object
best_mlp_model = random_search_result.best_estimator_.model_  # type: ignore
history_best_mlp = best_mlp_model.fit(x_train, y_train_cat, validation_split=0.1, epochs=12,
                                      batch_size=random_search_result.best_params_[
                                          'batch_size'],
                                      callbacks=[es])

train_loss_best = history_best_mlp.history['loss']
val_loss_best = history_best_mlp.history['val_loss']
train_acc_best = history_best_mlp.history['accuracy']
val_acc_best = history_best_mlp.history['val_accuracy']

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

epochs_best = range(len(train_loss_best))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

# Gráfico (a) Loss
ax1.plot(epochs_best, train_loss_best, label='Train')
ax1.plot(epochs_best, val_loss_best, label='Validation')
ax1.set_title('Loss do Melhor Modelo MLP')
ax1.set_xlabel('epoch\n\n(a) Loss')
ax1.legend()

# Gráfico (b) Accuracy
ax2.plot(epochs_best, train_acc_best, label='Train')
ax2.plot(epochs_best, val_acc_best, label='Validation')
ax2.set_title('Accuracy do Melhor Modelo MLP')
ax2.set_xlabel('epoch\n\n(b) Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()
