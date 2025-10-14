import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from main import *
import numpy as np
from keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping  # Import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


def create_model_cnn(conv_filters=(32, 64),
                     kernel_size=(3, 3),
                     dense_units=(128,),
                     dropout_rate=0.3,
                     optimizer="adam",
                     activation="relu"
                     ):
  model = Sequential()
  model.add(Input(shape=(28, 28, 1)))

  for filters in conv_filters:
      model.add(Conv2D(filters, kernel_size, activation=activation))
      model.add(BatchNormalization())
      model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dropout(dropout_rate))

  for units in dense_units:
    model.add(Dense(units, activation=activation))

  model.add(Dense(10, activation="softmax"))

  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model


cnn_model = KerasClassifier(
    model=create_model_cnn,
    verbose=0,
    model__conv_filters=(32, 64),
    model__kernel_size=(3, 3),
    model__dense_units=(128,),
    model__dropout_rate=0.3,
    model__optimizer='adam',
    validation_split=0.1
)

param_grid = {
    "model__conv_filters": [(32,), (32, 64)],
    "model__dense_units": [(128,), (256,)],

    "model__kernel_size": [(3, 3)],
    "model__dropout_rate": [0.3],
    "model__optimizer": ["adam"],
    "batch_size": [128],
    "epochs": [5],
    "callbacks": [[es]]
}

grid = GridSearchCV(estimator=cnn_model, param_grid=param_grid, cv=3)
grid_result = grid.fit(x_train_cnn, y_train_cat)

## Histórico de Métricas

print("Melhor score:", grid_result.best_score_)
print("Melhores parâmetros:", grid_result.best_params_)

# Evaluate the best model on the test set
best_cnn_model = grid_result.best_estimator_.model_
test_loss, test_acc = best_cnn_model.evaluate(
    x_test_cnn, y_test_cat, verbose=0) # type:ignore

print(f"\nAcurácia do melhor modelo no conjunto de teste: {test_acc:.4f}")
print(f"Perda do melhor modelo do conjunto de teste: {test_loss:.4f}")

# Refit the best model to get the history object
history_best_cnn = best_cnn_model.fit(x_train_cnn, y_train_cat, validation_split=0.1, epochs=5,
                                      batch_size=grid_result.best_params_[
                                          'batch_size'],
                                      callbacks=[es])

train_loss_best = history_best_cnn.history['loss']
val_loss_best = history_best_cnn.history['val_loss']
train_acc_best = history_best_cnn.history['accuracy']
val_acc_best = history_best_cnn.history['val_accuracy']


# 1. Fazer previsões para obter as classes
y_pred_proba = best_cnn_model.predict(x_test_cnn)
y_pred = np.argmax(y_pred_proba, axis=1)

# 2. Converter os rótulos de teste (que estão em one-hot encoding) de volta para 1D
y_test_labels = np.argmax(y_test_cat, axis=1)

# Acurácia, Precisão, Sensibilidade (Recall) e F1-Score
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
ax2.set_title('Accuracy do Melhor Modelo CNN')
ax2.set_xlabel('epoch\n\n(b) Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()
