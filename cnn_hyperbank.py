import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from main import *

import tensorflow as tf
import keras_tuner as kt
import time
from keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping


start_time = time.perf_counter()

es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)


def build_model(hp):

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))

    for i in range(hp.Int('conv_blocks', 1, 2, default=1)):
      model.add(Conv2D(filters=hp.Int(f'filters_{i}', 32, 128, step=32),
                       kernel_size=hp.Choice('kernel_size', [3]),
                       activation='relu',
                       padding='same'
                       ))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      if hp.Boolean(f'batch_norm_{i}', default=True):
        model.add(BatchNormalization())

    # Flatten + Dropout + Camadas densas
    model.add(Flatten())

    model.add(Dense(units=hp.Int('dense_units', 64, 128, step=64),
                    activation='relu'))

    model.add(Dropout(hp.Float('dropout', 0.2, 0.4, step=0.1)))

    model.add(Dense(10, activation="softmax"))

    hp_learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(
                      from_logits=False),
                  metrics=['accuracy'])
    return model


tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=6,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

tuner.search(x_train_cnn, y_train, epochs=6, validation_split=0.2, callbacks=[
             es], verbose=1)  # Corrected: use x_train_cnn and 'callbaks' to 'callbacks'

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Melhores hiperparâmetros encontrados:")
for k, v in best_hps.values.items():
    print(f"{k}: {v}")

model = tuner.hypermodel.build(best_hps)
history = model.fit(
    x_train_cnn, y_train,
    epochs=10,
    validation_split=0.2,
    callbacks=[es]
)

end_time = time.perf_counter()
elapsed_time = (end_time - start_time)/60

test_loss, test_acc = model.evaluate(x_test_cnn, y_test, verbose=0)
print(f"\nAcurácia no conjunto de teste: {test_acc:.4f}")

print("\n--- Análise Detalhada do Melhor Modelo CNN Hyperband no Conjunto de Teste ---")

y_pred_proba = model.predict(x_test_cnn)
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

train_loss_best = history.history['loss']
val_loss_best = history.history['val_loss']
train_acc_best = history.history['accuracy']
val_acc_best = history.history['val_accuracy']

epochs_best = range(len(train_loss_best))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

# Gráfico (a) Loss
ax1.plot(epochs_best, train_loss_best, label='Train')
ax1.plot(epochs_best, val_loss_best, label='Validation')
ax1.set_title('Loss do Melhor Modelo CNN - Bayesiana')
ax1.set_xlabel('epoch\n\n(a) Loss')
ax1.legend()

# Gráfico (b) Accuracy
ax2.plot(epochs_best, train_acc_best, label='Train')
ax2.plot(epochs_best, val_acc_best, label='Validation')
ax2.set_title('Accuracy do Melhor Modelo CNN - Bayesiana')
ax2.set_xlabel('epoch\n\n(b) Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()

with open("results_cnn.txt", "a") as f:
    f.write("CNN - Hyperband v1: \n")
    f.write(f"Time: {elapsed_time} min \n")

    f.write(
        "Trial 10 Complete [00h 11m 27s] \n val_accuracy: 0.9912499785423279 \n")
    f.write(
        "Best val_accuracy So Far: 0.991249978542327\n Total elapsed time: 01h 03m 50s \n")

    f.write("Melhores hiperparâmetros encontrados:\n")

    for k, v in best_hps.values.items():
        f.write(f"{k}: {v} \n")

    f.write("\n")

    f.write(f"Média treino accuracy: {np.mean(train_acc_best)}\n")
    f.write(f"Média validação accuracy: {np.mean(val_acc_best)}\n")
    f.write(f"Média treino loss: {np.mean(train_loss_best)}\n")
    f.write(f"Média validação loss: {np.mean(val_loss_best)}\n")

    f.write(f"Acurácia no teste: {test_acc:.4f}\n")
    f.write(f"Perda no teste: {test_loss:.4f}\n")
    f.close()

print("Resultados salvos em results_cnn.txt")
