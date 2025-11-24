from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from main import *
import tensorflow as tf
import keras_tuner as kt
from keras.callbacks import EarlyStopping
import time
from keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, SGD

# Cronômetro
start_time = time.perf_counter()

es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

def build_model(hp):

    model = Sequential()
    # Corrected: Specify input_shape directly in the first layer (Flatten)
    model.add(Flatten(input_shape=(28, 28, 1)))

    for i in range(hp.Int('num-layres', min_value=1, max_value=3)):
      model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                      activation=hp.Choice('activation', values=['relu', 'tanh'])))
      model.add(Dropout(rate=hp.Float(
          'dropout', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Dense(10, activation="softmax"))

    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    opt_choice = hp.Choice('optimizer', values=['adam', 'sgd'])

    if opt_choice == 'adam':
        optimizer = Adam(learning_rate=lr)
    else:
        optimizer = SGD(learning_rate=lr)

    # Mantendo sua loss function original
    model.compile(optimizer=optimizer,  # type: ignore
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=6,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

tuner.search(x_train_cnn, y_train_cat, epochs=6,
             validation_split=0.2, callbacks=[es], verbose=1)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Melhores hiperparâmetros encontrados:")
for k, v in best_hps.values.items():
    print(f"{k}: {v}")

model = tuner.hypermodel.build(best_hps) #type: ignore 
history = model.fit(
    x_train_cnn, y_train_cat,
    epochs=15,
    validation_split=0.2,
    callbacks=[es]
)

end_time = time.perf_counter()
elapsed_time = (end_time - start_time)/60

test_loss, test_acc = model.evaluate(x_test_cnn, y_test_cat, verbose=0)
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
