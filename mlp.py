import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from main import *

from keras.models import Sequential  
from keras.layers import Flatten, Dense, Dropout, Input 
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

model = Sequential([
    Input(shape=(28, 28)),  # 784, achatado
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 um pra cada dígito
])
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(x_train, y_train_cat, validation_split=0.1,
                    epochs=30, batch_size=128, callbacks=[es])


# Histórico de Métricas
print("\n--- Análise Detalhada do Modelo MLP no Conjunto de Teste ---")

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

print("Média treino accuracy:", np.mean(train_acc))
print("Média validação accuracy:", np.mean(val_acc))
print("Média treino loss:", np.mean(train_loss))
print("Média validação loss:", np.mean(val_loss))

test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0) #type: ignore
print(f"Acurácia no teste: {test_acc:.4f}")
print(f"Perda no teste: {test_loss:.4f}")

# Relatório de Classificação

y_pred_proba = model.predict(x_test)
y_pred = np.argmax(y_pred_proba, axis=1)

y_test_labels = np.argmax(y_test_cat, axis=1)

print("\n--- Relatório de Classificação ---")
print(classification_report(y_test_labels, y_pred))

# Matriz de Confusão

# Gerar a Matriz de Confusão para calcular a Especificidade
print("\n--- Matriz de Confusão ---")
cm = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True'); plt.show()

# Especificade para cada classe
print("\n--- Cálculo da Especificidade por Classe ---")
total_samples = np.sum(cm)
num_classes = len(cm)

for i in range(num_classes):
    tp = cm[i, i]
    fp = np.sum(cm[:, i]) - tp
    fn = np.sum(cm[i, :]) - tp
    tn = total_samples - (tp + fp + fn)

    # Fórmula
    specificity = tn / (tn + fp)

    print(f"Especificidade para a Classe {i}: {specificity:.4f}")
    print("--"*15)

# Gráfico de Loss e Accuracy

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

# gráfico de loss
epochs = range(len(history.history['loss']))
ax1.plot(epochs, train_loss, label='Train')
ax1.plot(epochs, val_loss, label='Validation')
ax1.set_title('Loss')
ax1.set_xlabel('epoch\n\n(a) Loss')
ax1.legend()

# gráfico de accuracy
ax2.plot(epochs, train_acc, label='Train')
ax2.plot(epochs, val_acc, label='Validation')
ax2.set_title('Accuracy')
ax2.set_xlabel('epoch\n\n(b) Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()
