from main import *
from mlp import es
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
# import tensorflow as tf 

# tf.device("GPU:0")

cnn = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam', loss='categorical_crossentropy',
            metrics=['accuracy'])
history_cnn = cnn.fit(x_train_cnn, y_train_cat, validation_split=0.1,
                      epochs=12, batch_size=128, callbacks=[es])

#histórico de métricas

acc = history_cnn.history['accuracy']
val_acc = history_cnn.history['val_accuracy']
loss = history_cnn.history['loss']
val_loss = history_cnn.history['val_loss']

print("Média treino accuracy:", np.mean(acc))
print("Média validação accuracy:", np.mean(val_acc))
print("Média treino loss:", np.mean(loss))
print("Média validação loss:", np.mean(val_loss))

test_loss, test_acc = cnn.evaluate(x_test_cnn, y_test_cat, verbose=0)
print(f"Acurácia no teste: {test_acc:.4f}")
print(f"Perda no teste: {test_loss:.4f}")

#relatório de classificação

print("\n-- Relatório de Classificação---")
y_pred_proba = cnn.predict(x_test_cnn)
y_pred = y_pred_proba.argmax(axis=1)

print(classification_report(y_test, y_pred))

#Matriz de comfusão

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True'); plt.show()

# Especificidade por classe

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

#Gráficos de Loss e Accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

# Gráfico (a) Loss
# Adjust epochs to match the actual number of epochs for CNN
epochs = range(len(history_cnn.history['loss']))
ax1.plot(epochs, loss, label='Train')
ax1.plot(epochs, val_loss, label='Validation')
ax1.set_title('Loss')
ax1.set_xlabel('epoch\n\n(a) Loss')
ax1.legend()

# Gráfico (b) Accuracy
ax2.plot(epochs, acc, label='Train')
ax2.plot(epochs, val_acc, label='Validation')
ax2.set_title('Accuracy')
ax2.set_xlabel('epoch\n\n(b) Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()
