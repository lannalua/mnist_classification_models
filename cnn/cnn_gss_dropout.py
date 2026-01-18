import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import *
import time
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


def train_evaluate(dropout_rate, x_train_cnn, y_train_cat):
    es = EarlyStopping(monitor='val_loss', patience=2,
                       restore_best_weights=True)
    cnn = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(dropout_rate),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    cnn.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
    history_cnn = cnn.fit(x_train_cnn, y_train_cat, validation_split=0.1,
                          epochs=12, batch_size=128, callbacks=[es], verbose=0)  
    best_accuracy = max(history_cnn.history['val_accuracy'])

    return best_accuracy


def golden_section_search(f, a, b, tol, *args):
    """
    Aplica o Golden Section Search (GSS) para MAXIMIZAR a função f no intervalo [a, b].
    """
    # Razão Áurea invertida
    r = (3 - np.sqrt(5)) / 2  # r ≈ 0.381966

    x1 = a + r * (b - a)
    x2 = b - r * (b - a)

    f1 = f(x1, *args)  # Acurácia em x1
    f2 = f(x2, *args)  # Acurácia em x2

    iteration = 0
    print(f"--- Início GSS | Otimizando Taxa de Dropout ---")

    while abs(b - a) > tol:
        iteration += 1

        if f1 > f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + r * (b - a)
            f1 = f(x1, *args)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - r * (b - a)
            f2 = f(x2, *args)

        print(
            f"Iteração {iteration}: Intervalo [{a:.4f}, {b:.4f}] | Testando {x1:.4f} e {x2:.4f} | Melhor Acurácia: {max(f1, f2):.4f}")

    return (a + b) / 2


a_inicial = 0.1
b_inicial = 0.5
tolerancia = 0.01

start_time_gss = time.perf_counter()
dropout_otimo = golden_section_search(
    train_evaluate,
    a_inicial,
    b_inicial,
    tolerancia,
    x_train_cnn, y_train_cat
)
end_time_gss = time.perf_counter()

print(f"\n--- Resultado Final ---")
print(f"✅ Taxa de Dropout Ótima Encontrada: {dropout_otimo:.4f}")
print(
    f"Tempo total gasto no GSS: {(end_time_gss - start_time_gss)/60:.2f} minutos")

start_time_gss = time.perf_counter()

print(f"drop={dropout_otimo}")

es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Recriar modelo final
model_final = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(dropout_otimo),  # Vencedor
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])


model_final.compile(
    optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

history_final = model_final.fit(
    x_train_cnn, y_train_cat,
    validation_split=0.1,
    epochs=15,
    batch_size=128,
    callbacks=[es],
    verbose=1  # type: ignore
)
end_time_gss = time.perf_counter()
print(
    f"Tempo total gasto no modelo final: {(end_time_gss - start_time_gss)/60:.2f} minutos")
print("Modelo Final Treinado!")

acc = history_final.history['accuracy']
val_acc = history_final.history['val_accuracy']
loss = history_final.history['loss']
val_loss = history_final.history['val_loss']

print("Média treino accuracy:", np.mean(acc))
print("Média validação accuracy:", np.mean(val_acc))
print("Média treino loss:", np.mean(loss))
print("Média validação loss:", np.mean(val_loss))

test_loss, test_acc = model_final.evaluate(x_test_cnn, y_test_cat, verbose=0) #type: ignore
print(f"Acurácia no teste: {test_acc:.4f}")
print(f"Perda no teste: {test_loss:.4f}")

# predição e relatório (para CNN)
print("\n-- Relatório de Classificação---")
y_pred_proba = model_final.predict(x_test_cnn)
y_pred = y_pred_proba.argmax(axis=1)

print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

# Gráfico (a) Loss
# Adjust epochs to match the actual number of epochs for CNN
epochs = range(len(history_final.history['loss']))
ax1.plot(epochs, loss, label='Train')
ax1.plot(epochs, val_loss, label='Validation')
ax1.set_title('Loss - CNN')
ax1.set_xlabel('epoch\n\n(a) Loss')
ax1.legend()

# Gráfico (b) Accuracy
ax2.plot(epochs, acc, label='Train')
ax2.plot(epochs, val_acc, label='Validation')
ax2.set_title('Accuracy - CNN')
ax2.set_xlabel('epoch\n\n(b) Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()
