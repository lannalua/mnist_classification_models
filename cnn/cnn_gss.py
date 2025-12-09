import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import numpy as np
from main import *
import time
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


def train_evaluate(lr, batch_size, dropout_rate):

  batch_size = int(batch_size)

  tf.keras.backend.clear_session() #type: ignore

  es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

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

  opt = Adam(learning_rate=lr)

  cnn.compile(optimizer=opt, loss='categorical_crossentropy',  # type: ignore
              metrics=['accuracy'])
  
  history_cnn = cnn.fit(x_train_cnn, y_train_cat,
                        validation_split=0.1,
                        epochs=6,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=0)  # type: ignore

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


def otimizar_lr(batch_fixo, dropout_fixo):
  print(
      f"\n--- [1/3] Buscando LR (Fixos: BS={batch_fixo}, Drop={dropout_fixo}) ---")

  def wrapper(val_log):
    real_lr = 10 ** val_log
    best_accuracy = train_evaluate(lr=real_lr,
                                   batch_size=batch_fixo,
                                   dropout_rate=0.0)
    return best_accuracy
  
  melhor_log = golden_section_search(wrapper, a=-4, b=-1, tol=0.2)
  return 10 ** melhor_log


def otimizar_dropout(lr_fixo, batch_fixo):
  print(
      f"\n--- [2/3] Buscando Dropout (Fixos: LR={lr_fixo:.5f}, BS={batch_fixo}) ---")

  def wrapper(val_drop):
    if val_drop < 0:
      val_drop = 0
    if val_drop >= 0.9:
      val_drop = 0.9

    best_accuracy = train_evaluate(
        lr=lr_fixo, batch_size=batch_fixo, dropout_rate=val_drop)
    print(f" Dropout Testado: {val_drop:.4f} -> Accuracy: {best_accuracy:.4f}")

    return best_accuracy

  return golden_section_search(wrapper, a=0.1, b=0.6, tol=0.01)


def otimizar_batch(lr_fixo, dropout_fixo):
  print(
      f"\n--- [3/3] Buscando Batch Size (Fixos: LR={lr_fixo:.5f}, Drop={dropout_fixo:.4f}) ---")

  def wrapper(val_bs):
    real_bs = int(val_bs)
    best_accuracy = train_evaluate(
        lr=lr_fixo, batch_size=real_bs, dropout_rate=dropout_fixo)
    print(f"   Batch Testado: {real_bs} -> Accuracy: {best_accuracy:.4f}")

    return best_accuracy

    # Busca entre 16 e 128
  res = golden_section_search(wrapper, a=16, b=128, tol=10)
  return int(res)


start_global = time.time()

start_time_lr = time.perf_counter()

# 1. Acha LR (com valores padrão seguros para os outros)
best_lr = otimizar_lr(batch_fixo=64, dropout_fixo=0.0)
print(f">>> Vencedor LR: {best_lr:.5f}")
end_time_lr = time.perf_counter()

start_time_dropout = time.perf_counter()
# 2. Acha Dropout (usando o LR que acabamos de achar)
best_drop = otimizar_dropout(lr_fixo=best_lr, batch_fixo=64)
print(f">>> Vencedor Dropout: {best_drop:.4f}")
end_time_dropout = time.perf_counter()

start_time_bs = time.perf_counter()
# 3. Acha Batch Size (usando LR e Dropout vencedores)
best_bs = otimizar_batch(lr_fixo=best_lr, dropout_fixo=best_drop)
print(f">>> Vencedor Batch Size: {best_bs}")
end_time_bs = time.perf_counter()

print(
    f"\nTempo Total do Learning Rate: {(end_time_lr - start_time_lr)/60:.1f} minutos")
print(
    f"\nTempo Total do Dropout: {(end_time_dropout - start_time_dropout)/60:.1f} minutos")
print(
    f"\nTempo Total do Batch_size: {(end_time_bs - start_time_bs)/60:.1f} minutos")
print(f"\nTempo Total da Busca: {(time.time() - start_global)/60:.1f} minutos")

start_time_gss = time.perf_counter()

print(f"Config: LR={best_lr} | Drop={best_drop} | Batch={best_bs}")

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
    Dropout(best_drop),  # Vencedor
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

opt_final = Adam(learning_rate=best_lr)  # Vencedor

model_final.compile(optimizer=opt_final, #type: ignore
                    loss='categorical_crossentropy', metrics=['accuracy'])

history_final = model_final.fit(
    x_train_cnn, y_train_cat,
    validation_split=0.1,
    epochs=15,
    batch_size=best_bs,
    callbacks=[es],
    verbose=1 #type: ignore
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
plt.ylabel('True'); plt.show()

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
