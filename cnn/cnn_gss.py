import numpy as np
from main import *

import time
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping

def train_evaluate(dropout_rate, x_train_cnn, y_train_cat):
  es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
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
                        epochs=12, batch_size=128, callbacks=[es], verbose=0) #type: ignore
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

