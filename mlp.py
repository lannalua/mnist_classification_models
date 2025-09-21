import numpy as np
from main import *

from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Flatten, Dense, Dropout  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

model = Sequential([
    Flatten(input_shape=(28, 28)), #784, achatado
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax') # 10 um pra cada dígito
])
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(x_train, y_train_cat, validation_split=0.1,
                    epochs=30, batch_size=128, callbacks=[es])

# histórico de métricas
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

print("Média treino accuracy:", np.mean(acc))
print("Média validação accuracy:", np.mean(val_acc))
print("Média treino loss:", np.mean(loss))
print("Média validação loss:", np.mean(val_loss))

test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Acurácia no teste: {test_acc:.4f}")
print(f"Perda no teste: {test_loss:.4f}")
