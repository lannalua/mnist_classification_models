from main import *
import numpy as np
import time
from keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from mealpy.swarm_based import GJO
from mealpy import FloatVar


CONV_FILTERS_SPACE = [(32,), (32, 64)]
DENSE_UNITS_SPACE = [(128,), (256,)]

DROPOUT_MIN = 0.1
DROPOUT_MAX = 0.5

es = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

def create_model_cnn(
    conv_filters=(32, 64),
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

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def objective_function(solution):
    dropout_rate = solution[0]

    conv_idx = int(np.round(solution[1]))
    dense_idx = int(np.round(solution[2]))

    conv_filters = CONV_FILTERS_SPACE[conv_idx]
    dense_units = DENSE_UNITS_SPACE[dense_idx]

    model = create_model_cnn(
        conv_filters=conv_filters,
        dense_units=dense_units,
        dropout_rate=dropout_rate
    )

    history = model.fit(
        x_train_cnn,
        y_train_cat,
        validation_split=0.1,
        epochs=3,          # poucas épocas para otimização
        batch_size=128,
        callbacks=[es],
        verbose=0
    )

    best_val_acc = max(history.history["val_accuracy"])
    return 1 - best_val_acc   # minimização


problem = {
    "obj_func": objective_function,
    "bounds": [
        FloatVar(lb=DROPOUT_MIN, ub=DROPOUT_MAX),           # dropout
        FloatVar(lb=0, ub=len(CONV_FILTERS_SPACE) - 1),     # conv idx
        FloatVar(lb=0, ub=len(DENSE_UNITS_SPACE) - 1)       # dense idx
    ],
    "minmax": "min"
}

start_time = time.perf_counter()

model = GJO.OriginalGJO(
    epoch=7,        # iterações
    pop_size=5      # população
)

best_agent = model.solve(problem)

end_time = time.perf_counter()

best_solution = best_agent.solution

best_dropout = best_solution[0]
best_conv_filters = CONV_FILTERS_SPACE[int(round(best_solution[1]))]
best_dense_units = DENSE_UNITS_SPACE[int(round(best_solution[2]))]

print("\nMelhores hiperparâmetros encontrados:")
print("Dropout rate:", best_dropout)
print("Conv filters:", best_conv_filters)
print("Dense units:", best_dense_units)
print(f"Tempo total: {end_time - start_time:.2f}s")

# =========================================================
# Treinamento FINAL do modelo
# =========================================================
final_model = create_model_cnn(
    conv_filters=best_conv_filters,
    kernel_size=(3, 3),
    dense_units=best_dense_units,
    dropout_rate=best_dropout,
    optimizer="adam"
)

final_model.fit(
    x_train_cnn,
    y_train_cat,
    validation_split=0.1,
    epochs=12,
    batch_size=128,
    callbacks=[es],
    verbose=1
)

# =========================================================
# Tempo total
# =========================================================

print(f"\nTempo total de execução: {end_time - start_time:.2f} segundos")
test_loss, test_acc = final_model.evaluate(x_test_cnn, y_test_cat, verbose=0)

print(f"Acurácia do modelo final no conjunto de teste: {test_acc:.4f}")
print(f"Perda do modelo final no conjunto de teste: {test_loss:.4f}")