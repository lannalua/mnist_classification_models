from main import *
import numpy as np
import time
from keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from mealpy import FloatVar, GJO

start_time = time.perf_counter()

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
    """
    solution = [dropout_rate, conv_idx, dense_idx]
    """
    dropout_rate = solution[0]
    conv_filters = CONV_FILTERS_SPACE[int(solution[1])]
    dense_units = DENSE_UNITS_SPACE[int(solution[2])]

    model = create_model_cnn(
        conv_filters=conv_filters,
        kernel_size=(3, 3),
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        optimizer="adam"
    )

    history = model.fit(
        x_train_cnn,
        y_train_cat,
        validation_split=0.1,
        epochs=3,                 # poucas épocas (otimização)
        batch_size=128,
        callbacks=[es],
        verbose=0
    )

    best_val_acc = max(history.history["val_accuracy"])
    return 1 - best_val_acc      # minimização


def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        dropout = np.random.uniform(DROPOUT_MIN, DROPOUT_MAX)
        conv_idx = np.random.randint(0, len(CONV_FILTERS_SPACE))
        dense_idx = np.random.randint(0, len(DENSE_UNITS_SPACE))
        population.append([dropout, conv_idx, dense_idx])
    return np.array(population, dtype=float)



def golden_jackal_optimization(
    pop_size=5,
    iterations=7
):
    population = initialize_population(pop_size)
    fitness = np.array([objective_function(p) for p in population])

    # líderes
    sorted_idx = np.argsort(fitness)
    alpha_male = population[sorted_idx[0]]
    alpha_female = population[sorted_idx[1]]

    for t in range(iterations):
        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()

            population[i] = (
                population[i]
                + r1 * (alpha_male - population[i])
                + r2 * (alpha_female - population[i])
            )

            # Limites
            population[i][0] = np.clip(
                population[i][0], DROPOUT_MIN, DROPOUT_MAX
            )
            population[i][1] = np.clip(
                population[i][1], 0, len(CONV_FILTERS_SPACE) - 1
            )
            population[i][2] = np.clip(
                population[i][2], 0, len(DENSE_UNITS_SPACE) - 1
            )

        fitness = np.array([objective_function(p) for p in population])
        sorted_idx = np.argsort(fitness)

        alpha_male = population[sorted_idx[0]]
        alpha_female = population[sorted_idx[1]]

        print(
            f"Iteração {t+1}/{iterations} "
            f"| Melhor val_acc = {1 - fitness[sorted_idx[0]]:.4f}"
        )

    return alpha_male

# =========================================================
# Execução do GJO
# =========================================================
best_solution = golden_jackal_optimization(
    pop_size=5,
    iterations=7
)

end_time = time.perf_counter()

best_dropout = best_solution[0]
best_conv_filters = CONV_FILTERS_SPACE[int(best_solution[1])]
best_dense_units = DENSE_UNITS_SPACE[int(best_solution[2])]

print("\nMelhores hiperparâmetros encontrados:")
print("Dropout rate:", best_dropout)
print("Conv filters:", best_conv_filters)
print("Dense units:", best_dense_units)

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