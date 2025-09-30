from main import *
from mlp import es
from keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV


def create_model_cnn(conv_filters=(32, 64),
                     # kernel_size should be a tuple for Conv2D
                     kernel_size=(3, 3),
                     dense_units=(128,),
                     dropout_rate=0.3,
                     optimizer="adam",
                     activation="relu"
                     ):
  model = Sequential()
  model.add(Input(shape=(28, 28, 1)))  # Add Input layer correctly

  # Add Convolutional and Pooling layers
  for filters in conv_filters:
      model.add(Conv2D(filters, kernel_size, activation=activation))
      model.add(BatchNormalization())
      model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dropout(dropout_rate))

  # Add Dense layers
  for units in dense_units:
    model.add(Dense(units, activation=activation))

  model.add(Dense(10, activation="softmax"))

  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model


cnn_model = KerasClassifier(
    model=create_model_cnn,
    verbose=0,
    model__conv_filters=(32, 64),
    model__kernel_size=(3, 3),
    model__dense_units=(128,),
    model__dropout_rate=0.3,
    model__optimizer='adam'
)

param_grid = {

    "model__conv_filters": [(32,), (32, 64)],   
    "model__dense_units": [(128,), (256,)],

    "model__kernel_size": [(3, 3)],                 # 1 opção (fixo)
    "model__dropout_rate": [0.3],                   # 1 opção (fixo)
    "model__optimizer": ["adam"],                   # 1 opção (fixo)
    "batch_size": [128],                             # 1 opção (fixo)
    "epochs": [12],
    "callbacks": [[es]]
}


grid = GridSearchCV(estimator=cnn_model, param_grid=param_grid, cv=3)
# Use x_train_cnn and y_train_cat for CNN
grid_result = grid.fit(x_train_cnn, y_train_cat)

print("Melhor score:", grid_result.best_score_)
print("Melhores parâmetros:", grid_result.best_params_)

# Evaluate the best model on the test set
best_cnn_model = grid_result.best_estimator_.model_
test_loss, test_acc = best_cnn_model.evaluate(
    x_test_cnn, y_test_cat, verbose=0) #type: ignore

print(f"\nAcurácia do melhor modelo no conjunto de teste: {test_acc:.4f}")
print(f"Perda do melhor modelo no conjunto de teste: {test_loss:.4f}")

"""
with epochs 12
Melhor score: 0.9875333333333334
Melhores parâmetros: {'batch_size': 128, 'callbacks': [<keras.src.callbacks.early_stopping.EarlyStopping object at 0x7c86582a37d0>], 'epochs': 12, 'model__conv_filters': (32, 64), 'model__dense_units': (256,), 'model__dropout_rate': 0.3, 'model__kernel_size': (3, 3), 'model__optimizer': 'adam'}

Acurácia do melhor modelo no conjunto de teste: 0.9898
Perda do melhor modelo no conjunto de teste: 0.0418

with epochs 5
Melhor score: 0.9869333333333333
Melhores parâmetros: {'batch_size': 128, 'callbacks': [<keras.src.callbacks.early_stopping.EarlyStopping object at 0x7c85b25ccbf0>], 'epochs': 5, 'model__conv_filters': (32, 64), 'model__dense_units': (256,), 'model__dropout_rate': 0.3, 'model__kernel_size': (3, 3), 'model__optimizer': 'adam'}

Acurácia do melhor modelo no conjunto de teste: 0.9917
Perda do melhor modelo no conjunto de teste: 0.0268
"""