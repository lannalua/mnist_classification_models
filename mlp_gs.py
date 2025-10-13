from main import *
from mlp import es

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Encapsular a criação do modelo em uma função
def create_model(neurons_per_layer=(256, 128), dropout_rate=0.3, optimizer='adam', activation='relu'):
  model = Sequential()
  model.add(Flatten(input_shape=(28, 28)))

  for neurons in neurons_per_layer:
      model.add(Dense(neurons, activation=activation))
      model.add(Dropout(dropout_rate))

  model.add(Dense(10, activation='softmax'))
  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy', metrics=['accuracy'])

  return model


param_grid = {
    'model__neurons_per_layer': [(256,), (256, 128)],
    'model__dropout_rate': [0.3, 0.4],
    'batch_size': [64, 128],
    'model__optimizer': ['adam', 'rmsprop'],
    'model__activation': ['relu']
}


keras_clf = KerasClassifier(
    model=create_model,
    epochs=30,
    callbacks=[es],
    verbose=0,
    validation_split=0.1
)

# cv=3 significa 3-fold cross-validation (treina 3 modelos para cada combinação)
grid = GridSearchCV(
    estimator=keras_clf,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2)

# O .fit() agora executa a busca completa, o que pode ser DEMORADO!
print("Iniciando o Grid Search...")
grid_result = grid.fit(x_train, y_train_cat)

print("Grid Search concluído!")

# Exibir os melhores resultados
print(
    f"Melhor score (acurácia média na validação cruzada): {grid_result.best_score_:.4f}")
print("Melhores parâmetros encontrados:")
print(grid_result.best_params_)

best_keras_model = grid_result.best_estimator_.model_
test_loss, test_acc = best_keras_model.evaluate(x_test, y_test_cat, verbose=0) #type: ignore
print(f"Perda do melhor modelo no conjunto de teste: {test_loss:.4f}")

"""
Grid Search concluído!
Melhor score (acurácia média na validação cruzada): 0.9764
Melhores parâmetros encontrados:
{'batch_size': 64, 'model__activation': 'relu', 'model__dropout_rate': 0.3, 'model__neurons_per_layer': (256,), 'model__optimizer': 'adam'}

Acurácia do melhor modelo no conjunto de teste: 0.9818
Perda do melhor modelo no conjunto de teste: 0.0630
"""
