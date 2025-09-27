from main import *
from mlp import es

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
# Encapsular a criação do modelo em uma função

def create_model(hidden_layers=1, neurons=256, dropout_rate=0.3, optimizer='adam', activation='relu'):
  # model = Sequential([
  #   Flatten(input_shape=(28, 28)), #784, achatado
  #   Dense(neurons_l1, activation='relu'),
  #   Dropout(dropout_rate),
  #   Dense(neurons_l2, activation='relu'),
  #   Dense(10, activation='softmax') # 10 um pra cada dígito
  # ])
  model = Sequential()
  model.add(Flatten(input_shape=(28, 28)))
  for _ in range(hidden_layers):
      model.add(Dense(neurons, activation=activation))
      model.add(Dropout(dropout_rate))

    # Camada de saída
  model.add(Dense(10, activation='softmax'))
  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy', metrics=['accuracy'])

  return model


param_grid = {
    'model__hidden_layers': [1, 2],
    'model__neurons': [128, 256],
    'model__dropout_rate': [0.3, 0.4],
    'batch_size': [64, 128],
    'model__optimizer': ['adam', 'rmsprop'],
    'model__activation': ['relu']
}

# X_SAMPLE = x_train[:5000]
# Y_SAMPLE = y_train_cat[:5000]

keras_clf = KerasClassifier(
    model=create_model,
    epochs=30,
    callbacks=[es],
    verbose=0,
    validation_split=0.1
)

# Criamos o objeto GridSearchCV
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
# grid_result = grid.fit(X_SAMPLE, Y_SAMPLE)
print("Grid Search concluído!")

# Exibir os melhores resultados
print(
    f"Melhor score (acurácia média na validação cruzada): {grid_result.best_score_:.4f}")
print("Melhores parâmetros encontrados:")
print(grid_result.best_params_)

best_keras_model = grid_result.best_estimator_.model_
test_loss, test_acc = best_keras_model.evaluate(x_test, y_test_cat, verbose=0)

print(f"\nAcurácia do melhor modelo no conjunto de teste: {test_acc:.4f}")
print(f"Perda do melhor modelo no conjunto de teste: {test_loss:.4f}")


"""
Grid Search concluído!
Melhor score (acurácia média na validação cruzada): 0.9756
Melhores parâmetros encontrados:
{'batch_size': 64, 'model__activation': 'relu', 'model__dropout_rate': 0.4, 'model__hidden_layers': 1, 'model__neurons': 256, 'model__optimizer': 'adam'}

Acurácia do melhor modelo no conjunto de teste: 0.9798
Perda do melhor modelo no conjunto de teste: 0.0667
"""