import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from main import *
import numpy as np
import time
from keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from keras.callbacks import EarlyStopping
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

start_time = time.perf_counter()

es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)


def create_model_cnn(conv_filters1=32,
                     conv_filters2=64,
                     kernel_size=(3, 3),
                     dense_units=128,
                     dropout_rate=0.3,
                     optimizer="adam",
                     activation="relu"):
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))

    # Primeira camada convolucional
    model.add(Conv2D(conv_filters1, kernel_size, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Segunda camada convolucional
    model.add(Conv2D(conv_filters2, kernel_size, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten + Dropout + Camadas densas
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation=activation))
    model.add(Dense(10, activation="softmax"))

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


cnn_model = KerasClassifier(model=create_model_cnn,
                            verbose=0, validation_split=0.1)

search_spaces = {
    "model__conv_filters1": Categorical([16, 32]),
    "model__conv_filters2": Categorical([32, 64]),
    "model__dense_units": Categorical([64, 128]),
    "model__dropout_rate": Real(0.3, 0.5, prior="uniform"),
    "model__optimizer": Categorical(["adam"]),
    "batch_size": Categorical([64, 128]),
    "epochs": Categorical([5])
}


bayes_search = BayesSearchCV(
    estimator=cnn_model,
    search_spaces=search_spaces,
    n_iter=5,             
    cv=3,                  
    n_jobs=-1,             
    scoring='accuracy',
    verbose=1,
    refit=True,
    random_state=42
)

bayes_search_result = bayes_search.fit(
    x_train_cnn, y_train_cat, callbacks=[es])


print(
    f"Melhor score (acurácia média na validação cruzada): {bayes_search_result.best_score_:.4f}")  #type:ignore
print("Melhores parâmetros encontrados:")
print(bayes_search_result.best_params_) #type:ignore

best_keras_model = bayes_search_result.best_estimator_.model_  #type:ignore
test_loss, test_acc = best_keras_model.evaluate(
    x_test_cnn, y_test_cat, verbose=0)

print(f"\nAcurácia do melhor modelo no conjunto de teste: {test_acc:.4f}")
print(f"Perda do melhor modelo no conjunto de teste: {test_loss:.4f}")

# Refit the best model to get the history object
best_bayes_search_model = bayes_search_result.best_estimator_.model_  # type:ignore
history_best_bayes_search = best_bayes_search_model.fit(x_train_cnn, y_train_cat, validation_split=0.1, epochs=12,
                                                        batch_size=bayes_search_result.best_params_[  # type:ignore
                                                            'batch_size'],
                                                        callbacks=[es])
end_time = time.perf_counter()
elapsed_time = (end_time - start_time)/60

train_loss_best = history_best_bayes_search.history['loss']
val_loss_best = history_best_bayes_search.history['val_loss']
train_acc_best = history_best_bayes_search.history['accuracy']
val_acc_best = history_best_bayes_search.history['val_accuracy']

print("\n--- Análise Detalhada do Melhor Modelo MLP no Conjunto de Teste ---")

y_pred_proba = best_keras_model.predict(x_test)
y_pred = np.argmax(y_pred_proba, axis=1)

y_test_labels = np.argmax(y_test_cat, axis=1)

print("\n--- Relatório de Classificação ---")
print(classification_report(y_test_labels, y_pred))

print("\n--- Matriz de Confusão ---")

cm = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True'); plt.show()

print("\n--- Cálculo da Especificidade por Classe ---")
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
    print("-"*15)

epochs_best = range(len(train_loss_best))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

# Gráfico (a) Loss
ax1.plot(epochs_best, train_loss_best, label='Train')
ax1.plot(epochs_best, val_loss_best, label='Validation')
ax1.set_title('Loss do Melhor Modelo CNN - Bayesiana')
ax1.set_xlabel('epoch\n\n(a) Loss')
ax1.legend()

# Gráfico (b) Accuracy
ax2.plot(epochs_best, train_acc_best, label='Train')
ax2.plot(epochs_best, val_acc_best, label='Validation')
ax2.set_title('Accuracy do Melhor Modelo CNN - Bayesiana')
ax2.set_xlabel('epoch\n\n(b) Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()

with open("results_cnn.txt", "a") as f:
    f.write("CNN - Bayesiana v2: \n")
    f.write(f"Time: {elapsed_time} min \n")
    f.write("Opção para a questão do erro de tupla \n")
    f.write(f"Melhor score: {bayes_search_result.best_score_:.4f}\n")  #type:ignore
    f.write(str(bayes_search_result.best_params_))  #type:ignore
    f.write("\n")

    f.write(f"Média treino accuracy: {np.mean(train_acc_best)}\n")
    f.write(f"Média validação accuracy: {np.mean(val_acc_best)}\n")
    f.write(f"Média treino loss: {np.mean(train_loss_best)}\n")
    f.write(f"Média validação loss: {np.mean(val_loss_best)}\n")

    f.write(f"Acurácia no teste: {test_acc:.4f}\n")
    f.write(f"Perda no teste: {test_loss:.4f}\n")
    f.close()

print("Resultados salvos em results_cnn.txt")
