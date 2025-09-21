"""
Import do dataset MNIST e pré-processamento
"""
#uso do mnist já incluso no keras 

from tensorflow.keras.datasets import mnist # type: ignore
import matplotlib.pyplot as plt
import numpy as np

# keras (baixa o dataset, divide dados, retorna na tupla)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if __name__ == "__main__":
    print(f'x_train: {x_train.shape} \ny_train: {y_train.shape} \nx_test {x_test.shape} \ny_test {y_test.shape}')

    fig, axes = plt.subplots(2,5, figsize=(10,5))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(x_train[i],cmap='gray')
        ax.set_title(int(y_train[i]))
        ax.axis('off')
    plt.show()

#Pré-processamento

# normalizar train e test
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

# formato para scikit-learn: achatar
x_train_flat = x_train.reshape(-1,28*28)
x_test_flat = x_test.reshape(-1,  28*28)

# formato para cnn: adicionar 1 canal
x_train_cnn = x_train.reshape(-1,28,28,1)
x_test_cnn = x_test.reshape(-1,28,28,1)

# one-hot para Keras
from tensorflow.keras.utils import to_categorical # type: ignore
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test, 10)
