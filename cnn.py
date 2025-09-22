from main import *
from mlp import es
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.models import Sequential # type: ignore

cnn = Sequential([
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam', loss='categorical_crossentropy',
            metrics=['accuracy'])
history_cnn = cnn.fit(x_train_cnn, y_train_cat, validation_split=0.1,
                      epochs=12, batch_size=128, callbacks=[es])

# predição e relatório (para CNN)
y_pred_proba = cnn.predict(x_test_cnn)
y_pred = y_pred_proba.argmax(axis=1)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True'); plt.show()