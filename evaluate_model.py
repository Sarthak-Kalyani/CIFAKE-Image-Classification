import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# -----------------------------
# Load processed CIFAKE dataset
# -----------------------------
X = np.load("model/X.txt.npy").astype("float32") / 255.0
Y = np.load("model/Y.txt.npy")

print("Dataset shape:", X.shape)
print("Labels shape:", Y.shape)


# -----------------------------
# Reproducible 80/20 split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print("Training images:", X_train.shape[0])
print("Testing images:", X_test.shape[0])


# -----------------------------
# Modified CNN2D architecture
# -----------------------------
model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        input_shape=(32, 32, 3),
        activation="relu"
    )
)

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu"
    )
)

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(2, activation="softmax"))


# -----------------------------
# Load trained weights
# -----------------------------
model.load_weights("model/extension_weights.hdf5")


# -----------------------------
# Predictions
# -----------------------------
predictions = model.predict(
    X_test,
    batch_size=32,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)


# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)


print("\n========== FINAL MODEL RESULTS ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)