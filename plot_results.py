import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Reproduce the same test split used in evaluate_model.py
from sklearn.model_selection import train_test_split

X = np.load("model/X.txt.npy").astype("float32") / 255.0
Y = np.load("model/Y.txt.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# Load saved predictions generated using the model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        input_shape=(32, 32, 3),
        activation="relu"
    )
)

model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu"
    )
)

model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.load_weights("model/extension_weights.hdf5")

predictions = model.predict(
    X_test,
    batch_size=32,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Plot
plt.figure(figsize=(7, 6))

plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Modified CNN2D")
plt.colorbar()

classes = ["REAL", "FAKE"]

plt.xticks([0, 1], classes)
plt.yticks([0, 1], classes)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: results/confusion_matrix.png")