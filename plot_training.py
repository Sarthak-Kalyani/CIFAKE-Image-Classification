import pickle
import matplotlib.pyplot as plt

# Load saved training history
with open("model/extension_history.pckl", "rb") as f:
    history = pickle.load(f)

epochs = range(1, len(history["accuracy"]) + 1)

# -----------------------------
# Accuracy Graph
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Modified CNN2D Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/training_validation_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------
# Loss Graph
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["loss"],
    label="Training Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    label="Validation Loss"
)

plt.title("Modified CNN2D Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/training_validation_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Training graphs generated successfully.")
print("Saved:")
print("results/training_validation_accuracy.png")
print("results/training_validation_loss.png")