
import numpy as np
from number_detector import load_data, train, save, load, predict

if __name__ == "__main__":
    print("start")
    X_train, Y_train = load_data("data")
    print("X shape:", X_train.shape)   # sollte (2500, N)
    print("Y shape:", Y_train.shape)   # sollte (10, N)
    print("min/max X:", X_train.min(), X_train.max())
    print("unique labels count:", np.unique(np.argmax(Y_train, axis=0), return_counts=True))

    train(X_train, Y_train, epochs=10000)

    save("number_detector.npz")

# load("number_detector.npz")

# label, probs = predict("test/8_0001.png")
# print("Prediction Label:", label)
# print("probabilities:", probs)