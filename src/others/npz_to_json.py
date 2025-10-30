import numpy as np
import json

filename = "models/nn_number_detector_0001.npz"
output = "nn_number_detector_0001.json"

data = np.load(filename, allow_pickle=True)
weights = list(data["W"])
bias = list(data["b"])

input_size = weights[0].shape[1]
hidden_layers = [w.shape[0] for w in weights[:-1]]
output_size = weights[-1].shape[0]

model = []
for W, b in zip(weights, bias):
    model.append({
        "W": W.tolist(),              # numpy → normale Python-Liste
        "b": b.flatten().tolist()     # (rows, 1) → [rows]
    })

model_data = {
    "input_size": int(input_size),
    "hidden_layers": hidden_layers,
    "output_size": int(output_size),
    "model": model
}

model_json = json.dumps(model)

with open(output, "w") as f:
    json.dump(model_data, f, indent=2)

print("✅ model.json erfolgreich exportiert!")

# print(model_json)