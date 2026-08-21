import json
import os
FILE_NAME = "knowledge.json"

def model_exists():
    return os.path.exists(FILE_NAME)

def save_model(rnn_model, network, current_epoch):
    data = {
        "epoch": current_epoch,
        "rnn": rnn_model.get_weights(),
        "network": network.get_weights()
    }
    with open(FILE_NAME, "w") as file:
        json.dump(data, file, indent=4)

def load_model(rnn_model, network):
    with open(FILE_NAME, "r") as file:
        data = json.load(file)
    rnn_model.set_weights(data["rnn"])
    network.set_weights(data["network"])
    return data.get("epoch", 0)

