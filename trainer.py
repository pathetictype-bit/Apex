from network import Network
from RNN import rnn
from encoder import encode_txt, encode
import math

class Trainer:
    def __init__(self, model, network, learning_rate):
        self.model = model
        self.network = network
        self.learning_rate = learning_rate

    def calculate_loss(self, prediction, target):
        eps = 1e-12

        loss = 0
        for p, t in zip(prediction, target):
            if t == 1:
                loss -= math.log(max(p, eps))

        return loss

    def calculate_gradient(self, prediction, target):
        return [p - t for p, t in zip(prediction, target)]
    
    def train_step(self, input_txt, target_txt):
        hidden_gradients = []
        self.model.hidden_state = [0] * self.model.hidden_size
        self.model.history.clear()
        self.network.reset_gradients()
        sequence = input_txt + "|" + target_txt + "~"
        encoded_sequence = encode_txt(sequence)


        total_loss = 0
        for i in range(len(encoded_sequence) - 1):

            char_vector = encoded_sequence[i]
            target = encoded_sequence[i + 1]

            hidden = self.model.forward(char_vector)
            prediction = self.network.forward(hidden)

            if any(math.isnan(x) or math.isinf(x) for x in prediction):
                print("🔥 NaN/INF in prediction")
                print("Timestep:", i)
                print("Prediction:", prediction)
                return float("nan")

            total_loss += self.calculate_loss(prediction, target)

            gradient = self.calculate_gradient(prediction, target)

            hidden_gradient = self.network.backward(gradient)

            if any(math.isnan(x) or math.isinf(x) for x in hidden_gradient):
                print("🔥 NaN/INF in network gradient")
                return float("nan")

            hidden_gradients.append(hidden_gradient)
        rnn_gradients = self.model.backward(hidden_gradients)

        self.network.update_parameters(self.learning_rate)
        self.model.update_parameter(rnn_gradients, self.learning_rate)

        return total_loss / (len(encoded_sequence) - 1)
    
    def load_dataset(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        dataset = []

        for i in range(0, len(lines), 2):
            input_txt = lines[i]
            target_txt = lines[i + 1]

            dataset.append((input_txt, target_txt))

        return dataset
    def train(self, filename, epochs, start_epoch=0):
        dataset = self.load_dataset(filename)

        for epoch in range(start_epoch, epochs):

            total_loss = 0

            for input_txt, target_txt in dataset:
                loss = self.train_step(input_txt, target_txt)
                total_loss += loss

            average_loss = total_loss / len(dataset)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {average_loss}"
            )
            from model_io import save_model
            save_model(self.model, self.network, epoch + 1)
            print(f"[SAFE] Epoch {epoch + 1} written to disk. Ready for interrupt.\n")

            