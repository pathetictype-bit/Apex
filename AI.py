from encoder import encode_txt, encode
from network import Network
from decoder import decode
from RNN import rnn
from model_io import save_model, load_model, model_exists
from Charecter_library import library
from trainer import Trainer

network = Network([128, 256, 512, 1024, 1024, 512, 256, 128, len(library)])
rnn_model = rnn(len(library), 128)
trainer = Trainer(rnn_model, network, learning_rate=0.0005)

start_epoch = 0
if model_exists():
    start_epoch = load_model(rnn_model, network)

print("Hello there how can I help you today? 😊\n")

while True:
    receive_input = input("User > ")
    prompt = receive_input + "|"
    encoded = encode_txt(prompt)


    rnn_model.hidden_state = [0] * rnn_model.hidden_size
    rnn_model.history.clear()
    

    for x in encoded:
        h = rnn_model.forward(x, store_history=True)

    response = []
    first_predictions = network.forward(h)


    for _ in range(50):   # max response length

        prediction_vector = network.forward(h)
        current_char = decode(prediction_vector)

        if current_char == "~":
            break

        response.append(current_char)

        # 3. ONLY NOW encode it and step the RNN forward to compute the NEXT hidden state 'h'
        current_vector = encode(current_char)
        h = rnn_model.forward(current_vector, store_history=False)

    print(f"Apex > {''.join(response)}\n")
    if receive_input == "hm":
        break