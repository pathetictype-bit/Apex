import math

class Trainer:
    def __init__(self, model, learning_rate):
        self.model = model
        self.backend = model.backend
        self.learning_rate = learning_rate

    def train_step(self, inputs, targets):
        hidden_states, previous_hidden_state, network_outputs, embeddings = \
            self.model.forward(inputs)
        losses = []
        doutputs = []
        supervised_target_count = sum(target is not None for target in targets)
        if supervised_target_count == 0:
            raise ValueError("Training step has no supervised targets.")
        gradient_scale = 1.0 / supervised_target_count

        for i, (output, target) in enumerate(zip(network_outputs, targets)):
            if target is None:
                if hasattr(output, "rows"):
                    output_rows, output_columns = output.rows, output.columns
                else:
                    output_rows, output_columns = output.shape
                doutputs.append(self.backend.zeros(output_rows, output_columns))
                continue

            loss, doutput = self.backend.loss(
                output,
                target,
                gradient_scale=gradient_scale,
            )



            if not getattr(self.backend, "loss_validates_gradient", False):
                if hasattr(doutput, "to_cpu"):
                    doutput_cpu = doutput.to_cpu()
                else:
                    doutput_cpu = doutput.data

                if not all(
                    math.isfinite(value)
                    for row in doutput_cpu
                    for value in row
                ):
                    print("Non-finite output gradient detected.")
                    raise RuntimeError("Non-finite doutput detected")

            losses.append(loss)
            doutputs.append(doutput)

        gradient = self.model.backward(
            inputs,
            hidden_states,
            network_outputs,
            previous_hidden_state,
            doutputs
        )

        (
            dWxh,
            dWhh,
            dbh,
            network_weight_gradients,
            network_bias_gradients,
            embedding_gradients
        ) = gradient

        self.model.update_parameters(
            self.learning_rate,
            dWxh,
            dWhh,
            dbh,
            network_weight_gradients,
            network_bias_gradients,
            embedding_gradients
        )

        return losses