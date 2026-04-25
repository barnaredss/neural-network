from dataclasses import dataclass
import numpy as np
from load_data_mnist import load_mnist
from uo import UOSolver

@dataclass
class Layer:
    id: int
    incoming_weights: np.ndarray
    biases: np.ndarray
    activations: np.ndarray = None

class NeuralNetwork:
    def __init__(self, hidden_layers: list[int], n_inputs: int, n_outputs: int, activation_f_type: str, task: str = "multiclass", learning_rate: float = 0.01) -> None:
        """Initialization function"""
        
        self.hidden_layers: list[Layer] = []
        self.activation_f_type = activation_f_type
        self.task = task
        self.learning_rate = learning_rate
        self.errors: list[float] = []

        for i in range(len(hidden_layers)):
            input_size = n_inputs if i == 0 else hidden_layers[i - 1]
            W = np.random.randn(input_size, hidden_layers[i]) * np.sqrt(1 / input_size)
            b = np.random.uniform(-1, 1, size=(hidden_layers[i],))
            self.hidden_layers.append(Layer(id=i, incoming_weights=W, biases=b))

        output_input_size = hidden_layers[-1] if hidden_layers else n_inputs
        W_out = np.random.randn(output_input_size, n_outputs) * np.sqrt(1 / output_input_size)  # Compute weights for incoming w for output layer
        b_out = np.zeros(n_outputs)
        self.output_layer = Layer(id=len(hidden_layers), incoming_weights=W_out, biases=b_out)

    def _output_activation(self, x: np.ndarray) -> np.ndarray:
        """Activation function of last layer"""
        
        if self.task == "multiclass":
            e = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e / np.sum(e, axis=1, keepdims=True)
        if self.task == "binary":
            return 1 / (1 + np.exp(-x))
        if self.task == "regression":
            return x

    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        """Calculates activation for all layers except last for given type of activation function"""
        
        if self.activation_f_type == "ReLu":
            return np.maximum(0, x)
        if self.activation_f_type == "Sigmoid":
            return 1 / (1 + np.exp(-x))
        if self.activation_f_type == "Tanh":
            return np.tanh(x)

    def _cost_function(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Calculates error given a certain cost function"""
        
        if self.task == "multiclass":
            return -np.mean(np.sum(target * np.log(prediction + 1e-9), axis=1))
        if self.task == "binary":
            return -np.mean(target * np.log(prediction + 1e-9) + (1 - target) * np.log(1 - prediction + 1e-9))
        if self.task == "regression":
            return np.mean((prediction - target) ** 2)

    def _d_activation_function(self, x: np.ndarray) -> np.ndarray:
        """Returns output of derivative of activation function for a certain type of activation function"""
        
        if self.activation_f_type == "ReLu":
            return (x > 0).astype(float)
        if self.activation_f_type == "Sigmoid":
            return x * (1 - x)
        if self.activation_f_type == "Tanh":
            return 1 - np.tanh(x) ** 2
        
    def _d_cost_function(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Returns output of the derivative of cost function for a certain type of cost function"""
        
        return (prediction - target) / len(prediction)

    def _forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """Calculates forward propagation"""
        
        prev_layer = X
        for layer in self.hidden_layers:
            z = np.dot(prev_layer, layer.incoming_weights) + layer.biases
            layer.activations = self._activation_function(z)
            prev_layer = layer.activations

        z_out = np.dot(prev_layer, self.output_layer.incoming_weights) + self.output_layer.biases
        self.output_layer.activations = self._output_activation(z_out)
        return self.output_layer.activations

    def _flatten_params(self) -> np.ndarray:
        """Returns 1d array with all weights and biases flattened, ordered layer by layer"""
        
        parts: list[np.ndarray] = []
        for layer in self.hidden_layers:
            parts.append(layer.incoming_weights.flatten())
            parts.append(layer.biases.flatten())
        parts.append(self.output_layer.incoming_weights.flatten())
        parts.append(self.output_layer.biases.flatten())
        return np.concatenate(parts)

    def _unflatten_params(self, flat: np.ndarray) -> None:
        """Restores layer weights and biases from a flat 1d vector"""
        
        idx = 0
        for layer in self.hidden_layers:
            w_size = layer.incoming_weights.size
            b_size = layer.biases.size
            layer.incoming_weights = flat[idx:idx + w_size].reshape(layer.incoming_weights.shape)
            idx += w_size
            layer.biases = flat[idx:idx + b_size].copy()
            idx += b_size
        w_size = self.output_layer.incoming_weights.size
        b_size = self.output_layer.biases.size
        self.output_layer.incoming_weights = flat[idx:idx + w_size].reshape(self.output_layer.incoming_weights.shape)
        idx += w_size
        self.output_layer.biases = flat[idx:idx + b_size].copy()

    def _compute_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Runs forward pass then backpropagation, returning all gradients as a flat vector (same order as _flatten_params)"""
        
        prediction = self._forward_propagation(X)
        delta = self._d_cost_function(prediction, y)
        prev_act = self.hidden_layers[-1].activations if self.hidden_layers else X
        dW_out = prev_act.T @ delta
        db_out = np.sum(delta, axis=0)
        
        hidden_grads: list[np.ndarray] = []
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            next_weights = self.output_layer.incoming_weights if i == len(self.hidden_layers) - 1 else self.hidden_layers[i + 1].incoming_weights
            delta = (delta @ next_weights.T) * self._d_activation_function(self.hidden_layers[i].activations)
            prev_activations = X if i == 0 else self.hidden_layers[i - 1].activations
            hidden_grads.append((prev_activations.T @ delta, np.sum(delta, axis=0)))
            
        parts: list[np.ndarray] = []
        for dW, db in reversed(hidden_grads):
            parts.append(dW.flatten())
            parts.append(db.flatten())
        parts.append(dW_out.flatten())
        parts.append(db_out.flatten())
        return np.concatenate(parts)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int = 32, optimizer: str = "adam") -> None:
        """Trains neural network with X, y dataset using UOSolver and mini-batch gradient descent"""
        
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        self._batch_X = X_train
        self._batch_y = y_train

        def loss_fn(params):
            self._unflatten_params(params)
            pred = self._forward_propagation(self._batch_X)
            return float(self._cost_function(pred, self._batch_y))

        def grad_fn(params):
            self._unflatten_params(params)
            return self._compute_gradient(self._batch_X, self._batch_y)

        solver = UOSolver(
            f=loss_fn,
            g=grad_fn,
            direction=optimizer,
            dimensions=len(self._flatten_params()),
            x0=self._flatten_params().copy(),
            adam_alpha=self.learning_rate,
            lr=self.learning_rate,
        )

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            epoch_loss = 0.0

            for start in range(0, n_samples, batch_size):
                X_batch = X_shuffled[start:start + batch_size]
                y_batch = y_shuffled[start:start + batch_size]
                self._batch_X = X_batch
                self._batch_y = y_batch

                gradient = self._compute_gradient(X_batch, y_batch)
                solver.x_1 = solver.x.copy()
                d, alpha = solver.step(gradient)
                solver.k += 1
                solver.x = solver.x - alpha * d
                self._unflatten_params(solver.x)

                pred = self._forward_propagation(X_batch)
                epoch_loss += self._cost_function(pred, y_batch)

            avg_loss = epoch_loss / n_batches
            self.errors.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes nn predict for given X data"""
        
        return self._forward_propagation(X)

def main() -> None:
    """Main function, makes predictions for MNIST dataset"""
    
    X_train, X_test, y_train, y_test = load_mnist()

    nn = NeuralNetwork(hidden_layers=[256, 10], n_inputs=X_train.shape[1], n_outputs=y_train.shape[1], activation_f_type="Tanh", task="multiclass", learning_rate=0.001)
    nn.train(X_train.values, y_train.values, epochs=10, batch_size=64, optimizer="adam")

    predictions = nn.predict(X_test.values)

    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test.values, axis=1)

    accuracy = np.mean(pred_classes == true_classes)
    print(f"Accuracy: {accuracy:.4f}")

    n_classes = y_train.shape[1]
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(true_classes, pred_classes):
        confusion[true][pred] += 1

    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(confusion)

    print("\nPer-class metrics:")
    print(f"{'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for c in range(n_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"{c:<8} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")


if __name__ == "__main__":
    main()
