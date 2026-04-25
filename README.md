# Neural Network

A feedforward neural network trained on MNIST, built from scratch with NumPy.

Optimization is handled by `UOSolver` (`uo.py`), a custom unconstrained optimization solver supporting Adam, AdamW, SGD with momentum, BFGS, and Newton-CG. Training uses mini-batch gradient descent with gradients computed via backpropagation.

## Setup

```bash
pip install -r requirements.txt
```

Download the MNIST binary files and place them under `data/`:

```
data/
  train-images-idx3-ubyte/train-images-idx3-ubyte
  train-labels-idx1-ubyte/train-labels-idx1-ubyte
  t10k-images-idx3-ubyte/t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte
```

## Usage

```bash
python nn.py
```

Prints per-epoch loss, final accuracy, a confusion matrix, and per-class precision/recall/F1.

## Configuration

Edit the `NeuralNetwork` constructor call in `main()`:

| Parameter | Description |
|---|---|
| `hidden_layers` | List of neuron counts per hidden layer |
| `activation_f_type` | `"ReLu"`, `"Sigmoid"`, or `"Tanh"` |
| `learning_rate` | Step size passed to UOSolver |
| `epochs` | Number of full passes over the training data |
| `batch_size` | Samples per mini-batch |
| `optimizer` | `"adam"`, `"adamw"`, `"sgdm"`, `"bfgs"`, `"newtoncg"` |
