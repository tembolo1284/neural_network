
# Rust Neural Network Example
Simple feedforward neural network to learn XOR in Rust.

We have 2 neurons for the input layer, 2 neurons for the hidden layer, and 1 neuron for the output layer. 

This is a simple implementation of a feedforward neural network in Rust. The neural network has one hidden layer and uses the sigmoid activation function. It is trained using backpropagation.

## Dependencies

This code relies on the `rand` crate for random number generation. You can add it to your project's dependencies by including it in your `Cargo.toml` file:

```
[dependencies]
rand = "0.8"
```

## Code Explanation

### Activation Functions

Two activation functions are defined:

1. sigmoid(x: f64) -> f64: Implements the sigmoid activation function.
2. dsigmoid(x: f64) -> f64: Implements the derivative of the sigmoid function.

## Weight Initialization
The init_weights() function generates random initial weights between 0.0 and 1.0.

## Shuffling
The shuffle(vec: &mut Vec<usize>, n: usize) function shuffles a vector of indices. It is used for randomizing the order of training samples.

## Neural Network Configuration

The neural network is configured with the following constants:

* NUM_INPUTS: Number of input nodes (2).
* NUM_HIDDEN_NODES: Number of nodes in the hidden layer (2).
* NUM_OUTPUTS: Number of output nodes (1).
* NUM_TRAINING_SETS: Number of training sets (4).
* LR: Learning rate (0.2).

The weights and biases for the hidden and output layers are initialized as empty vectors with appropriate dimensions. Training data and corresponding outputs are also defined.

## Training Loop

The main training loop runs for a specified number of epochs (10,000). It performs the following steps for each epoch:

1. Shuffle the training set order.

2. For each training sample:
    * Perform a forward pass through the neural network to compute predictions.
    * Calculate and print the error and predicted output.
    * Perform backpropagation to update weights and biases.

## Backpropagation

Backpropagation is used to update the weights and biases of the neural network. It consists of the following steps:

1. Calculate the output layer error and deltas.
2. Calculate the hidden layer error and deltas.
3. Update output layer biases and weights.
4. Update hidden layer biases and weights.

## Printing Weights
After training, the final weights and biases for the hidden and output layers are printed to the console.

## Usage
To run the code, ensure you have the rand crate added to your dependencies. You can execute the code using:

```
cargo run
```