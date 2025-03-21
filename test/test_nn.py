# TODO: import dependencies and write unit tests below

import pytest
import numpy as np
from nn import NeuralNetwork
from nn import preprocess

def test_single_forward():
    nn_arch = [
        {"input_dim": 2, "output_dim": 3, "activation": "relu"},
        {"input_dim": 3, "output_dim": 1, "activation": "sigmoid"}
    ]

    # Instantiate the model with required parameters
    model = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mse")

    W_curr = np.array([[0.15, 0.20], [0.25, 0.30]], dtype=np.float64)  # Weights (2x2)
    b_curr = np.array([[0.35], [0.35]], dtype=np.float64)  # Biases (2x1)
    A_prev = np.array([[0.05], [0.10]], dtype=np.float64)

    A_curr, Z_curr = model._single_forward(W_curr, b_curr, A_prev, 'sigmoid')

    assert A_curr == pytest.approx(np.array([[0.59326999], [0.59688438]]), rel=1e-6, abs=1e-6)
    assert Z_curr == pytest.approx(np.array([[0.3775], [0.3925]]), rel=1e-6, abs=1e-6)

def test_forward():
    """
    Test the forward function using a small example and comparing to manually computed values.
    """

    # Define a simple 2-layer network architecture
    nn_arch = [
        {"input_dim": 2, "output_dim": 2, "activation": "sigmoid"},
        {"input_dim": 2, "output_dim": 1, "activation": "sigmoid"}
    ]

    # Instantiate the model
    model = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mse")

    # Define a small example input (2x1)
    X = np.array([[0.05], [0.10]], dtype=np.float64).T  # Input matrix

    # Perform forward pass
    output, cache = model.forward(X)

    # Expected values computed manually
    expected_Z1 = np.array([[-0.02231441], [-0.00494495]], dtype=np.float64)
    expected_A1 = np.array([[0.49442163], [0.49876376]], dtype=np.float64)
    expected_Z2 = np.array([[0.06940912]], dtype=np.float64)
    expected_A2 = np.array([[0.51734532]], dtype=np.float64)

    assert cache["Z1"] == pytest.approx(expected_Z1, rel=1e-6, abs=1e-6)
    assert cache["A1"] == pytest.approx(expected_A1, rel=1e-6, abs=1e-6)
    assert cache["Z2"] == pytest.approx(expected_Z2, rel=1e-6, abs=1e-6)
    assert output == pytest.approx(expected_A2, rel=1e-6, abs=1e-6)


def test_single_backprop():
    """
    Test the single backprop function using a small example and comparing to manually calculated values.
    """

    # Define small example inputs (explicit values)
    W_curr = np.array([[0.15, 0.20], [0.25, 0.30]], dtype=np.float64)  # Weights (2x2)
    b_curr = np.array([[0.35], [0.35]], dtype=np.float64)  # Biases (2x1)
    Z_curr = np.array([[0.3775], [0.3925]], dtype=np.float64)  # Linear transformation (2x1)
    A_prev = np.array([[0.05], [0.10]], dtype=np.float64)  # Previous layer activation (2x1)
    dA_curr = np.array([[0.2], [0.3]], dtype=np.float64)  # Partial derivative of loss (2x1)


    nn_arch = [
        {"input_dim": 2, "output_dim": 3, "activation": "sigmoid"},
        {"input_dim": 3, "output_dim": 1, "activation": "sigmoid"}
    ]

    # Instantiate the model with required parameters
    model = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mse")

    computed_dA_prev, computed_dW_curr, computed_db_curr = model._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, "sigmoid")

    # Assertions to ensure computed values match the expected manually computed values

    expected_dW = np.array([[0.00241301, 0.00482601], [0.00360920, 0.00721840]], dtype=np.float64)
    expected_db = np.array([[0.04826014], [0.07218403]], dtype=np.float64)
    expected_dA = np.array([[0.02528503], [0.03130724]], dtype=np.float64)
    assert computed_dW_curr == pytest.approx(expected_dW, rel=1e-6, abs=1e-6)
    assert computed_db_curr == pytest.approx(expected_db, rel=1e-6, abs=1e-6)
    assert computed_dA_prev == pytest.approx(expected_dA, rel=1e-6, abs=1e-6)

def test_predict():
    # Define a tiny architecture: input dim 2 -> hidden dim 2 -> output dim 1
    arch = [
        {'input_dim': 2, 'output_dim': 2, 'activation': 'relu'},
        {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}
    ]

    # Create neural network instance
    model = NeuralNetwork(
        nn_arch=arch,
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=1,
        loss_function="binary_cross_entropy"
    )

    # Tiny input: 2 features, 1 sample
    X = np.array([[0.5, -1.2]])

    # Predict
    y_hat = model.predict(X)

    # Check output shape: should be (1, 1) => (output_dim, batch_size)
    assert y_hat.shape == (1, 1), f"Expected shape (1,1), got {y_hat.shape}"

    # Check values are between 0 and 1 due to sigmoid activation
    assert np.all((0 <= y_hat) & (y_hat <= 1)), "Output is not in [0, 1] range"

    # Check it's a float array
    assert isinstance(y_hat, np.ndarray), "Output is not a numpy array"
    assert y_hat.dtype in [np.float32, np.float64], "Output dtype is not float"


def test_binary_cross_entropy():
    # Instantiate the model with required parameters
    nn_arch = [
        {"input_dim": 2, "output_dim": 3, "activation": "sigmoid"},
        {"input_dim": 3, "output_dim": 1, "activation": "sigmoid"}
    ]
    model = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mse")

    # Tiny example
    y = np.array([[1], [0], [1]], dtype=np.float32)
    y_hat = np.array([[0.9], [0.1], [0.8]], dtype=np.float32)

    # Expected value computed manually
    # -1/3 * [log(0.9) + log(1 - 0.1) + log(0.8)]
    expected = -np.mean([
        np.log(0.9),       # for y=1, y_hat=0.9
        np.log(1 - 0.1),   # for y=0, y_hat=0.1
        np.log(0.8)        # for y=1, y_hat=0.8
    ])

    result = model._binary_cross_entropy(y, y_hat)

    assert isinstance(result, np.float32), "Output should be a float"
    assert np.isclose(result, expected, atol=1e-6), f"Expected {expected}, got {result}"

def test_binary_cross_entropy_backprop():
    # Instantiate the model with required parameters
    nn_arch = [
        {"input_dim": 2, "output_dim": 3, "activation": "sigmoid"},
        {"input_dim": 3, "output_dim": 1, "activation": "sigmoid"}
    ]
    model = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mse")

    y = np.array([[1], [0], [1]], dtype=np.float32)
    y_hat = np.array([[0.9], [0.1], [0.8]], dtype=np.float32)

    # Manually compute expected result
    # n = 3
    expected = (1/3) * ((y_hat - y) / (y_hat * (1 - y_hat)))

    # Run the method
    result = model._binary_cross_entropy_backprop(y, y_hat)

    assert result.shape == expected.shape, "Shape mismatch"
    assert np.allclose(result, expected, atol=1e-6), f"Backprop output incorrect:\nExpected:\n{expected}\nGot:\n{result}"

def test_relu_backprop():
    Z = np.array([[0.5, -1.0, 0.0], [1.2, -0.5, 0.7]])  # Linear transform (2x3)
    dA = np.array([[0.2, 0.1, 0.3], [0.4, 0.3, 0.2]])  # Partial derivative (2x3)

    relu_derivative = np.where(Z > 0, 1, 0)  # ReLU derivative
    expected_dZ = dA * relu_derivative

    nn_arch = [
        {"input_dim": 2, "output_dim": 3, "activation": "relu"},
        {"input_dim": 3, "output_dim": 1, "activation": "sigmoid"}
    ]

    # Instantiate the model with required parameters
    model = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mse")
    dZ = model._relu_backprop(dA, Z)
    
    assert dZ == pytest.approx(expected_dZ, rel=1e-6, abs=1e-6)

def test_mean_squared_error():
    pass

def test_mean_squared_error_backprop():
    y = np.array([[1, 0, 1], [0, 1, 0]])  # Ground truth (2x3)
    y_hat = np.array([[0.8, 0.2, 0.9], [0.1, 0.9, 0.2]])  # Predictions (2x3)

    expected_dA = (2 / y.shape[0]) * (y_hat - y)  # Expected derivative

    nn_arch = [
        {"input_dim": 2, "output_dim": 3, "activation": "relu"},
        {"input_dim": 3, "output_dim": 1, "activation": "sigmoid"}
    ]

    # Instantiate the model with required parameters
    model = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mse")

    dA = model._mean_squared_error_backprop(y, y_hat)

    assert dA == pytest.approx(expected_dA, rel=1e-6, abs=1e-6)

def test_sigmoid_backprop():
    Z = np.array([[0.5, -1.0, 0.0], [1.2, -0.5, 0.7]])  # Linear transform (2x3)
    dA = np.array([[0.2, 0.1, 0.3], [0.4, 0.3, 0.2]])  # Partial derivative (2x3)

    sigmoid = 1 / (1 + np.exp(-Z))
    expected_dZ = dA * sigmoid * (1 - sigmoid)  # Expected derivative

    nn_arch = [
        {"input_dim": 2, "output_dim": 3, "activation": "relu"},
        {"input_dim": 3, "output_dim": 1, "activation": "sigmoid"}
    ]

    # Instantiate the model with required parameters
    model = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mse")

    dZ = model._sigmoid_backprop(dA, Z)

    assert dZ == pytest.approx(expected_dZ, rel=1e-6, abs=1e-6)



def test_sample_seqs(): 
    # Example input
    seqs = ["AG", "CT", "G", "TGA"]  # lengths: 2, 2, 1, 3
    labels = [True, False, False, False]  # 1 positive, 3 negatives

    # Run sampling
    sampled_seqs, sampled_labels = preprocess.sample_seqs(seqs, labels)

    # Check that the number of sequences matches the number of labels
    assert len(sampled_seqs) == len(sampled_labels), "Mismatched number of sequences and labels"

    # Expect 3 positives and 3 negatives after upsampling
    num_pos = sum(sampled_labels)
    num_neg = len(sampled_labels) - num_pos

    assert num_pos == num_neg, "Labels are not balanced after sampling"
    assert len(sampled_seqs) == 6, "Total number of samples should match balanced class size"

    # Check all sequences are the same length
    seq_lengths = [len(seq) for seq in sampled_seqs]
    first_len = seq_lengths[0]
    for length in seq_lengths:
        assert length == first_len, "Sequences are not all padded to the same length"

def test_one_hot_encode_seqs():
    # Example input
    input_seqs = ["AG", "CT", "TP"]  # 3 sequences of length 2, which already include padding P

    # Expected output:
    # "A" -> [1, 0, 0, 0]
    # "G" -> [0, 0, 0, 1]
    # "C" -> [0, 0, 1, 0]
    # "T" -> [0, 1, 0, 0]
    # "P" -> [2, 2, 2, 2]

    result = preprocess.one_hot_encode_seqs(input_seqs)

    expected_output = np.array([
        [1, 0, 0, 0, 0, 0, 0, 1],      # AG
        [0, 0, 1, 0, 0, 1, 0, 0],      # CT
        [0, 1, 0, 0, 2, 2, 2, 2]       # T + P padding
    ], dtype=np.float32)

    assert result.shape == expected_output.shape, f"Shape mismatch: {result.shape} vs {expected_output.shape}"
    assert np.array_equal(result, expected_output), "Encoded values do not match expected output"