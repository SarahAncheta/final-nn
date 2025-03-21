# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        if "sigmoid" in activation.lower():
            A_curr = self._sigmoid(Z_curr)
        
        elif "relu" in activation.lower():
            A_curr = self._relu(Z_curr)
        else:
            raise ValueError("Activation function not supported: can either be ReLu or sigmoid")
        
        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
   
        A_curr = X.T # Input becomes activation for first layer
        cache = {"A0": A_curr}

        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            W_curr = self._param_dict[f'W{layer_idx}']
            b_curr = self._param_dict[f'b{layer_idx}']
            activation = layer['activation']

            cache[f'A{layer_idx}'] = A_curr
            
            # Perform a single forward pass
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_curr, activation)

            # Store intermediate values for backpropagation
            cache[f'Z{layer_idx}'] = Z_curr
            cache[f'A{layer_idx}'] = A_curr

        return A_curr, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        # print(f"DEBUG: dA_curr entering _single_backprop for activation {activation_curr}:", dA_curr)

        if "sigmoid" in activation_curr.lower():
            
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        
        elif "relu" in activation_curr.lower():
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)

        else:
            raise ValueError("Activation function not supported: can either be relu or sigmoid")
        
        # print("DEBUG: dZ_curr in _single_backprop:", dZ_curr)

        m = A_prev.shape[1]

        dW_curr = (1 / m) * np.dot(dZ_curr, A_prev.T)
        db_curr = (1 / m) * np.sum(dZ_curr, axis=1, keepdims=True)
        dA_prev = np.dot(W_curr.T, dZ_curr)

        # print("DEBUG: dW_curr in _single_backprop:", dW_curr)
        # print("DEBUG: db_curr in _single_backprop:", db_curr)
        # print("DEBUG: dA_prev in _single_backprop:", dA_prev)

        return dA_prev, dW_curr, db_curr



    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        y = y.T
        
        grad_dict = {}  # Dictionary to store gradients

        L = len(self.arch)  # Number of layers

        # Compute dA for the last layer based on the loss function
        if "mse" in self._loss_func.lower():
            dA_curr = self._mean_squared_error_backprop(y, y_hat)
        elif "binary_cross_entropy" in self._loss_func.lower():
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)
        else:
            raise ValueError("Loss function not supported: can either bd 'mse' or 'binary_cross_entropy'.")

        # Backpropagate through each layer in reverse order
        for layer_idx in range(L, 0, -1):
            W_curr = self._param_dict[f'W{layer_idx}']
            b_curr = self._param_dict[f'b{layer_idx}']
            Z_curr = cache[f'Z{layer_idx}']
            A_prev = cache[f'A{layer_idx - 1}']  # Previous layer activation

            activation_curr = self.arch[layer_idx - 1]["activation"]

            # Compute gradients using single-layer backpropagation
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)

            # Store computed gradients
            grad_dict[f'dW{layer_idx}'] = dW_curr
            grad_dict[f'db{layer_idx}'] = db_curr

            # Move to the previous layer
            dA_curr = dA_prev

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for layer_idx in range(1, len(self.arch) + 1):  # Iterate through layers
            W_key = f"W{layer_idx}"
            b_key = f"b{layer_idx}"

            dW_key = f"dW{layer_idx}"
            db_key = f"db{layer_idx}"

            # Update weights and biases using gradient descent
            self._param_dict[W_key] -= self._lr * grad_dict[dW_key]  # Update weights
            self._param_dict[b_key] -= self._lr * grad_dict[db_key]  # Update biases

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        Trains the neural network using mini-batch gradient descent.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per-epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per-epoch loss for validation set.
        """
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        num_samples = X_train.shape[0]
        num_batches = num_samples // self._batch_size
        if num_samples % self._batch_size != 0:
            num_batches += 1  # Account for the last smaller batch

        for epoch in range(self._epochs):
            # Shuffle dataset
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            epoch_loss = 0  # Track training loss

            # Process mini-batches
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self._batch_size
                end_idx = min(start_idx + self._batch_size, num_samples)

                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]

                # Forward pass
                y_hat_batch, cache_batch = self.forward(X_batch)

                # Compute loss for the batch
                if self._loss_func == "mse":
                    loss_batch = self._mean_squared_error(y_batch.T, y_hat_batch)
                elif self._loss_func == "binary_cross_entropy":
                    loss_batch = self._binary_cross_entropy(y_batch.T, y_hat_batch)
                else:
                    raise ValueError("Loss function not supported. Use 'mse' or 'binary_cross_entropy'.")

                epoch_loss += loss_batch  # Accumulate loss

                # Backpropagation
                grad_dict = self.backprop(y_batch, y_hat_batch, cache_batch)

                # Update parameters
                self._update_params(grad_dict)

            # Store average training loss for the epoch
            per_epoch_loss_train.append(epoch_loss / num_batches)

            # Compute validation loss at the end of each epoch
            y_hat_val, _ = self.forward(X_val)
            if self._loss_func == "mse":
                loss_val = self._mean_squared_error(y_val.T, y_hat_val)
            elif self._loss_func == "binary_cross_entropy":
                loss_val = self._binary_cross_entropy(y_val.T, y_hat_val)

            per_epoch_loss_val.append(loss_val)

            print(f"Epoch {epoch+1}/{self._epochs} - Train Loss: {per_epoch_loss_train[-1]:.6f} - Val Loss: {loss_val:.6f}")

        return per_epoch_loss_train, per_epoch_loss_val

    # def fit(
    #     self,
    #     X_train: ArrayLike,
    #     y_train: ArrayLike,
    #     X_val: ArrayLike,
    #     y_val: ArrayLike
    # ) -> Tuple[List[float], List[float]]:
    #     """
    #     This function trains the neural network by backpropagation for the number of epochs defined at
    #     the initialization of this class instance.

    #     Args:
    #         X_train: ArrayLike
    #             Input features of training set.
    #         y_train: ArrayLike
    #             Labels for training set.
    #         X_val: ArrayLike
    #             Input features of validation set.
    #         y_val: ArrayLike
    #             Labels for validation set.

    #     Returns:
    #         per_epoch_loss_train: List[float]
    #             List of per epoch loss for training set.
    #         per_epoch_loss_val: List[float]
    #             List of per epoch loss for validation set.
    #     """
    #     per_epoch_loss_train = []
    #     per_epoch_loss_val = []

    #     for epoch in range(self._epochs):
    #         # Forward pass
    #         y_hat_train, cache_train = self.forward(X_train)

    #         # Compute loss
    #         if self._loss_func == "mse":
    #             loss_train = self._mean_squared_error(y_train.T, y_hat_train)
    #         elif self._loss_func == "binary_cross_entropy":
    #             loss_train = self._binary_cross_entropy(y_train.T, y_hat_train)
    #         else:
    #             raise ValueError("Loss function not supported. Use 'mse' or 'binary_cross_entropy'.")

    #         per_epoch_loss_train.append(loss_train)

    #         # Backpropagation
    #         grad_dict = self.backprop(y_train, y_hat_train, cache_train)

    #         # Update parameters
    #         self._update_params(grad_dict)

    #         # Validation loss
    #         y_hat_val, _ = self.forward(X_val)
    #         if self._loss_func == "mse":
    #             loss_val = self._mean_squared_error(y_val.T, y_hat_val)
    #         elif self._loss_func == "binary_cross_entropy":
    #             loss_val = self._binary_cross_entropy(y_val.T, y_hat_val)

    #         per_epoch_loss_val.append(loss_val)

    #         print(f"Epoch {epoch+1}/{self._epochs} - Train Loss: {loss_train:.6f} - Val Loss: {loss_val:.6f}")

    #     return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, _ = self.forward(X)  # Perform a forward pass
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        Z = np.asarray(Z)
        nl_transform = 1 / (1+np.exp(-Z))

        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dA = np.asarray(dA)
        Z = np.asarray(Z)

        sigmoid = 1 / (1+np.exp(-Z))

        dZ = dA * sigmoid * (1-sigmoid)
        
        return dZ

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        Z = np.asarray(Z)
        nl_transform = np.maximum(Z, 0)

        return nl_transform

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        Z = np.asarray(Z)
        dA = np.asarray(dA)

        dZ = dA * (Z > 0)

        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        y = np.asarray(y)
        y_hat = np.asarray(y_hat)

        #got this idea from ChatGPT, and also used it in the backprop (avoid division by 0)
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)

        loss = -np.mean(y*np.log(y_hat) + (1 - y)*np.log(1-y_hat))

        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        y = np.asarray(y)
        y_hat = np.asarray(y_hat)

        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        n = y.shape[0]

        dA = (1/n)*((y_hat - y)/(y_hat*(1-y_hat)))

        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        y = np.asarray(y)
        y_hat = np.asarray(y_hat)

        loss = np.mean((y - y_hat)**2)
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        y = np.asarray(y)
        y_hat = np.asarray(y_hat)

        n = y.shape[0]
        dA = (2/n)*(y_hat-y)

        return dA