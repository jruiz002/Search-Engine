import numpy as np
from typing import List, Tuple
import pickle

class MLP:   
    def __init__(self, input_size: int = 3, hidden_sizes: List[int] = [32, 16], output_size: int = 11, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
        self.layers = []
        
        # Construir arquitectura de la red
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Inicializar pesos y sesgos para cada capa
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # Inicialización Xavier/Glorot
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i + 1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        # Prevenir overflow
        x = np.clip(x, -250, 250)
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        # Prevenir overflow restando el máximo
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        activations = [X]  # Incluir entrada como primera activación
        z_values = []
        
        current_input = X
        
        # Forward pass a través de todas las capas
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(current_input, weight) + bias
            z_values.append(z)
            
            # Usar ReLU para capas ocultas y softmax para la capa de salida
            if i == len(self.weights) - 1:  # Última capa
                activation = self._softmax(z)
            else:  # Capas ocultas
                activation = self._relu(z)
            
            activations.append(activation)
            current_input = activation
        
        return current_input, activations, z_values
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], z_values: List[np.ndarray]) -> None:
        m = X.shape[0]  # Número de ejemplos
        
        # Convertir etiquetas a one-hot si no lo están ya
        if len(y.shape) == 1:
            n_classes = activations[-1].shape[1]  # Obtener número de clases de la salida
            y_onehot = np.eye(n_classes)[y]
        else:
            y_onehot = y
        
        # Calcular error de la capa de salida
        dz = activations[-1] - y_onehot
        
        # Backpropagation a través de todas las capas
        for i in reversed(range(len(self.weights))):
            # Gradientes para pesos y sesgos
            dw = np.dot(activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            # Actualizar pesos y sesgos
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db
            
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self._relu_derivative(activations[i])
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Convertir etiquetas a one-hot si no lo están ya
        if len(y_true.shape) == 1:
            n_classes = y_pred.shape[1]
            y_onehot = np.eye(n_classes)[y_true]
        else:
            y_onehot = y_true
        
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_onehot * np.log(y_pred_clipped)) / y_true.shape[0]
        return loss
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 100, batch_size: int = 32, verbose: bool = True) -> List[float]:
        losses = []
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Barajar datos
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0.0
            
            # Mini-batch SGD
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                # Forward pass
                output, activations, z_values = self.forward(X_batch)
                
                # Backward pass
                self.backward(X_batch, y_batch, activations, z_values)
                
                # Calcular pérdida para este batch
                batch_loss = self.compute_loss(y_batch, output)
                epoch_loss += batch_loss * len(X_batch)
            
            # Pérdida promedio de la época
            avg_loss = epoch_loss / n_samples
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        output, _, _ = self.forward(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        output, _, _ = self.forward(X)
        return output
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy
    
    def save(self, filepath: str) -> None:
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'learning_rate': self.learning_rate
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.learning_rate = model_data['learning_rate']