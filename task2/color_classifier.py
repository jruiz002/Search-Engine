import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, List
import os

try:
    from .neural_network import MLP
except ImportError:
    from neural_network import MLP

class ColorClassifier:
    
    def __init__(self, dataset_path: str = "assets/final_data_colors.csv"):
        self.dataset_path = dataset_path
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Mapeo de etiquetas de color a costos de movimiento
        self.cost_mapping = {
            'White': 1,    # Camino libre - fácil de transitar
            'Grey': 1,     # Pavimento - fácil de transitar  
            'Brown': 2,    # Tierra - moderadamente fácil
            'Green': 3,    # Grama - ligeramente difícil
            'Pink': 4,     # Flores - obstáculos menores
            'Yellow': 5,   # Arena - requiere más esfuerzo
            'Purple': 8,   # Pantano - muy difícil
            'Blue': 10,    # Agua - muy costoso pero posible
            'Orange': 12,  # Fuego - extremadamente peligroso
            'Red': 15,     # Lava - casi imposible
            'Black': 999   # Obstáculo completo
        }
        
        # Mapeo inverso para facilitar búsquedas
        self.cost_to_material = {v: k for k, v in self.cost_mapping.items()}
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset no encontrado: {self.dataset_path}")
        
        # Cargar dataset
        df = pd.read_csv(self.dataset_path)
        
        # Verificar columnas requeridas
        required_cols = ['red', 'green', 'blue', 'label']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Dataset debe contener columnas: {required_cols}")
        
        # Extraer características RGB
        X = df[['red', 'green', 'blue']].values.astype(np.float32)
        
        # Normalizar RGB a [0, 1]
        X = X / 255.0
        
        # Codificar etiquetas
        y = self.label_encoder.fit_transform(df['label'].values)
        
        print(f"Dataset cargado: {len(X)} muestras, {len(np.unique(y))} clases")
        print(f"Clases encontradas: {list(self.label_encoder.classes_)}")
        
        return X, y
    
    def train_model(self, test_size: float = 0.2, epochs: int = 150, batch_size: int = 64, 
                   hidden_sizes: List[int] = [64, 32], learning_rate: float = 0.01) -> Dict:

        # Cargar datos
        X, y = self.load_data()
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Conjunto de entrenamiento: {len(X_train)} muestras")
        print(f"Conjunto de prueba: {len(X_test)} muestras")
        
        # Verificar que las etiquetas estén en el rango correcto
        n_classes = len(self.label_encoder.classes_)
        max_label = max(max(y_train), max(y_test))
        min_label = min(min(y_train), min(y_test))
        
        print(f"Número de clases: {n_classes}")
        print(f"Rango de etiquetas: {min_label} - {max_label}")
        
        if max_label >= n_classes:
            raise ValueError(f"Etiqueta máxima {max_label} es >= número de clases {n_classes}")
        
        # Crear y configurar modelo
        self.model = MLP(
            input_size=3,
            hidden_sizes=hidden_sizes,
            output_size=n_classes,
            learning_rate=learning_rate
        )
        
        print(f"Arquitectura de la red: 3 -> {' -> '.join(map(str, hidden_sizes))} -> {n_classes}")
        print(f"Iniciando entrenamiento...")
        
        # Entrenar modelo
        losses = self.model.train(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=True)
        
        # Evaluar en conjunto de prueba
        train_accuracy = self.model.evaluate(X_train, y_train)
        test_accuracy = self.model.evaluate(X_test, y_test)
        
        self.is_trained = True
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'losses': losses,
            'n_classes': n_classes,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        print(f"\n{'='*50}")
        print(f"RESULTADOS DEL ENTRENAMIENTO:")
        print(f"{'='*50}")
        print(f"Accuracy en entrenamiento: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"Accuracy en prueba: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"{'='*50}")
        
        return results
    
    def predict_color(self, rgb: Tuple[int, int, int]) -> str:
        if not self.is_trained or self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        # Normalizar entrada
        rgb_normalized = np.array(rgb, dtype=np.float32).reshape(1, -1) / 255.0
        
        # Hacer predicción
        prediction = self.model.predict(rgb_normalized)[0]
        
        # Decodificar etiqueta
        color_label = self.label_encoder.inverse_transform([prediction])[0]
        
        return color_label
    
    def predict_cost(self, rgb: Tuple[int, int, int]) -> float:
        color_label = self.predict_color(rgb)
        return self.cost_mapping.get(color_label, 5.0)  # Costo por defecto si no se encuentra
    
    def get_average_rgb_from_tile(self, image, tile_row: int, tile_col: int, tile_size: int) -> Tuple[int, int, int]:
        pixels = image.load()
        width, height = image.size
        
        # Calcular límites del tile
        x_start = tile_col * tile_size
        y_start = tile_row * tile_size
        x_end = min(x_start + tile_size, width)
        y_end = min(y_start + tile_size, height)
        
        # Acumular valores RGB
        total_r, total_g, total_b = 0, 0, 0
        pixel_count = 0
        
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                r, g, b = pixels[x, y]
                total_r += r
                total_g += g
                total_b += b
                pixel_count += 1
        
        if pixel_count == 0:
            return (255, 255, 255)  # Blanco por defecto
        
        # Calcular promedio
        avg_r = int(total_r / pixel_count)
        avg_g = int(total_g / pixel_count)
        avg_b = int(total_b / pixel_count)
        
        return (avg_r, avg_g, avg_b)
    
    def save_model(self, filepath: str) -> None:
        if not self.is_trained or self.model is None:
            raise ValueError("No hay modelo entrenado para guardar")
        
        # Guardar red neuronal
        self.model.save(filepath)
        
        # Guardar label encoder
        import pickle
        encoder_path = filepath.replace('.pkl', '_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Modelo guardado en: {filepath}")
        print(f"Encoder guardado en: {encoder_path}")
    
    def load_model(self, filepath: str) -> None:
        # Cargar datos para configurar el encoder
        _, y = self.load_data()
        
        # Crear modelo
        n_classes = len(np.unique(y))
        self.model = MLP(input_size=3, hidden_sizes=[64, 32], output_size=n_classes)
        
        # Cargar pesos de la red neuronal
        self.model.load(filepath)
        
        # Cargar label encoder
        import pickle
        encoder_path = filepath.replace('.pkl', '_encoder.pkl')
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.is_trained = True
        print(f"Modelo cargado desde: {filepath}")
    
    def get_cost_mapping(self) -> Dict[str, float]:
        return self.cost_mapping.copy()
    
    def print_cost_mapping(self) -> None:
        print("\n" + "="*50)
        print("MAPEO DE MATERIALES A COSTOS")
        print("="*50)
        for material, cost in sorted(self.cost_mapping.items(), key=lambda x: x[1]):
            print(f"{material:12} -> Costo: {cost:3}")
        print("="*50)