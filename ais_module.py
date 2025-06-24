import numpy as np
import random
import pytesseract
import cv2
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from scipy.spatial.distance import cosine
import logging
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import collections
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# 1. Путь к Tesseract (укажите при необходимости)
# ------------------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# ------------------------------------------------------------------------------
# 2. Вспомогательные функции
# ------------------------------------------------------------------------------
def cosine_similarity(vec1, vec2):
    """Улучшенная функция косинусного сходства с обработкой нулевых векторов"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return 1 - cosine(vec1, vec2)  # Используем scipy для большей точности

def mutate_vector(center, mutation_rate, temperature=1.0):
    """Улучшенная функция мутации с температурой"""
    mutated = center.copy()
    size = mutated.shape[0]
    num_mutations = int(np.ceil(size * mutation_rate))
    mutation_positions = np.random.choice(size, size=num_mutations, replace=False)
    
    for pos in mutation_positions:
        # Адаптивная мутация в зависимости от температуры
        delta = np.random.normal(loc=0.0, scale=0.1 * temperature)
        mutated[pos] += delta
        
        # Ограничение значений
        mutated[pos] = np.clip(mutated[pos], -1.0, 1.0)
    
    return mutated

# ------------------------------------------------------------------------------
# 3. AIS Классификатор
# ------------------------------------------------------------------------------
class AISClassifier:
    def __init__(self, affinity_threshold=0.7, clone_factor=3, mutation_rate=0.01, 
                 epochs=15, max_detectors=4000, temperature=1.0, memory_cells=True,
                 batch_size=100):
        self.affinity_threshold = affinity_threshold
        self.clone_factor = clone_factor
        self.mutation_rate = mutation_rate
        self.epochs = epochs
        self.max_detectors = max_detectors
        self.temperature = temperature
        self.memory_cells = memory_cells
        self.detectors = []
        self.memory_cells_list = []
        self.best_accuracy = 0.0
        self.training_history = []
        self.best_detectors = []
        self.learning_rate = 0.1
        self.min_affinity = 0.5
        self.class_names = None
        self.vectorizer = None
        self.batch_size = batch_size
        self.validation_split = 0.2
        self.stemmer = SnowballStemmer('english')

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Zа-яА-Я\s]', ' ', text)
        text = ' '.join([self.stemmer.stem(w) for w in text.split()])
        text = ' '.join(text.split())
        return text

    def fit(self, X, y, class_names=None):
        if hasattr(X, "toarray"):
            X = X.toarray()
        self.class_names = class_names
        logger.info(f"Начало обучения AIS с параметрами: threshold={self.affinity_threshold}, "
                   f"clone_factor={self.clone_factor}, mutation_rate={self.mutation_rate}")
        indices = np.random.permutation(X.shape[0])
        split_idx = int(X.shape[0] * (1 - self.validation_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
        best_val_accuracy = 0.0
        stagnation_counter = 0
        patience = 3
        unique_classes = np.unique(y_train)
        for class_label in unique_classes:
            class_samples = X_train[y_train == class_label]
            if len(class_samples) > 0:
                for _ in range(5):  # Больше начальных детекторов
                    sample_idx = np.random.randint(0, len(class_samples))
                    initial_detector = {
                        "center": class_samples[sample_idx].copy(),
                        "label": class_label,
                        "affinity": 1.0,
                        "age": 0,
                        "success_rate": 1.0,
                        "confidence": 1.0
                    }
                    self.detectors.append(initial_detector)
        for epoch in range(self.epochs):
            indices = np.random.permutation(len(X_train))
            epoch_detectors = []
            for batch_start in range(0, len(indices), self.batch_size):
                batch_indices = indices[batch_start:batch_start + self.batch_size]
                batch_vectors = X_train[batch_indices]
                batch_labels = y_train[batch_indices]
                for idx, (sample_vector, sample_label) in enumerate(zip(batch_vectors, batch_labels)):
                    best_affinity = -1
                    best_detector_idx = -1
                    for i, detector in enumerate(self.detectors):
                        if detector["label"] == sample_label:
                            aff = cosine_similarity(sample_vector, detector["center"])
                            if aff > best_affinity:
                                best_affinity = aff
                                best_detector_idx = i
                    if best_affinity < self.affinity_threshold:
                        noise = np.random.normal(0, 0.1, sample_vector.shape)
                        new_center = sample_vector + noise
                        new_center = np.clip(new_center, -1, 1)
                        new_detector = {
                            "center": new_center,
                            "label": sample_label,
                            "affinity": 1.0,
                            "age": 0,
                            "success_rate": 1.0,
                            "confidence": 1.0
                        }
                        epoch_detectors.append(new_detector)
                    else:
                        best_detector = self.detectors[best_detector_idx]
                        num_clones = int(self.clone_factor * (1 + best_affinity))
                        for _ in range(num_clones):
                            mutation_rate = self.mutation_rate * (1 - best_affinity)
                            new_center = mutate_vector(
                                best_detector["center"],
                                mutation_rate,
                                self.temperature
                            )
                            new_affinity = cosine_similarity(sample_vector, new_center)
                            if new_affinity > self.min_affinity:
                                new_detector = {
                                    "center": new_center,
                                    "label": best_detector["label"],
                                    "affinity": new_affinity,
                                    "age": 0,
                                    "success_rate": best_detector.get("success_rate", 1.0),
                                    "confidence": new_affinity
                                }
                                epoch_detectors.append(new_detector)
            self.detectors.extend(epoch_detectors)
            if self.max_detectors is not None and len(self.detectors) > self.max_detectors:
                self.detectors.sort(key=lambda x: (
                    x.get("success_rate", 0.0) * x.get("confidence", 0.0),
                    -x.get("age", 0)
                ))
                self.detectors = self.detectors[-self.max_detectors:]
            for detector in self.detectors:
                detector["age"] += 1
                detector["success_rate"] *= 0.99
            val_predictions, _ = self.predict(X_val)
            val_accuracy = np.mean(val_predictions == y_val)
            self.training_history.append(val_accuracy)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.best_detectors = self.detectors.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            self._adapt_parameters(epoch, val_accuracy)
            if stagnation_counter >= patience:
                logger.info("Обнаружен застой в обучении, применяем дополнительные меры")
                self._handle_stagnation()
                if stagnation_counter >= patience * 2:
                    logger.info("Прекращаем обучение из-за длительного застоя")
                    break
            logger.info(f"Эпоха {epoch + 1} завершена.")
        self.detectors = self.best_detectors

    def predict(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        predictions = []
        confidences = []
        low_affinity_count = 0
        for i in range(X.shape[0]):
            sample_vector = X[i]
            class_affinities = collections.defaultdict(list)
            for detector in self.detectors:
                aff = cosine_similarity(sample_vector, detector["center"])
                class_affinities[detector["label"]].append(aff)
            mean_affinities = {label: np.mean(affs) for label, affs in class_affinities.items()}
            if mean_affinities:
                best_label = max(mean_affinities, key=mean_affinities.get)
                best_conf = mean_affinities[best_label]
                if best_conf < 0.2:
                    low_affinity_count += 1
                predictions.append(best_label)
                confidences.append(best_conf)
            else:
                predictions.append(0)
                confidences.append(0.0)
        logger.info(f"Количество низких affinity (<0.2): {low_affinity_count} из {X.shape[0]}")
        return np.array(predictions), np.array(confidences)

    def predict_text(self, text):
        """Метод для классификации текста с возвратом уверенности"""
        if self.vectorizer is None:
            raise ValueError("Векторизатор не инициализирован. Сначала обучите модель.")

        # Предобработка текста
        processed_text = self.preprocess_text(text)
        
        # Векторизация текста
        text_vector = self.vectorizer.transform([processed_text])
        
        # Получение предсказания и уверенности
        prediction, confidence = self.predict(text_vector)
        
        # Получение имени класса
        class_name = self.class_names[prediction[0]] if self.class_names is not None else str(prediction[0])
        
        return {
            'class': class_name,
            'confidence': float(confidence[0]),
            'raw_prediction': int(prediction[0])
        }

    def _adapt_parameters(self, epoch, current_accuracy):
        """Улучшенная адаптация параметров"""
        # Адаптивная температура
        self.temperature = max(0.5, self.temperature * 0.95)
        
        # Адаптивный порог аффинити
        if epoch > 0 and len(self.training_history) > 1:
            prev_accuracy = self.training_history[-2]
            if current_accuracy < prev_accuracy:
                self.affinity_threshold *= 0.98
            elif current_accuracy > prev_accuracy:
                self.affinity_threshold *= 1.02

        # Адаптивный фактор клонирования
        if current_accuracy < 0.5:
            self.clone_factor = min(3, self.clone_factor * 1.05)
        elif current_accuracy > 0.8:
            self.clone_factor = max(2, self.clone_factor * 0.95)

        # Адаптивная скорость мутации
        self.mutation_rate = max(0.005, min(0.05, self.mutation_rate * (1 + (current_accuracy - 0.5))))

    def _handle_stagnation(self):
        """Обработка застоя в обучении"""
        self.mutation_rate *= 1.5
        self.temperature = min(1.0, self.temperature * 1.5)
        
        if len(self.detectors) > self.max_detectors // 2:
            self.detectors.sort(key=lambda x: x.get("success_rate", 0.0))
            self.detectors = self.detectors[-self.max_detectors//2:]

    def save(self, path):
        """Сохранение модели с дополнительными параметрами"""
        model_data = {
            'detectors': self.detectors,
            'memory_cells': self.memory_cells_list,
            'parameters': {
                'affinity_threshold': self.affinity_threshold,
                'clone_factor': self.clone_factor,
                'mutation_rate': self.mutation_rate,
                'temperature': self.temperature
            },
            'class_names': self.class_names,
            'vectorizer': self.vectorizer
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Модель сохранена в {path}")

    def load(self, path):
        """Загрузка модели с дополнительными параметрами"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.detectors = model_data['detectors']
        self.memory_cells_list = model_data['memory_cells']
        params = model_data['parameters']
        self.affinity_threshold = params['affinity_threshold']
        self.clone_factor = params['clone_factor']
        self.mutation_rate = params['mutation_rate']
        self.temperature = params['temperature']
        self.class_names = model_data.get('class_names')
        self.vectorizer = model_data.get('vectorizer')
        logger.info(f"Модель загружена из {path}")

# ------------------------------------------------------------------------------
# 4. Загрузка данных и подготовка
# ------------------------------------------------------------------------------
print("Загружаем датасет '20 Newsgroups' (5 классов)...")
selected_categories = ['sci.space', 'rec.sport.baseball', 'talk.politics.mideast', 'comp.graphics', 'soc.religion.christian']
newsgroups_data = fetch_20newsgroups(subset='all', categories=selected_categories, remove=('headers', 'footers', 'quotes'))
X_texts = newsgroups_data.data
y_labels = newsgroups_data.target
class_names = newsgroups_data.target_names

from sklearn.decomposition import TruncatedSVD
vectorizer = TfidfVectorizer(max_features=3000)
X_vectors = vectorizer.fit_transform(X_texts)
svd = TruncatedSVD(n_components=300, random_state=42)
X_vectors = svd.fit_transform(X_vectors)

max_samples = 4000
X_vectors = X_vectors[:max_samples]
y_labels = y_labels[:max_samples]

X_train, X_test, y_train, y_test = train_test_split(X_vectors, y_labels, test_size=0.2, random_state=42, shuffle=True)

ais = AISClassifier(
    affinity_threshold=0.65,
    clone_factor=6,
    mutation_rate=0.015,
    epochs=50,
    max_detectors=4000,
    temperature=1.0,
    memory_cells=True,
    batch_size=100
)

# Функция для вывода confusion matrix после обучения
def show_confusion_matrix():
    y_pred, _ = ais.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title('Confusion Matrix (AIS)')
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
# 5. OCR Функция
# ------------------------------------------------------------------------------
def recognize_text_from_image(image_path):
    """Распознавание текста с сохранением строк и пустых строк между абзацами по координатам pytesseract"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return "", 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((1, 1), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        from pytesseract import Output
        data = pytesseract.image_to_data(gray, lang='eng', config=custom_config, output_type=Output.DICT)
        lines = []
        last_bottom = None
        last_line_num = None
        last_block_num = None
        current_line_words = []
        current_line_top = None
        current_line_height = None
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if not text:
                continue
            line_num = data['line_num'][i]
            block_num = data['block_num'][i]
            top = data['top'][i]
            height = data['height'][i]
            bottom = top + height
            if (last_line_num is not None and (line_num != last_line_num or block_num != last_block_num)):
                # Завершили строку, анализируем зазор
                if last_bottom is not None and (current_line_top - last_bottom) > current_line_height * 1.2:
                    lines.append("")  # пустая строка между абзацами
                lines.append(" ".join(current_line_words))
                last_bottom = current_line_top + current_line_height
                current_line_words = []
            if not current_line_words:
                current_line_top = top
                current_line_height = height
            current_line_words.append(text)
            last_line_num = line_num
            last_block_num = block_num
        # Добавить последнюю строку
        if current_line_words:
            if last_bottom is not None and (current_line_top - last_bottom) > current_line_height * 1.2:
                lines.append("")
            lines.append(" ".join(current_line_words))
        text = "\n".join(lines)
        text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E]', '', text)
        text = text.rstrip()
        confs = []
        for c in data['conf']:
            try:
                val = float(c)
                if val >= 0:
                    confs.append(val)
            except Exception:
                continue
        confidence = float(np.mean(confs)) / 100 if confs else 0.0
        return text, confidence
    except Exception as e:
        logger.error(f"Ошибка при распознавании текста: {e}")
        return "", 0.0

