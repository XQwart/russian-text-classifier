import os
import re
from typing import Dict, List, Tuple, Any

import nltk
import numpy as np
import pandas as pd
import pymorphy2
import torch
import nlpaug.augmenter.word as naw
from datasets import Dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

# Загружаем необходимые ресурсы NLTK
print("Загрузка ресурсов NLTK...")
nltk.download('stopwords')
nltk.download('punkt')

# Определяем устройство для обучения
print("Определение устройства для обучения...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Используется устройство:', device)

# Инициализируем необходимые инструменты
print("Инициализация инструментов...")
russian_stopwords = stopwords.words('russian')
morph = pymorphy2.MorphAnalyzer()

def preprocess_text(text: str) -> str:
    """
    Предобработка текста: приведение к нижнему регистру, удаление спецсимволов,
    токенизация, удаление стоп-слов и лемматизация.
    
    Args:
        text: Исходный текст для обработки
        
    Returns:
        str: Обработанный текст
    """
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+|mailto:\S+', '', text)
    text = re.sub(r'[^а-яё\s]', '', text)
    tokens = word_tokenize(text, language='russian')
    tokens = [
        morph.normal_forms(token)[0] 
        for token in tokens 
        if token not in russian_stopwords and token.isalpha()
    ]
    return ' '.join(tokens)

def augment_text(text: str) -> str:
    """
    Аугментация текста с помощью контекстуальной замены слов.
    
    Args:
        text: Исходный текст
        
    Returns:
        str: Аугментированный текст
    """
    augmented = augmenter.augment(text)
    return augmented[0] if isinstance(augmented, list) else augmented

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Вычисление метрик качества модели.
    
    Args:
        eval_pred: Кортеж из предсказаний модели и истинных меток
        
    Returns:
        Dict[str, float]: Словарь с метриками качества
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, target_names=label_list, output_dict=True)
    return {
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
    }

# Загружаем и подготавливаем данные
print("Загрузка данных...")
dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset.csv"))
dataset["cleaned_text"] = dataset["Текст"].apply(preprocess_text)

# Создаем аугментатор
print("Создание аугментатора...")
augmenter = naw.ContextualWordEmbsAug(
    model_path="DeepPavlov/rubert-base-cased",
    action="substitute",
    device=str(device),
    aug_p=0.1,
    aug_max=5
)

# Разделяем данные на обучающую и тестовую выборки
print("Разделение данных на обучающую и тестовую выборки...")
train_data, test_data = train_test_split(
    dataset, 
    test_size=0.2, 
    random_state=42,
    stratify=dataset['Категория']
)

# Аугментация обучающей выборки
print("Аугментация обучающей выборки...")
data_augmented = train_data.copy()
data_augmented['cleaned_text'] = data_augmented['cleaned_text'].apply(augment_text)
train_data = pd.concat([train_data, data_augmented]).reset_index(drop=True)

# Подготовка меток
print("Подготовка меток...")
label_list = dataset['Категория'].unique().tolist()
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

# Преобразуем метки в числовой формат
train_data['labels'] = train_data['Категория'].map(label_to_id)
test_data['labels'] = test_data['Категория'].map(label_to_id)

# Создание датасетов
print("Создание датасетов...")
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Инициализация токенизатора и модели
print("Инициализация токенизатора и модели...")
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "DeepPavlov/rubert-base-cased", 
    num_labels=len(label_list)
).to(device)

# Токенизация данных
def tokenize_function(examples: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Токенизация текстов с помощью BERT токенизатора.
    
    Args:
        examples: Словарь с примерами для токенизации
        
    Returns:
        Dict[str, torch.Tensor]: Токенизированные тексты
    """
    return tokenizer(
        examples["cleaned_text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

print("Токенизация данных...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Подготовка данных для обучения
print("Подготовка данных для обучения...")
train_dataset = train_dataset.remove_columns(['cleaned_text', 'Категория', 'Текст'])
test_dataset = test_dataset.remove_columns(['cleaned_text', 'Категория', 'Текст'])
train_dataset.set_format('torch')
test_dataset.set_format('torch')

# Настройка параметров обучения
print("Настройка параметров обучения...")
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy='epoch',
    save_strategy='no',
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=False,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    gradient_accumulation_steps=2
)

# Создание и обучение модели
print("Создание и обучение модели...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("Обучение модели...")
trainer.train()

# Оценка результатов
print("Оценка модели...")
eval_results = trainer.evaluate()
print("\nРезультаты оценки:")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")

# Сохранение модели и токенизатора
print("Сохранение модели...")
try:
    os.makedirs("./best_model", exist_ok=True)
    model.save_pretrained("./best_model", safe_serialization=True)
    tokenizer.save_pretrained("./best_model")
    print("\nМодель и токенизатор успешно сохранены в директории './best_model'")
except Exception as e:
    print(f"\nОшибка при сохранении модели: {str(e)}")
