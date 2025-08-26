# Disaster-Tweets-Classification-EDA-ML-Pipeline / Классификация твитов о катастрофах
Проект по автоматическому определению, сообщает ли твит о реальной катастрофе или просто использует соответствующую лексику в переносном смысле. Полный цикл работы с данными: от очистки и анализа до построения и оценки ML-модели.

##  О проекте

Данный проект демонстрирует полный пайплайн работы с данными для задачи бинарной классификации текстов:
- Загрузка и предобработка данных
- Анализ и визуализация (EDA)
- Сохранение в базу данных (MySQL)
- Векторизация текста и обучение модели
- Оценка качества и сохранение результатов

##  Набор данных

Используется датасет [Disaster Tweets](https://www.kaggle.com/datasets/vstepanenko/disaster-tweets?resource=download) с Kaggle:
- **Объем:** 11 370 твитов
- **Признаки:** id, keyword, location, text, target
- **Целевая переменная:** 
  - `0` - не катастрофа (9 256 примеров)
  - `1` - катастрофа (2 114 примеров)
- **Дисбаланс классов:** 81.4% vs 18.6%

##  Технологический стек

- **Язык программирования:** Python 3.13.7
- **Обработка данных:** Pandas, NumPy
- **Визуализация:** Matplotlib, Seaborn
- **База данных:** MySQL, SQLAlchemy
- **Обработка естественного языка:** NLTK (stopwords, lemmatization)
- **Машинное обучение:** Scikit-learn (TF-IDF, Logistic Regression)
- **Сериализация:** Joblib

##  Основные этапы выполнения

### 1. Загрузка и анализ данных
- Загружено 11 370 твитов
- Проведен первичный анализ структуры данных
- Выявлен дисбаланс классов: 81.4% (не катастрофа) vs 18.6% (катастрофа)

### 2. Предобработка текста
- Приведение к нижнему регистру
- Удаление упоминаний пользователей (@username)
- Удаление URL-ссылок
- Удаление специальных символов и цифр
- Удаление стоп-слов и лемматизация
- Пример очистки: 
  - **Было:** `"Arsonist sets cars ablaze at dealership https://t.co/gOQvyJbpVI"`
  - **Стало:** `"arsonist set car ablaze dealership"`

### 3. Работа с базой данных
- Попытка сохранения в MySQL (требует предварительного создания БД)
- Использование SQLAlchemy для ORM

### 4. Разведочный анализ (EDA)
- Визуализация распределения классов
- Анализ длины текстов:
  - Средняя длина твитов о катастрофах: **70.74 символов**
  - Средняя длина обычных твитов: **61.90 символов**

### 5. Машинное обучение
- **Векторизация:** TF-IDF с 5000 features и биграммами
- **Модель:** Логистическая регрессия с учетом дисбаланса классов
- **Разделение данных:** 80% train / 20% test
- **Размерность данных:** 9096×5000

##  Результаты модели

### Метрики качества:
- **Accuracy:** 0.8470
- **Precision (Disaster):** 0.57
- **Recall (Disaster):** 0.75
- **F1-score (Disaster):** 0.64

### Матрица ошибок:
```python
              Not Disaster    Disaster
Not Disaster      1607          244
Disaster          105           318
```

##  Как запустить проект

1. **Установить зависимости:**
git clone [https://github.com/ВАШ_ЛОГИН/disaster-tweets-classification.git](https://github.com/Strawberry-LuLu/Disaster-Tweets-Classification-EDA-ML-Pipeline.git)
cd disaster-tweets-classification

2. **Создайте виртуальное окружение:**
python -m venv venv

3. **Активируйте venv:**
# Для Windows:
.\venv\Scripts\activate

# Для Linux/Mac:
source venv/bin/activate

4. **Установите зависимости:**
pip install -r requirements.txt

5. **Установить NLTK данные:**
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

6. **Запустить основной скрипт:**
python disaster_tweets_pipeline.py

## Структура проекта

- **disaster-tweets-classification/** - корневая папка проекта
  - **disaster_tweets_pipeline.py** - основной скрипт пайплайна
  - **tweets.csv** - исходные данные
  - **disaster_tweets_model.pkl** - обученная модель
  - **tfidf_vectorizer.pkl** - векторизатор текста
  - **plots/** - папка с графиками
    - target_distribution.png
    - text_length_comparison.png
    - text_length_by_class.png
    - confusion_matrix.png
    - feature_importance.png
  - **README.md** - документация

## Возможности улучшения

- Балансировка классов (SMOTE, undersampling)
- Эксперименты с другими моделями (Random Forest, SVM, BERT)
- Настройка гиперпараметров
- Улучшение предобработки текста
- Создание веб-интерфейса для предсказаний

## Статус проекта: Завершен
