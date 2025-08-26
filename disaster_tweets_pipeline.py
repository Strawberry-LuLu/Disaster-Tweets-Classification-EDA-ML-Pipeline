import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, exc
import nltk
import joblib
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Создаем папку для графиков
os.makedirs('plots', exist_ok=True)

# Загружаем stopwords и WordNet
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Загрузка stopwords и WordNet...")
    nltk.download('stopwords')
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

print("Все библиотеки успешно импортированы.")

# Шаг 1: Загрузка и первичный осмотр данных
print("\n--- ШАГ 1: Загрузка данных ---")

try:
    df = pd.read_csv('tweets.csv', quotechar='"', delimiter=',', encoding='utf-8')
    print("Данные загружены. Размер DataFrame:", df.shape)
    print("\nПервые 5 строк:")
    print(df.head())
    print("\nИнформация о DataFrame:")
    print(df.info())
    print(f"\nРаспределение целевой переменной:\n{df['target'].value_counts()}")
except FileNotFoundError:
    print("Ошибка: Файл 'tweets.csv' не найден!")
    exit()

# Шаг 2: Предобработка текста (очистка твитов)
print("\n--- ШАГ 2: Предобработка текста ---")

def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    text = re.sub(r'@\w+', '', text)
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]  # убираем очень короткие слова
    
    words = [lemmatizer.lemmatize(word) for word in words]
    
    cleaned_text = ' '.join(words)
    
    return cleaned_text

print("Старт очистки текста...")
df['cleaned_text'] = df['text'].apply(clean_text)
print("Очистка текста завершена!")

# Проверяем, что очистка прошла успешно
print(f"\nПримеры очищенных текстов (первые 3):")
for i in range(3):
    print(f"\n{i+1}. Оригинал: {df['text'].iloc[i][:100]}...")
    print(f"   Очищенный: {df['cleaned_text'].iloc[i][:100]}...")
    print("-" * 80)

# Шаг 3: Сохранение в базу данных (MySQL)
print("\n--- ШАГ 3: Загрузка в MySQL ---")
try:
    user = 'root'
    password = '1111'
    host = '127.0.0.1'
    database = 'disaster_tweets_db'

    connection_string = f"mysql+mysqlconnector://{user}:{password}@{host}/{database}"
    engine = create_engine(connection_string)
    
    # Проверяем подключение
    with engine.connect() as conn:
        print("Подключение к MySQL успешно!")
    
    print("Начинаю загрузку данных в MySQL...")
    df.to_sql('tweets', con=engine, if_exists='replace', index=False, chunksize=1000)
    print("Данные успешно загружены в MySQL!")
    
except exc.SQLAlchemyError as e:
    print(f"Ошибка подключения к MySQL: {e}")
    print("Продолжаем без сохранения в БД...")

# Шаг 4: Разведочный анализ данных (EDA)
print("\n--- ШАГ 4: Разведочный анализ (EDA) ---")
sns.set_style("whitegrid")

# 1. Распределение целевой переменной
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
df['target'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Распределение твитов по классам')
plt.xlabel('Класс (0 - Не катастрофа, 1 - Катастрофа)')
plt.ylabel('Количество твитов')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
df['target'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title('Процентное распределение')
plt.ylabel('')

plt.tight_layout()
plt.savefig('plots/target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("График распределения сохранен: plots/target_distribution.png")

# 2. Длина твитов до и после очистки
df['text_length'] = df['text'].apply(len)
df['cleaned_text_length'] = df['cleaned_text'].apply(len)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='text_length', hue='target', kde=True, bins=50, alpha=0.6)
plt.title('Длина исходного текста')
plt.xlabel('Длина текста')

plt.subplot(1, 2, 2)
sns.histplot(data=df, x='cleaned_text_length', hue='target', kde=True, bins=50, alpha=0.6)
plt.title('Длина очищенного текста')
plt.xlabel('Длина текста')

plt.tight_layout()
plt.savefig('plots/text_length_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("График длины текста сохранен: plots/text_length_comparison.png")

# 3. Средняя длина текста по классам
mean_lengths = df.groupby('target')['cleaned_text_length'].mean()
print(f"\nСредняя длина текста по классам:")
print(f"Не катастрофа (0): {mean_lengths[0]:.2f} символов")
print(f"Катастрофа (1): {mean_lengths[1]:.2f} символов")

# 4. Дополнительный график: распределение длины текста по классам (boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='target', y='cleaned_text_length', palette=['skyblue', 'salmon'])
plt.title('Распределение длины очищенного текста по классам')
plt.xlabel('Класс (0 - Не катастрофа, 1 - Катастрофа)')
plt.ylabel('Длина текста')
plt.xticks([0, 1], ['Не катастрофа', 'Катастрофа'])
plt.savefig('plots/text_length_by_class.png', dpi=300, bbox_inches='tight')
plt.close()
print("Boxplot длины текста сохранен: plots/text_length_by_class.png")

print("\nПостроение графиков завершено. Все графики сохранены в папке plots/")

# Шаг 5: Подготовка данных и обучение модели
print("\n--- ШАГ 5: Обучение модели Машинного обучения ---")

# Проверяем, что данные готовы
if df['cleaned_text'].isna().any():
    df['cleaned_text'] = df['cleaned_text'].fillna('')

print("Разделяем данные на признаки (X) и целевую переменную (y)...")
X = df['cleaned_text']
y = df['target']

# 80% данных - на обучение, 20% - на тестирование
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Размер обучающей выборки: {len(X_train)}")
print(f"Размер тестовой выборки: {len(X_test)}")

print("Преобразование текста в числа...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,  # игнорируем слова, которые встречаются реже 2 раз
    max_df=0.8  # игнорируем слова, которые встречаются в более чем 80% документов
)

print("Векторизуем тексты...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print(f"Размерность матрицы признаков: {X_train_tfidf.shape}")

print("Обучаем модель логистической регрессии...")
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',
    C=1.0  # параметр регуляризации
)
model.fit(X_train_tfidf, y_train)

print("Делаем предсказания на тестовых данных...")
y_pred = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)

print("\n" + "="*60)
print("ОЦЕНКА КАЧЕСТВА МОДЕЛИ:")
print("="*60)

accuracy = accuracy_score(y_test, y_pred)
print(f"Точность (Accuracy): {accuracy:.4f}")

print("\nПодробный отчет по классификации:")
print(classification_report(y_test, y_pred, target_names=['Not Disaster', 'Disaster']))

# Матрица ошибок
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Disaster', 'Disaster'],
            yticklabels=['Not Disaster', 'Disaster'])
plt.title('Матрица ошибок (Confusion Matrix)')
plt.ylabel('Истинный класс')
plt.xlabel('Предсказанный класс')
plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Матрица ошибок сохранена: plots/confusion_matrix.png")

# Примеры предсказаний
print("\n" + "="*60)
print("ПРИМЕРЫ ПРЕДСКАЗАНИЙ:")
print("="*60)

sample_size = min(5, len(X_test))
sample_indices = np.random.choice(len(X_test), sample_size, replace=False)

correct = 0
for i, idx in enumerate(sample_indices):
    true_class = y_test.iloc[idx]
    pred_class = y_pred[idx]
    probability = max(y_pred_proba[idx])
    
    status = "✓" if true_class == pred_class else "✗"
    if true_class == pred_class:
        correct += 1
    
    print(f"\nПример {i + 1} {status}:")
    print(f"Текст: {X_test.iloc[idx][:80]}...")
    print(f"Истинный класс: {true_class} ({'Disaster' if true_class == 1 else 'Not Disaster'})")
    print(f"Предсказанный класс: {pred_class} ({'Disaster' if pred_class == 1 else 'Not Disaster'})")
    print(f"Вероятность: {probability:.3f}")

print(f"\nТочность на примерах: {correct}/{sample_size}")

# Сохранение модели
print("\nСохраняем модель и векторизатор...")
try:
    joblib.dump(model, 'disaster_tweets_model.pkl')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    print("Модель и векторизатор успешно сохранены в файлы:")
    print("- disaster_tweets_model.pkl")
    print("- tfidf_vectorizer.pkl")
except Exception as e:
    print(f"Ошибка при сохранении: {e}")

# Дополнительно: график важности признаков (топ-20 слов)
print("\nСоздание графика важности признаков...")
feature_names = tfidf_vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

# Сортируем признаки по важности
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': coefficients
}).sort_values('importance', ascending=False)

# Топ-20 самых важных признаков для класса "Катастрофа"
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(20)
sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
plt.title('Топ-20 самых важных слов для предсказания катастрофы')
plt.xlabel('Важность (коэффициент модели)')
plt.ylabel('Слово')
plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("График важности признаков сохранен: plots/feature_importance.png")

print("\n" + "="*60)
print("ВСЕ ЭТАПЫ ПАЙПЛАЙНА УСПЕШНО ВЫПОЛНЕНЫ!")
print("Графики сохранены в папке plots/")
print("="*60)