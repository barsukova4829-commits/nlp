import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymorphy3
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ============================================================
# Шаг 1. Загрузка данных
# ============================================================
print("=" * 60)
print("Шаг 1. Загрузка данных")
print("=" * 60)

train_df = pd.read_json("train.jsonl", lines=True)
val_df = pd.read_json("validation.jsonl", lines=True)
test_df = pd.read_json("test.jsonl", lines=True)

print(f"Train:      {len(train_df)} записей")
print(f"Validation: {len(val_df)} записей")
print(f"Test:       {len(test_df)} записей")
print()

# ============================================================
# Шаг 2. Разведочный анализ данных (EDA)
# ============================================================
print("=" * 60)
print("Шаг 2. Разведочный анализ данных (EDA)")
print("=" * 60)

# 2.1 Распределение классов
print("\nРаспределение классов (train):")
print(train_df["label_text"].value_counts())

fig, ax = plt.subplots(figsize=(6, 4))
train_df["label_text"].value_counts().reindex(["Bad", "Neutral", "Good"]).plot(
    kind="bar", color=["#e74c3c", "#95a5a6", "#2ecc71"], ax=ax
)
ax.set_title("Распределение классов (train)")
ax.set_xlabel("Класс")
ax.set_ylabel("Количество")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=100)
plt.close()
print("  -> График сохранён: class_distribution.png")

# 2.2 Длина текстов
train_df["text_len"] = train_df["text"].str.len()
train_df["word_count"] = train_df["text"].str.split().str.len()

print(f"\nДлина текстов (символы): min={train_df['text_len'].min()}, "
      f"median={train_df['text_len'].median():.0f}, max={train_df['text_len'].max()}")
print(f"Количество слов:        min={train_df['word_count'].min()}, "
      f"median={train_df['word_count'].median():.0f}, max={train_df['word_count'].max()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for label_text, color in [("Bad", "#e74c3c"), ("Neutral", "#95a5a6"), ("Good", "#2ecc71")]:
    subset = train_df[train_df["label_text"] == label_text]
    axes[0].hist(subset["text_len"], bins=50, alpha=0.5, label=label_text, color=color)
    axes[1].hist(subset["word_count"], bins=50, alpha=0.5, label=label_text, color=color)
axes[0].set_title("Распределение длины текстов (символы)")
axes[0].legend()
axes[1].set_title("Распределение количества слов")
axes[1].legend()
plt.tight_layout()
plt.savefig("text_length_distribution.png", dpi=100)
plt.close()
print("  -> График сохранён: text_length_distribution.png")

# 2.3 Примеры текстов
print("\n--- Примеры текстов ---")
for label_text in ["Bad", "Neutral", "Good"]:
    sample = train_df[train_df["label_text"] == label_text].iloc[0]
    print(f"\n[{label_text}] (первые 200 символов):")
    print(f"  {sample['text'][:200]}...")

print()

# ============================================================
# Шаг 3. Предобработка текста
# ============================================================
print("=" * 60)
print("Шаг 3. Предобработка текста")
print("=" * 60)

morph = pymorphy3.MorphAnalyzer()
russian_stopwords = set(stopwords.words("russian"))


def preprocess(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)           # HTML-теги
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # URL
    text = re.sub(r"\S+@\S+", " ", text)            # email
    text = re.sub(r"[^а-яёa-z\s]", " ", text)       # всё кроме букв
    text = re.sub(r"\s+", " ", text).strip()         # лишние пробелы

    tokens = word_tokenize(text, language="russian")
    tokens = [morph.parse(t)[0].normal_form for t in tokens if t not in russian_stopwords and len(t) > 1]
    return " ".join(tokens)


print("Предобработка train...")
train_df["clean_text"] = train_df["text"].apply(preprocess)
print("Предобработка validation...")
val_df["clean_text"] = val_df["text"].apply(preprocess)
print("Предобработка test...")
test_df["clean_text"] = test_df["text"].apply(preprocess)

print(f"\nПример до:    {train_df['text'].iloc[0][:120]}...")
print(f"Пример после: {train_df['clean_text'].iloc[0][:120]}...")
print()

# ============================================================
# Шаг 4. Векторизация TF-IDF
# ============================================================
print("=" * 60)
print("Шаг 4. Векторизация TF-IDF")
print("=" * 60)

tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
)

X_train = tfidf.fit_transform(train_df["clean_text"])
X_val = tfidf.transform(val_df["clean_text"])
X_test = tfidf.transform(test_df["clean_text"])

y_train = train_df["label"].values
y_val = val_df["label"].values
y_test = test_df["label"].values

print(f"Размер словаря:       {len(tfidf.vocabulary_)}")
print(f"Размер матрицы train: {X_train.shape}")
print(f"Размер матрицы val:   {X_val.shape}")
print(f"Размер матрицы test:  {X_test.shape}")
print()

# ============================================================
# Шаг 5. Обучение моделей
# ============================================================
print("=" * 60)
print("Шаг 5. Обучение и оценка моделей на валидации")
print("=" * 60)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "LinearSVC": LinearSVC(max_iter=2000, random_state=42),
    "MultinomialNB": MultinomialNB(alpha=1.0),
    "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
}

results = {}
label_names = ["Bad", "Neutral", "Good"]

for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    results[name] = {"accuracy": acc, "model": model, "y_pred": y_pred}
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred, target_names=label_names))

# Сравнение моделей
print("\n--- Сравнение моделей (Accuracy на валидации) ---")
for name, res in sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
    print(f"  {name:25s} {res['accuracy']:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
names = list(results.keys())
accs = [results[n]["accuracy"] for n in names]
bars = ax.barh(names, accs, color=["#3498db", "#2ecc71", "#e67e22", "#9b59b6"])
ax.set_xlim(0, 1)
ax.set_xlabel("Accuracy")
ax.set_title("Сравнение моделей на валидации")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{acc:.4f}", va="center")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=100)
plt.close()
print("  -> График сохранён: model_comparison.png")

# ============================================================
# Шаг 6. Подбор гиперпараметров лучшей модели
# ============================================================
print("\n" + "=" * 60)
print("Шаг 6. Подбор гиперпараметров (LogisticRegression + TF-IDF)")
print("=" * 60)

best_overall_acc = 0
best_params = {}

for max_feat in [10000, 20000, 50000]:
    for ngram in [(1, 1), (1, 2), (1, 3)]:
        for C_val in [0.1, 1.0, 10.0]:
            tfidf_tune = TfidfVectorizer(
                max_features=max_feat, ngram_range=ngram, min_df=3, max_df=0.95
            )
            X_tr = tfidf_tune.fit_transform(train_df["clean_text"])
            X_v = tfidf_tune.transform(val_df["clean_text"])

            lr = LogisticRegression(max_iter=1000, C=C_val, random_state=42)
            lr.fit(X_tr, y_train)
            acc = accuracy_score(y_val, lr.predict(X_v))

            if acc > best_overall_acc:
                best_overall_acc = acc
                best_params = {"max_features": max_feat, "ngram_range": ngram, "C": C_val}
                best_tfidf = tfidf_tune
                best_model = lr

print(f"Лучшие параметры: {best_params}")
print(f"Лучший Accuracy на валидации: {best_overall_acc:.4f}")
print()

# ============================================================
# Шаг 7. Финальная оценка на тестовой выборке
# ============================================================
print("=" * 60)
print("Шаг 7. Финальная оценка на тестовой выборке")
print("=" * 60)

X_test_final = best_tfidf.transform(test_df["clean_text"])
y_test_pred = best_model.predict(X_test_final)

test_acc = accuracy_score(y_test, y_test_pred)
print(f"\nAccuracy на тесте: {test_acc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=label_names))

cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names, ax=ax)
ax.set_xlabel("Предсказано")
ax.set_ylabel("Истинное значение")
ax.set_title(f"Confusion Matrix (Test, Accuracy={test_acc:.4f})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
plt.close()
print("  -> График сохранён: confusion_matrix.png")

# ============================================================
# Шаг 8. Демонстрация работы
# ============================================================
print("\n" + "=" * 60)
print("Шаг 8. Демонстрация на примерах")
print("=" * 60)

LABELS = {0: "Bad", 1: "Neutral", 2: "Good"}


def predict_sentiment(text):
    processed = preprocess(text)
    vector = best_tfidf.transform([processed])
    prediction = best_model.predict(vector)[0]
    return LABELS[prediction]


examples = [
    "Отличный фильм! Прекрасная игра актёров, захватывающий сюжет. Рекомендую всем!",
    "Фильм так себе, ничего особенного. Можно посмотреть один раз, но пересматривать не буду.",
    "Ужасный фильм, потраченное время. Сюжет бессмысленный, актёры играют отвратительно.",
]

for text in examples:
    result = predict_sentiment(text)
    print(f"\nТекст: {text}")
    print(f"  -> Тональность: {result}")

print("\n" + "=" * 60)
print("Готово! Все графики сохранены в текущую директорию.")
print("=" * 60)
