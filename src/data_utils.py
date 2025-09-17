import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

def download_and_load_data(path='data/tweets.txt'):
    """Загружает исходный текстовый файл в DataFrame"""
    if os.path.exists(path):
        # Предполагается, что каждая строка в файле - отдельный твит
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Очищаем строки от лишних пробелов и переносов
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        df = pd.DataFrame(cleaned_lines, columns=['text'])
        print(f"Успешно загружено {len(df)} строк")
        return df
    return None

def clean_text(text):
    """Очищает и нормализует текст"""
    if not isinstance(text, str):
        return ""
    # Привести к нижнему регистру
    text = text.lower()
    # Удалить ссылки
    text = re.sub(r'http\S+', '', text)
    # Удалить упоминания (@username) и хештеги
    text = re.sub(r'[@#]\w+', '', text)
    # Удалить всё, кроме букв, цифр и основных знаков препинания
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9.!?,;:']", ' ', text)
    # Заменить множественные пробелы на один
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def simple_tokenize(text):
    """Простая токенизация по пробелам и знакам препинания"""
    # Это упрощенный пример. Для трансформера мы будем использовать его собственный токенизатор.
    # Разделяем слова, но оставляем знаки препинания отдельно
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens

def prepare_datasets(sample_size=200000):
    """Основная функция подготовки данных"""
    print("Загрузка данных...")
    df = download_and_load_data()

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size)
        print(f"Взята случайная выборка из {sample_size} примеров")

    print("Очистка текста...")
    df['text_clean'] = df['text'].apply(clean_text)

    # Удаляем пустые строки после очистки
    df = df[df['text_clean'].str.len() > 0]

    print("Токенизация...")
    # Для LSTM нам нужен список токенов
    df['tokens'] = df['text_clean'].apply(simple_tokenize)

    # Для последующего использования с трансформером сохраним чистый текст
    df['text_clean'].to_csv('data/dataset_processed.csv', index=False, header=False)

    # Разделяем данные: 80% train, 20% временный
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    # Делим временную выборку пополам: 10% val, 10% test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Сохраняем разделенные данные (тokens для LSTM)
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    print(f"Данные подготовлены. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Построение словаря на ТРЕНИРОВОЧНЫХ данных
    print("Построение словаря...")
    vocab = set()

    # Считаем частоту слов
    word_counts = Counter()
    for token_list in train_df['tokens']:
        word_counts.update(token_list)

    # Берем только N самых частых слов
    VOCAB_LIMIT = 10000  # Ограничиваем размер словаря
    vocab = {word for word, count in word_counts.most_common(VOCAB_LIMIT)}

    # Добавляем специальные токены
    vocab.add('<PAD>')  # Для дополнения последовательностей
    vocab.add('<UNK>')  # Для неизвестных слов

    # Преобразуем множество в словарь с индексами
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    # Сохраняем словарь для использования при обучении
    import json
    with open('data/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(word_to_idx, f, ensure_ascii=False)

    print(f"Размер словаря: {len(word_to_idx)}")
    return word_to_idx, idx_to_word

if __name__ == '__main__':
    prepare_datasets()