flowchart TD
    A[Исходные данные<br>data/tweets.txt] --> B[src/data_utils.py<br>Очистка, токенизация, split]
    B -- Создает --> C[Словарь<br>data/vocab.json]
    B -- Создает --> D[Наборы данных<br>data/train/.val/.test.csv]
    
    D & C --> E[src/next_token_dataset.py<br>Обертка в torch.Dataset]
    E --> F[src/lstm_train.py<br>Обучение модели]
    F -- Сохраняет --> G[Веса модели<br>models/lstm_model.pth]
    
    D --> H[src/eval_lstm.py<br>Оценка LSTM]
    G --> H
    
    A --> I[src/eval_transformer_pipeline.py<br>Использует чистый текст]
    I --> J[Предобученная модель<br>distilgpt2]
    
    H -- Метрики --> K[Сравнение и выводы<br>solution.ipynb]
    J -- Метрики --> K