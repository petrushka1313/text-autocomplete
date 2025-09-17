import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.next_token_dataset import NextTokenDataset
from src.lstm_model import TextLSTM
from src.eval_lstm import calculate_rouge_lstm  # Функцию для оценки напишем отдельно
import pandas as pd
import json
from tqdm import tqdm

def train_model():
    try:
        # Hyperparameters
        BATCH_SIZE = 64 # Уменьшили с 256
        EMBEDDING_DIM = 128  # Уменьшили с 128
        HIDDEN_DIM = 256  # Уменьшили с 256
        NUM_LAYERS = 2
        SEQ_LENGTH = 30
        LEARNING_RATE = 0.001
        NUM_EPOCHS = 5  # Уменьшили с 10

        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load data and vocabulary
        print("Loading data...")
        train_df = pd.read_csv('data/train.csv')
        val_df = pd.read_csv('data/val.csv')

        with open('data/vocab.json', 'r', encoding='utf-8') as f:
            word_to_idx = json.load(f)
        vocab_size = len(word_to_idx)

        # Create Datasets and DataLoaders
        train_dataset = NextTokenDataset(train_df, word_to_idx, seq_length=SEQ_LENGTH)
        val_dataset = NextTokenDataset(val_df, word_to_idx, seq_length=SEQ_LENGTH)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(f"Размер батча: {BATCH_SIZE}")
        print(f"Количество батчей в тренировочной выборке: {len(train_loader)}")
        print(f"Количество батчей в валидационной выборке: {len(val_loader)}")

        # Посмотрим на один батч
        sample_batch = next(iter(train_loader))
        print(f"\nФорма входных данных: {sample_batch[0].shape}")
        print(f"Форма целевых данных: {sample_batch[1].shape}")

        # Initialize model, loss, optimizer
        model = TextLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<PAD>'])
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(NUM_EPOCHS):
            model.train()
            model.to(device)
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')

            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                output, _ = model(inputs)
                # output: (batch, seq, vocab), targets: (batch, seq)
                loss = criterion(output.reshape(-1, vocab_size), targets.reshape(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f"Epoch {epoch+1} | Average Train Loss: {avg_train_loss:.4f}")

            # Validation Loss and ROUGE
            model.eval()
            model.to(device)
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    output, _ = model(inputs)
                    loss = criterion(output.reshape(-1, vocab_size), targets.reshape(-1))
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch+1} | Validation Loss: {avg_val_loss:.4f}")

            # Calculate ROUGE every few epochs or at the end
            if (epoch + 1) % 2 == 0:
                rouge_scores = calculate_rouge_lstm(model, val_dataset, word_to_idx)
                print(f"ROUGE Scores: {rouge_scores}")

        # Save the trained model
        torch.save(model.state_dict(), 'models/lstm_model.pth')
        print("Training finished. Model saved.")

        return model, train_losses, val_losses

    except Exception as e:
        print(f"Ошибка в train_model: {e}")
        import traceback
        traceback.print_exc()
        return None, [], []  # Возвращаем пустые значения вместо None