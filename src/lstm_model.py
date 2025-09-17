import torch
import torch.nn as nn

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.2):
        super(TextLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Слои модели
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x) # (batch_size, seq_length, embedding_dim)
        
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)
        
        lstm_out = self.dropout(lstm_out)
        # Выравниваем вывод для полносвязного слоя
        out = self.fc(lstm_out) # (batch_size, seq_length, vocab_size)
        return out, hidden

    def generate(self, input_sequence, max_length=20, temperature=1.0):
        """Генерирует продолжение последовательности"""
        self.eval()
        self.to(self.device)

        # ✅ ПРОВЕРКА: входная последовательность не пустая
        if not input_sequence:
            return input_sequence.copy()

        generated = input_sequence.copy() # Копируем вход
        hidden = None

        with torch.no_grad():
            for _ in range(max_length):
                try:
                    # ✅ ПРОВЕРКА: generated не пустой
                    if not generated:
                        break
                    # Подготовка последнего слова к подаче в модель
                    input_tensor = torch.tensor([generated[-1]], device=self.device).unsqueeze(0) # (1, 1)
                    # Предикт
                    output, hidden = self.forward(input_tensor, hidden)
                    # output shape: (1, 1, vocab_size)
                    output = output[:, -1, :] / temperature
                    # Применяем softmax для получения вероятностей
                    probs = torch.softmax(output, dim=-1)
                    # Берем наиболее вероятный следующий индекс
                    next_token_id = torch.argmax(probs, dim=-1).item()
                    generated.append(next_token_id)
                    # Остановка, если модель выдала <PAD> или <UNK> (логику можно улучшить)
                    if next_token_id == 0: 
                        break
                except Exception as e:
                    print(f"Ошибка во время генерации: {e}")
                    break
        return generated