import torch
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
from src.data_utils import simple_tokenize  # Импортируем нашу функцию токенизации

class NextTokenDataset(Dataset):
    def __init__(self, dataframe, word_to_idx, seq_length=20):
        self.data = dataframe
        self.word_to_idx = word_to_idx
        self.seq_length = seq_length
        self.unk_idx = word_to_idx.get('<UNK>', 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Берем список токенов для данного примера
        tokens = self.data.iloc[idx]['tokens']
        if isinstance(tokens, str):
            tokens = eval(tokens)  # Если tokens сохранены как строка, преобразуем обратно в список

        # Конвертируем слова в индексы
        indices = [self.word_to_idx.get(token, self.unk_idx) for token in tokens]

        # Разделяем на входную последовательность и target (сдвинут на 1)
        # Обрезаем последовательность, если она слишком длинная
        if len(indices) - 1 > self.seq_length:
            input_seq = indices[:self.seq_length]
            target_seq = indices[1:self.seq_length+1]
        else:
            input_seq = indices[:-1]
            target_seq = indices[1:]
            # Дополняем последовательности до seq_length
            pad_idx = self.word_to_idx['<PAD>']
            input_seq.extend([pad_idx] * (self.seq_length - len(input_seq)))
            target_seq.extend([pad_idx] * (self.seq_length - len(target_seq)))

        return torch.tensor(input_seq), torch.tensor(target_seq)