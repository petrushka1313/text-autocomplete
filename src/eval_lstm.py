from rouge_score import rouge_scorer
import torch

def calculate_rouge_lstm(model, dataset, word_to_idx, num_samples=100, max_gen_length=15):
    """
    Вычисляет ROUGE-1 и ROUGE-2 для LSTM модели.
    Берет начало последовательности (3/4) и генерирует продолжение (1/4).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []

    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    for i in range(min(num_samples, len(dataset))):
        # Берем пример из датасета
        input_seq, target_seq = dataset[i]

        input_seq = input_seq.to(device)
        # input_seq уже обрезана/дополнена до SEQ_LENGTH. Нас интересует только не-pad часть.
        actual_tokens = [idx_to_word.get(idx, '<UNK>') for idx in input_seq.tolist() if idx != word_to_idx['<PAD>']]

        # Берем первые 3/4 как префикс
        split_point = int(len(actual_tokens) * 0.75)
        prefix_tokens = actual_tokens[:split_point]
        reference_continuation = actual_tokens[split_point:] # Идеальное продолжение

        # Генерируем продолжение с помощью модели
        # Конвертируем префикс в индексы
        prefix_indices = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in prefix_tokens]
        # Генерируем, начиная с последнего токена префикса
        generated_indices = model.generate(prefix_indices, max_length=max_gen_length)
        # Конвертируем сгенерированные индексы обратно в слова
        generated_continuation = [idx_to_word.get(idx, '<UNK>') for idx in generated_indices[len(prefix_indices):]]

        # Преобразуем списки токенов в строки для расчета ROUGE
        reference_text = ' '.join(reference_continuation)
        generated_text = ' '.join(generated_continuation)

        scores = scorer.score(reference_text, generated_text)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)

    # Усредняем результаты
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)

    return {'rouge1': avg_rouge1, 'rouge2': avg_rouge2}