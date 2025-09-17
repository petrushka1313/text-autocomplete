from transformers import pipeline, AutoTokenizer
from rouge_score import rouge_scorer
import pandas as pd
import torch# Очистка кэша CUDA
import gc


def evaluate_transformer(model_name="distilgpt2", num_samples=100):
    """Сравнивает качество предобученной модели Transformer"""
    print(f"Loading model {model_name}...")
    # Используем pipeline для простоты
    # Проверяем доступность GPU
    device = 0 if torch.cuda.is_available() else -1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    generator = pipeline("text-generation", model=model_name, device=device,
                         torch_dtype=torch.float16 if device == 0 else torch.float32 ) # device=0 для GPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # Устанавливаем pad token

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []

    # Загружаем валидационные данные (чистый текст)
    val_texts = pd.read_csv('data/val.csv')['text_clean'].tolist()[:num_samples]

    for text in val_texts:
        if not isinstance(text, str) or len(text) < 10:
            continue

        # Определяем префикс (3/4 текста)
        split_point = int(len(text) * 0.75)
        prefix = text[:split_point]
        reference = text[split_point:]

        # Генерируем продолжение
        # Важно подобрать параметры. do_sample=True и temperature/too_k/topp дают более разнообразные результаты.
        result = generator(
            prefix,
            max_new_tokens=20,
            max_length=len(text) + 20, # Макс. длина = исходная длина + немного
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = result[0]['generated_text']

        # Убираем префикс из сгенерированного текста, чтобы получить чистое продолжение
        prediction = generated_text[len(prefix):]

        # Считаем ROUGE
        scores = scorer.score(reference, prediction)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)

    print(f"Transformer ({model_name}) ROUGE-1: {avg_rouge1:.4f}, ROUGE-2: {avg_rouge2:.4f}")
    return {'rouge1': avg_rouge1, 'rouge2': avg_rouge2}

if __name__ == '__main__':
    evaluate_transformer()