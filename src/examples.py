from transformers import pipeline, AutoTokenizer
import torch
from rouge_score import rouge_scorer
import pandas as pd

def generate_examples_lstm(model, word_to_idx, idx_to_word, examples, device='cuda', max_length=15):
    """Генерирует примеры для LSTM модели"""
    results = []
    model.eval()
    model.to(device)
    
    for example in examples:
        try:
            # Токенизируем префикс
            tokens = example.lower().split()
            token_indices = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]
            
            # Генерируем продолжение
            generated_indices = model.generate(token_indices, max_length=max_length)
            generated_tokens = [idx_to_word.get(idx, '<UNK>') for idx in generated_indices[len(token_indices):]]
            
            # Объединяем в строку
            generated_text = ' '.join(generated_tokens).strip()
            results.append(generated_text)
            
        except Exception as e:
            print(f"Ошибка генерации для '{example}': {e}")
            results.append("ошибка генерации")
    
    return results

def generate_examples_transformer(examples, model_name="distilgpt2", max_new_tokens=20):
    """Генерирует примеры для Transformer модели"""
    results = []
    
    # Инициализируем pipeline один раз
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text-generation", 
        model=model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    for example in examples:
        try:
            result = generator(
                example,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                truncation=True
            )
            
            generated_text = result[0]['generated_text']
            # Убираем оригинальный префикс
            continuation = generated_text[len(example):].strip()
            # ✅ Очищаем от некорректных Unicode символов
            continuation = ''.join(char for char in continuation if char.isprintable())
            continuation = continuation.encode('utf-8', 'ignore').decode('utf-8')

            results.append(continuation)
            
        except Exception as e:
            print(f"Ошибка генерации для '{example}': {e}")
            results.append("ошибка генерации")
    
    return results