import numpy as np
from transformers import pipeline, AutoTokenizer, TFAutoModelForSeq2SeqLM
import tensorflow as tf

class InferenceEngine:
    """Inference utilities"""
    
    def __init__(self, model_path: str, tokenizer_path: str):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None
    
    def predict_caption_length(self, caption: str) -> str:
        # Giả sử đã load model phân loại độ dài caption
        if self.model is None:
            self.model = tf.keras.models.load_model(self.model_path)
        # Tiền xử lý caption nếu cần
        # Xử lý đơn giản: đếm số từ
        n_words = len(caption.split())
        if n_words < 5:
            return 'short'
        elif n_words < 10:
            return 'medium'
        else:
            return 'long'
    
    def translate_text(self, text: str, source_lang: str = 'vi') -> str:
        # Sử dụng pipeline transformers cho dịch
        if self.model is None or self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self.model = TFAutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        pipe = pipeline('translation', model=self.model, tokenizer=self.tokenizer, src_lang=source_lang)
        result = pipe(text)
        return result[0]['translation_text'] if result else ''
    
    def generate_image_caption(self, image_path: str) -> str:
        # Sử dụng pipeline transformers cho image captioning
        pipe = pipeline('image-to-text', model=self.model_path)
        result = pipe(image_path)
        return result[0]['generated_text'] if result else ''