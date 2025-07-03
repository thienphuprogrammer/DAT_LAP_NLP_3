import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, TFAutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from typing import Any

class TransformerClassifier:
    """Fine-tuned transformer cho classification"""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased',
                 num_classes: int = 3):
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def build_model(self) -> TFAutoModelForSequenceClassification:
        self.model = TFAutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_classes)
        return self.model

    def train(self, train_dataset, val_dataset, epochs: int = 3):
        # train_dataset, val_dataset: tf.data.Dataset
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        self.model.compile(optimizer=optimizer, loss=self.model.compute_loss, metrics=['accuracy'])
        history = self.model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
        return history

class TransformerTranslator:
    """Fine-tuned transformer cho translation"""
    
    def __init__(self, model_name: str = 'Helsinki-NLP/opus-mt-vi-en'):
        self.model_name = model_name
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def build_model(self) -> TFAutoModelForSeq2SeqLM:
        self.model = TFAutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        return self.model

    def train(self, train_dataset, val_dataset, epochs: int = 3):
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        self.model.compile(optimizer=optimizer, loss=self.model.compute_loss)
        history = self.model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
        return history

    def translate(self, text: str) -> str:
        inputs = self.tokenizer([text], return_tensors='tf')
        output = self.model.generate(**inputs, max_length=50)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

class TransformerCaptioner:
    """Fine-tuned transformer cho image captioning"""
    
    def __init__(self, model_name: str = 'nlpconnect/vit-gpt2-image-captioning'):
        self.model_name = model_name
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = None

    def build_model(self):
        # Sử dụng pipeline cho inference
        self.pipe = pipeline('image-to-text', model=self.model_name)
        return self.pipe

    def generate_caption(self, image_path: str) -> str:
        if self.pipe is None:
            self.build_model()
        result = self.pipe(image_path)
        return result[0]['generated_text'] if result else ''