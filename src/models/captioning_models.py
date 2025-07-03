import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class CNNEncoder:
    """CNN encoder cho image captioning"""
    
    def __init__(self, cnn_model: str = 'InceptionV3'):
        self.cnn_model = cnn_model
        if cnn_model == 'InceptionV3':
            base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', pooling='avg')
            self.model = models.Model(base_model.input, base_model.output)
            self.feature_dim = 2048
        else:
            raise NotImplementedError(f"Model {cnn_model} not supported.")

    def build(self) -> tf.keras.Model:
        return self.model

class BahdanauAttention(layers.Layer):
    """Bahdanau attention mechanism"""
    
    def __init__(self, units: int):
        super().__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, query, value):
        # query: (batch, hidden), value: (batch, seq, hidden)
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(value)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * value
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class RNNDecoder:
    """RNN decoder cho image captioning"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256,
                 hidden_dim: int = 512, feature_dim: int = 2048):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.attention = BahdanauAttention(hidden_dim)

    def build(self) -> tf.keras.Model:
        features = layers.Input(shape=(self.feature_dim,))
        dec_input = layers.Input(shape=(None,))
        x = layers.Embedding(self.vocab_size, self.embedding_dim)(dec_input)
        hidden_state = layers.Input(shape=(self.hidden_dim,))
        # Expand dims for attention
        features_seq = layers.RepeatVector(tf.shape(x)[1])(features)
        context_vector, _ = self.attention(hidden_state, features_seq)
        x = layers.Concatenate()([tf.expand_dims(context_vector, 1), x])
        output, state_h, state_c = layers.LSTM(self.hidden_dim, return_state=True, return_sequences=True)(x, initial_state=[hidden_state, hidden_state])
        output = layers.TimeDistributed(layers.Dense(self.vocab_size, activation='softmax'))(output)
        return models.Model([features, dec_input, hidden_state], output)

class ImageCaptioner:
    """Complete image captioning model"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256,
                 hidden_dim: int = 512, use_attention: bool = True):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.encoder = CNNEncoder()
        self.decoder = RNNDecoder(vocab_size, embedding_dim, hidden_dim)
        self.model = None

    def build_model(self) -> tf.keras.Model:
        # Encoder
        img_input = layers.Input(shape=(299, 299, 3))
        cnn_model = self.encoder.build()
        features = cnn_model(img_input)
        # Decoder
        cap_input = layers.Input(shape=(None,))
        hidden_input = layers.Input(shape=(self.hidden_dim,))
        dec_model = self.decoder.build()
        outputs = dec_model([features, cap_input, hidden_input])
        model = models.Model([img_input, cap_input, hidden_input], outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    def train(self, image_features, captions, epochs: int = 50):
        if self.model is None:
            self.build_model()
        # image_features: (num_samples, 299, 299, 3)
        # captions: (num_samples, seq_len)
        # hidden_input: zeros
        hidden_input = np.zeros((image_features.shape[0], self.hidden_dim))
        history = self.model.fit(
            [image_features, captions, hidden_input],
            captions,
            epochs=epochs
        )
        return history

    def generate_caption(self, image_features: np.ndarray, 
                        max_length: int = 20) -> str:
        # Giả sử đã build model và load weights
        cnn_model = self.encoder.build()
        dec_model = self.decoder.build()
        features = cnn_model.predict(np.expand_dims(image_features, 0))
        hidden = np.zeros((1, self.hidden_dim))
        input_seq = [1]  # <BOS>
        result = []
        for _ in range(max_length):
            preds = dec_model.predict([features, np.array([input_seq]), hidden])
            pred_id = int(np.argmax(preds[0, -1, :]))
            if pred_id == 2:  # <EOS>
                break
            result.append(pred_id)
            input_seq.append(pred_id)
        # Cần map id -> word ở ngoài
        return result