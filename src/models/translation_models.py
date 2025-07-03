import tensorflow as tf
from tensorflow.keras import layers, models
from typing import List

class Encoder:
    """Encoder cho translation model"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256,
                 hidden_dim: int = 512):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

    def build(self) -> tf.keras.layers.Layer:
        inputs = layers.Input(shape=(None,))
        x = layers.Embedding(self.vocab_size, self.embedding_dim)(inputs)
        output, state_h, state_c = layers.LSTM(self.hidden_dim, return_state=True, return_sequences=True)(x)
        return models.Model(inputs, [output, state_h, state_c])

class Decoder:
    """Decoder cho translation model"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256,
                 hidden_dim: int = 512):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

    def build(self) -> tf.keras.layers.Layer:
        inputs = layers.Input(shape=(None,))
        enc_output = layers.Input(shape=(None, self.hidden_dim))
        state_h = layers.Input(shape=(self.hidden_dim,))
        state_c = layers.Input(shape=(self.hidden_dim,))
        x = layers.Embedding(self.vocab_size, self.embedding_dim)(inputs)
        lstm_out, _, _ = layers.LSTM(self.hidden_dim, return_sequences=True, return_state=True)(x, initial_state=[state_h, state_c])
        context, attn_weights = AttentionLayer(self.hidden_dim)([lstm_out, enc_output, enc_output])
        concat = layers.Concatenate()([lstm_out, context])
        outputs = layers.TimeDistributed(layers.Dense(self.vocab_size, activation='softmax'))(concat)
        return models.Model([inputs, enc_output, state_h, state_c], outputs)

class Seq2SeqTranslator:
    """Sequence-to-sequence translation model"""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 embedding_dim: int = 256, hidden_dim: int = 512,
                 use_attention: bool = True):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.encoder = Encoder(src_vocab_size, embedding_dim, hidden_dim)
        self.decoder = Decoder(tgt_vocab_size, embedding_dim, hidden_dim)
        self.model = None

    def build_model(self) -> tf.keras.Model:
        # Encoder
        enc_inputs = layers.Input(shape=(None,))
        enc_model = self.encoder.build()
        enc_outputs, state_h, state_c = enc_model(enc_inputs)
        # Decoder
        dec_inputs = layers.Input(shape=(None,))
        dec_model = self.decoder.build()
        dec_outputs = dec_model([dec_inputs, enc_outputs, state_h, state_c])
        model = models.Model([enc_inputs, dec_inputs], dec_outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val,
              epochs: int = 50, batch_size: int = 32):
        if self.model is None:
            self.build_model()
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size
        )
        return history

    def translate(self, input_sequence: List[int]) -> List[int]:
        # Giả sử đã build model và load weights
        enc_model = self.encoder.build()
        dec_model = self.decoder.build()
        # Lấy encoder states
        enc_outputs, state_h, state_c = enc_model.predict([input_sequence])
        # Bắt đầu với token <BOS>
        target_seq = [1]  # Giả sử 1 là <BOS>
        output_seq = []
        for _ in range(50):
            dec_out = dec_model.predict([target_seq, enc_outputs, state_h, state_c])
            sampled_token = int(tf.argmax(dec_out[0, -1, :]))
            if sampled_token == 2:  # Giả sử 2 là <EOS>
                break
            output_seq.append(sampled_token)
            target_seq = [sampled_token]
        return output_seq

class AttentionLayer(tf.keras.layers.Layer):
    """Custom attention layer"""
    
    def __init__(self, units: int):
        super().__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, inputs, **kwargs):
        # query: decoder hidden state (batch, t, h)
        # value/key: encoder outputs (batch, s, h)
        query, value, key = inputs
        query_with_time_axis = tf.expand_dims(query, 2) if len(query.shape) == 2 else query
        score = self.V(tf.nn.tanh(self.W1(query) + self.W2(value)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * value
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights