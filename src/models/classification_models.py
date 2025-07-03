import tensorflow as tf
from tensorflow.keras import layers, models

class RNNClassifier:
    """RNN-based text classifier"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 64, num_classes: int = 3):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.model = None

    def build_model(self) -> tf.keras.Model:
        inputs = layers.Input(shape=(None,))
        x = layers.Embedding(self.vocab_size, self.embedding_dim)(inputs)
        x = layers.SimpleRNN(self.hidden_dim)(x)
        x = layers.Dense(self.num_classes, activation='softmax')(x)
        model = models.Model(inputs, x)
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

class LSTMClassifier:
    """LSTM-based text classifier"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 64, num_classes: int = 3,
                 bidirectional: bool = True):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.model = None

    def build_model(self) -> tf.keras.Model:
        inputs = layers.Input(shape=(None,))
        x = layers.Embedding(self.vocab_size, self.embedding_dim)(inputs)
        if self.bidirectional:
            x = layers.Bidirectional(layers.LSTM(self.hidden_dim))(x)
        else:
            x = layers.LSTM(self.hidden_dim)(x)
        x = layers.Dense(self.num_classes, activation='softmax')(x)
        model = models.Model(inputs, x)
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

class SimpleAttention(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features):
        # features: (batch, timesteps, hidden)
        score = tf.nn.tanh(self.W(features))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context = attention_weights * features
        context = tf.reduce_sum(context, axis=1)
        return context, attention_weights

class AttentionClassifier:
    """Attention-based text classifier"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 64, num_classes: int = 3):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.model = None

    def build_model(self) -> tf.keras.Model:
        inputs = layers.Input(shape=(None,))
        x = layers.Embedding(self.vocab_size, self.embedding_dim)(inputs)
        x = layers.Bidirectional(layers.LSTM(self.hidden_dim, return_sequences=True))(x)
        context, _ = SimpleAttention(self.hidden_dim)(x)
        x = layers.Dense(self.num_classes, activation='softmax')(context)
        model = models.Model(inputs, x)
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