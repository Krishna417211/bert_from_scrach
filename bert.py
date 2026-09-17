import math
import tensorflow as tf
from tensorflow.keras import layers, Model


class PositionalEmbedding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.pe = layers.Embedding(input_dim=max_len, output_dim=d_model)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        batch_size = tf.shape(inputs)[0]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = tf.expand_dims(positions, 0)
        positions = tf.tile(positions, [batch_size, 1])
        return self.pe(positions)


class BERTEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_model, max_len=512, dropout=0.1, pad_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.token = layers.Embedding(input_dim=vocab_size, output_dim=d_model, mask_zero=False)
        self.position = PositionalEmbedding(max_len=max_len, d_model=d_model)
        self.segment = layers.Embedding(input_dim=2, output_dim=d_model)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(rate=dropout)

    def call(self, sequence, segment_label=None, training=False):
        if segment_label is None:
            segment_label = tf.zeros_like(sequence)
            
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(self.norm(x), training=training)


class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, d_model, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
        self.dropout = layers.Dropout(rate=dropout)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None, training=False):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(self.depth, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
            
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        
        return self.dense(concat_attention)


class PositionwiseFeedForward(layers.Layer):
    def __init__(self, d_model, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.w_1 = layers.Dense(d_ff, activation="gelu")
        self.w_2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(rate=dropout)

    def call(self, x, training=False):
        x = self.w_1(x)
        x = self.dropout(x, training=training)
        return self.w_2(x)


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(rate=dropout)
        self.dropout2 = layers.Dropout(rate=dropout)

    def call(self, x, mask=None, training=False):
        attn_output = self.mha(x, x, x, mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class BERTPooler(layers.Layer):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(hidden_size, activation="tanh")

    def call(self, hidden_states):
        return self.dense(hidden_states[:, 0, :])


class BERT(Model):
    def __init__(self, vocab_size, d_model=128, n_layers=4, heads=4, dropout=0.1, max_len=512, d_ff=None, pad_idx=0, **kwargs):
        super().__init__(**kwargs)
        if d_ff is None:
            d_ff = d_model * 4

        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        self.pad_idx = pad_idx

        self.embedding = BERTEmbedding(vocab_size=vocab_size, d_model=d_model, max_len=max_len, dropout=dropout, pad_idx=pad_idx)
        self.encoder_blocks = [
            TransformerBlock(d_model=d_model, num_heads=heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ]
        self.pooler = BERTPooler(d_model)

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, self.pad_idx), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def call(self, inputs, segment_info=None, training=False):
        if isinstance(inputs, (tuple, list)):
            x = inputs[0]
            if len(inputs) > 1:
                segment_info = inputs[1]
        else:
            x = inputs

        mask = self.create_padding_mask(x)
        x = self.embedding(x, segment_label=segment_info, training=training)

        for block in self.encoder_blocks:
            x = block(x, mask=mask, training=training)

        return x, self.pooler(x)


class MaskedLanguageModel(layers.Layer):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(vocab_size)

    def call(self, x):
        return self.dense(x)


class NextSentencePrediction(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(2)

    def call(self, x):
        return self.dense(x)


class BERTLM(Model):
    def __init__(self, bert: BERT, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(vocab_size)
        self.next_sentence = NextSentencePrediction()

    def call(self, inputs, segment_label=None, training=False):
        if isinstance(inputs, (tuple, list)):
            x = inputs[0]
            if len(inputs) > 1:
                segment_label = inputs[1]
        else:
            x = inputs

        sequence_output, pooled_output = self.bert((x, segment_label), training=training)
        return self.mask_lm(sequence_output), self.next_sentence(pooled_output)


if __name__ == "__main__":
    vocab_size = 10000
    batch_size = 2
    seq_len = 10

    bert_model = BERT(vocab_size=vocab_size, d_model=128, n_layers=4, heads=4)

    input_ids = tf.constant([
        [12, 345, 67, 890, 11, 44, 55, 66, 0, 0],
        [99, 123, 456, 789, 10, 20, 30, 40, 50, 60]
    ], dtype=tf.int32)
    
    segment_ids = tf.constant([
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    ], dtype=tf.int32)

    seq_out, pooled_out = bert_model((input_ids, segment_ids))
    print("Input shape :", input_ids.shape)
    print("Sequence output shape:", seq_out.shape)
    print("Pooled output shape  :", pooled_out.shape)

    bert_lm = BERTLM(bert_model, vocab_size=vocab_size)
    mlm_logits, nsp_logits = bert_lm((input_ids, segment_ids))
    print("MLM logits shape:", mlm_logits.shape)
    print("NSP logits shape:", nsp_logits.shape)
