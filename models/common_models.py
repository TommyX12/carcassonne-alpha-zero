import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from keras.layers import Layer


class TestModel(Model):

    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation='relu')
        self.d2 = layers.Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x


class SpatialPositionalEncoding(Layer):
    def __init__(self, pos_embedding_dim):
        super().__init__()
        self.pos_embedding_dim = pos_embedding_dim

    def call(self, x):
        """
        x: shape (batch_size, c)
        """
        positions = [x]
        # Add sin and cos positional encoding (2 * 3 * pos_embedding_dim)
        for i in range(self.pos_embedding_dim):
            for fn in [tf.sin, tf.cos]:
                positions.append(fn(2.0 ** i * x))

        # Output Shape: [..., 2 * c * pos_embedding_dim + c]
        return tf.concat(positions, axis=-1)


class SimpleFractalModel(Model):
    def __init__(self):
        super().__init__()
        self.dense_layers = [
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1),
        ]

    def call(self, x):
        """
        x['z']: shape (batch_size, 2)
        x['c']: shape (batch_size, 2)
        """
        x = tf.concat([x['z'], x['c']], axis=1)
        for layer in self.dense_layers:
            x = layer(x)
        return x


class SimpleFractalModel2(Model):
    def __init__(self):
        super().__init__()
        self.d1 = layers.Dense(64, activation='relu')
        self.dense_layers = [
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
        ]
        self.d_final = layers.Dense(1)

    def call(self, x):
        """
        x['z']: shape (batch_size, 2)
        x['c']: shape (batch_size, 2)
        """
        x = tf.concat([x['z'], x['c']], axis=1)
        x = self.d1(x)
        for layer in self.dense_layers:
            x = x + layer(x)  # Residual connection

        x = self.d_final(x)
        return x


class SimpleFractalModel3(Model):
    def __init__(self):
        super().__init__()
        self.encoding = SpatialPositionalEncoding(15)
        self.d1 = layers.Dense(64, activation='relu')
        self.dense_layers = [
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
        ]
        self.d_final = layers.Dense(1)

    def call(self, x):
        """
        x['z']: shape (batch_size, 2)
        x['c']: shape (batch_size, 2)
        """
        x = tf.concat([x['z'], x['c']], axis=1)
        x = self.encoding(x)
        x = self.d1(x)
        for layer in self.dense_layers:
            x = x + layer(x)  # Residual connection

        x = self.d_final(x)
        return x


class TransformerFeedForward(Layer):
    def __init__(self, dim1, dim2, **kwargs):
        super().__init__(**kwargs)
        self.d1 = layers.Dense(dim1, activation='relu')
        self.d2 = layers.Dense(dim2)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return x


class TransformerEncoderLayer(Layer):
    def __init__(self, num_heads=8, key_dim=64, value_dim=64, ff_dim=2048, model_dim=512, dropout_rate=0.1):
        """
        Note: if dropout rate is 0, then the dropout layers will be disabled.
        """
        super().__init__()
        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, value_dim=value_dim)
        self.dropout1 = layers.Dropout(
            dropout_rate) if dropout_rate > 0 else None
        self.norm1 = layers.LayerNormalization()
        self.ff = TransformerFeedForward(ff_dim, model_dim)
        self.dropout2 = layers.Dropout(
            dropout_rate) if dropout_rate > 0 else None
        self.norm2 = layers.LayerNormalization()

    def call(self, x, padding_mask=None):
        attention_out = self.multi_head_attention(
            x, x, x, attention_mask=padding_mask)
        if self.dropout1 is not None:
            attention_out = self.dropout1(attention_out)
        x = self.norm1(x + attention_out)
        ff_out = self.ff(x)
        if self.dropout2 is not None:
            ff_out = self.dropout2(ff_out)
        x = self.norm2(x + ff_out)
        return x


class SimpleFractalModel4(Model):
    def __init__(self):
        super().__init__()
        self.encoding_count = 15
        self.encoding = SpatialPositionalEncoding(self.encoding_count)
        self.encoding_output_dim = 2 * 4 * self.encoding_count + 4
        self.model_dim = 64
        self.num_heads = 2
        self.ff_dim = 64
        self.d1 = layers.Dense(self.encoding_output_dim *
                               self.model_dim, activation='relu')
        self.transformer_layers = [
            TransformerEncoderLayer(
                model_dim=self.model_dim, num_heads=self.num_heads, ff_dim=self.ff_dim),
            TransformerEncoderLayer(
                model_dim=self.model_dim, num_heads=self.num_heads, ff_dim=self.ff_dim),
        ]
        self.d_final = layers.Dense(1)

    def call(self, x):
        """
        x['z']: shape (batch_size, 2)
        x['c']: shape (batch_size, 2)
        """
        x = tf.concat([x['z'], x['c']], axis=1)
        # x: shape (batch_size, 4)
        x = self.encoding(x)
        # x: shape (batch_size, 2 * 4 * 15 + 4)
        x = self.d1(x)
        x = tf.reshape(x, (-1, self.encoding_output_dim, self.model_dim))
        for layer in self.transformer_layers:
            x = layer(x)

        x = tf.reshape(x, (-1, self.encoding_output_dim * self.model_dim))
        x = self.d_final(x)
        return x


class ConstantScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def __call__(self, step):
        return self.learning_rate


class TransformerScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, model_dim, warmup_steps=4000):
        super().__init__()

        self.model_dim = model_dim
        self.model_dim = tf.cast(self.model_dim, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.model_dim) * tf.math.minimum(arg1, arg2)


class SimpleFractalModel5(Model):
    def __init__(self):
        super().__init__()
        self.encoding = SpatialPositionalEncoding(15)
        self.d1 = layers.Dense(128, activation='relu')
        self.dense_layers = [
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
        ]
        self.d_final = layers.Dense(1)

    def call(self, x):
        """
        x['z']: shape (batch_size, 2)
        x['c']: shape (batch_size, 2)
        """
        x = tf.concat([x['z'], x['c']], axis=1)
        x = self.encoding(x)
        x = self.d1(x)
        for layer in self.dense_layers:
            x = x + layer(x)  # Residual connection

        x = self.d_final(x)
        return x


class SimpleFractalModel6(Model):
    def __init__(self):
        super().__init__()
        self.encoding = SpatialPositionalEncoding(15)
        self.d1 = layers.Dense(16, activation='leaky_relu')
        self.dense_layers = [
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
            layers.Dense(16, activation='leaky_relu'),
        ]
        self.d_final = layers.Dense(1)

    def call(self, x):
        """
        x['z']: shape (batch_size, 2)
        x['c']: shape (batch_size, 2)
        """
        x = tf.concat([x['z'], x['c']], axis=1)
        x = self.encoding(x)
        x = self.d1(x)
        for layer in self.dense_layers:
            x = x + layer(x)  # Residual connection

        x = self.d_final(x)
        return x


class SimpleFractalModel7(Model):
    def __init__(self):
        super().__init__()
        self.encoding = SpatialPositionalEncoding(15)
        self.d1 = layers.Dense(128, activation='relu')
        d = layers.Dense(128, activation='relu')
        self.dense_layers = [
            d,
            d,
            d,
            d,
        ]
        self.d_final = layers.Dense(1)

    def call(self, x):
        """
        x['z']: shape (batch_size, 2)
        x['c']: shape (batch_size, 2)
        """
        x = tf.concat([x['z'], x['c']], axis=1)
        x = self.encoding(x)
        x = self.d1(x)
        for layer in self.dense_layers:
            x = x + layer(x)  # Residual connection

        x = self.d_final(x)
        return x


class SimpleFractalModel8(Model):
    def __init__(self):
        super().__init__()
        self.encoding = SpatialPositionalEncoding(15)
        self.d1 = layers.Dense(16, activation='leaky_relu')
        d = layers.Dense(16, activation='leaky_relu')
        self.dense_layers = [
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
        ]
        self.d_final = layers.Dense(1)

    def call(self, x):
        """
        x['z']: shape (batch_size, 2)
        x['c']: shape (batch_size, 2)
        """
        x = tf.concat([x['z'], x['c']], axis=1)
        x = self.encoding(x)
        x = self.d1(x)
        for layer in self.dense_layers:
            x = x + layer(x)  # Residual connection

        x = self.d_final(x)
        return x


class SimpleFractalModel9(Model):
    def __init__(self):
        super().__init__()
        self.encoding = SpatialPositionalEncoding(15)
        self.d1 = layers.Dense(16, activation='leaky_relu')
        d = layers.Dense(16, activation='leaky_relu')
        self.dense_layers = [
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
            d,
        ]
        self.d_final = layers.Dense(1)

    def call(self, x):
        """
        x['z']: shape (batch_size, 2)
        x['c']: shape (batch_size, 2)
        """
        x = tf.concat([x['z'], x['c']], axis=1)
        x = self.encoding(x)
        x = self.d1(x)
        for layer in self.dense_layers:
            x = x + layer(x)  # Residual connection

        x = self.d_final(x)
        return x


class SimpleRNNModule(Model):
    def __init__(self, state_dim):
        super().__init__()
        self.init_stddev = 0.01
        self.d1 = layers.Dense(state_dim, activation='leaky_relu',
                               kernel_initializer=keras.initializers.RandomNormal(
                                   stddev=self.init_stddev),
                               bias_initializer=keras.initializers.Zeros())
        self.d2 = layers.Dense(state_dim,
                               kernel_initializer=keras.initializers.RandomNormal(
                                   stddev=self.init_stddev),
                               bias_initializer=keras.initializers.Zeros())

    def call(self, state):
        d_out = self.d1(state)
        d_out = self.d2(d_out)
        state = state + d_out
        return state


class SimpleProbe(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_stddev = 0.01
        self.d1 = layers.Dense(2, kernel_initializer=keras.initializers.RandomNormal(
            stddev=self.init_stddev),
            bias_initializer=keras.initializers.Zeros())
        self.d2 = layers.Dense(1, kernel_initializer=keras.initializers.RandomNormal(
            stddev=self.init_stddev),
            bias_initializer=keras.initializers.Zeros())

    def call(self, x):
        return self.d1(x), self.d2(x)


class SimpleRNNFractalModel(Model):
    def __init__(self, rnn_steps):
        super().__init__()
        self.encoding = SpatialPositionalEncoding(15)
        self.state_dim = 32
        self.rnn_steps = rnn_steps
        self.d1 = layers.Dense(self.state_dim, activation='leaky_relu')
        self.rnn = SimpleRNNModule(self.state_dim)
        self.d_final = layers.Dense(1)

        self.probe = SimpleProbe()

    def call(self, x):
        """
        x['z']: shape (batch_size, 2)
        x['c']: shape (batch_size, 2)
        """
        x = tf.concat([x['z'], x['c']], axis=1)
        x = self.encoding(x)
        x = self.d1(x)
        probed_states, probed_iters = self.probe(x)
        result_probed_states = [probed_states]
        result_probed_iters = [probed_iters]
        for _ in range(self.rnn_steps):
            x = self.rnn(x)
            probed_states, probed_iters = self.probe(x)
            result_probed_states.append(probed_states)
            result_probed_iters.append(probed_iters)

        x = self.d_final(x)
        return x, result_probed_states, result_probed_iters


class ExplicitRNNModule(Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.init_stddev = 0.01
        self.z_d1 = layers.Dense(latent_dim, activation='leaky_relu',
                               kernel_initializer=keras.initializers.RandomNormal(
                                   stddev=self.init_stddev),
                               bias_initializer=keras.initializers.Zeros())
        self.z_d2 = layers.Dense(latent_dim, activation='leaky_relu',
                               kernel_initializer=keras.initializers.RandomNormal(
                                   stddev=self.init_stddev),
                               bias_initializer=keras.initializers.Zeros())
        self.z_d3 = layers.Dense(2,
                               kernel_initializer=keras.initializers.RandomNormal(
                                   stddev=self.init_stddev),
                               bias_initializer=keras.initializers.Zeros())
        self.iters_d1 = layers.Dense(latent_dim, activation='leaky_relu',
                            kernel_initializer=keras.initializers.RandomNormal(
                                stddev=self.init_stddev),
                            bias_initializer=keras.initializers.Zeros())
        self.iters_d2 = layers.Dense(latent_dim, activation='leaky_relu',
                            kernel_initializer=keras.initializers.RandomNormal(
                                stddev=self.init_stddev),
                            bias_initializer=keras.initializers.Zeros())
        self.iters_d3 = layers.Dense(1,
                            kernel_initializer=keras.initializers.RandomNormal(
                                stddev=self.init_stddev),
                            bias_initializer=keras.initializers.Zeros())

    def call(self, state):
        batch_size = tf.shape(state)[0]
        z = state[:, :2]
        c = state[:, 2:4]
        iters = state[:, 4:5]

        x = tf.concat([z, c], axis=1)
        x = tf.concat([x, tf.reshape(tf.einsum('bi,bj->bij', x, x), (batch_size, -1))], axis=1)
        x = self.z_d1(x)
        x = self.z_d2(x)
        z_out = self.z_d3(x)

        x = z_out
        x = tf.concat([x, tf.reshape(tf.einsum('bi,bj->bij', x, x), (batch_size, -1))], axis=1)
        x = self.iters_d1(x)
        x = self.iters_d2(x)
        iters_out = self.iters_d3(x)

        return tf.concat([z_out, iters + iters_out], axis=1)


class ExplicitRNNFractalModel(Model):
    def __init__(self, rnn_steps):
        super().__init__()
        self.encoding = SpatialPositionalEncoding(15)
        self.latent_dim = 64
        self.rnn_steps = rnn_steps
        self.rnn = ExplicitRNNModule(self.latent_dim)

    def call(self, x, gt_z=None, gt_iters=None):
        """
        x['z']: shape (batch_size, 2)
        x['c']: shape (batch_size, 2)
        """
        z = x['z']
        iters = tf.zeros((tf.shape(x['z'])[0], 1))
        result_probed_states = [z]
        result_probed_iters = [iters]
        for i in range(self.rnn_steps):
            if gt_z is not None:
                z = gt_z[i]
            if gt_iters is not None:
                iters = gt_iters[i]
            rnn_out = self.rnn(tf.concat([z, x['c'], iters], axis=1))
            z = rnn_out[:, :2]
            iters = rnn_out[:, 2:]
            result_probed_states.append(z)
            result_probed_iters.append(iters)

        return iters, result_probed_states, result_probed_iters