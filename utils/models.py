import tensorflow as tf
# import tensorflow_addons as tfa

class AdversarialModelAgregator(tf.keras.Model):
    def call(self, outputs):
        # Distributional :
        # Value stream (batch, atoms)
        # Action stream (batch, actions, atoms)

        # Not Distributional (Classic)
        # Value stream (batch, )
        # Action stream (batch, actions)
        outputs = tf.expand_dims(outputs["value"], axis = 1) + outputs["actions"] - tf.math.reduce_mean(outputs["actions"], axis = 1, keepdims=True)
        return outputs


class CustomNoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, sigma=0.1, **kwargs):
        super(CustomNoisyDense, self).__init__(**kwargs)
        self.units = units
        self.sigma = sigma

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, inputs, training=False):
        if training:
            W = self.w + tf.random.normal(self.w.shape, stddev=self.sigma)
        else:
            W = self.w
        return tf.matmul(inputs, W) + self.b
    

class Conv2Plus1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding='same'):
        super(Conv2Plus1D, self).__init__()
        self.seq = tf.keras.Sequential([
            # Spatial decomposition
            tf.keras.layers.Conv3D(filters=filters,
                          kernel_size=(1, kernel_size[1], kernel_size[2]),
                          padding=padding),
            # Temporal decomposition
            tf.keras.layers.Conv3D(filters=filters,
                          kernel_size=(kernel_size[0], 1, 1),
                          padding=padding)
        ])

    def call(self, x):
        return self.seq(x)


class ModelBuilder():
    def __init__(self, units, dropout, nb_states, nb_actions, l2_reg, window, distributional, nb_atoms, adversarial, noisy):
        self.units = units
        self.dropout = dropout
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.l2_reg = l2_reg
        self.recurrent = (window > 1)
        self.window = window
        self.distributional = distributional
        self.nb_atoms = nb_atoms
        self.adversarial = adversarial
        self.noisy = noisy

    
    def dense(self, *args, **kwargs):
        # if self.noisy: return tfa.layers.NoisyDense(*args, sigma= 0.1, **kwargs)
        if self.noisy: return CustomNoisyDense(*args, sigma= 0.1, **kwargs)
        return tf.keras.layers.Dense(*args, **kwargs)
        

    # def build_model(self, trainable = True):
    #     if self.recurrent: inputs = tf.keras.layers.Input(shape=(self.window, self.nb_states))
    #     else : inputs = tf.keras.layers.Input(shape=(self.nb_states,))
    #     main_stream = inputs
    def build_model(self, trainable = True):
        if self.recurrent: 
            inputs = tf.keras.layers.Input(shape=(self.window, self.nb_states))
            # Apply the Conv2Plus1D layer
            inputs = Conv2Plus1D(filters=32, kernel_size=(3, 3, 3))(inputs)
            inputs = tf.keras.layers.MaxPooling3D(pool_size=(2, 1, 1))(inputs)
            inputs = Conv2Plus1D(filters=64, kernel_size=(3, 3, 3))(inputs)
            inputs = tf.keras.layers.MaxPooling3D(pool_size=(2, 1, 1))(inputs)

            # Flatten and add dense layers
            inputs = tf.keras.layers.Flatten()(inputs)
        else : 
            inputs = tf.keras.layers.Input(shape=(self.nb_states,))
            # Apply the Conv2Plus1D layer
            inputs = Conv2Plus1D(filters=32, kernel_size=(3, 3, 3))(inputs)
            inputs = tf.keras.layers.MaxPooling3D(pool_size=(2, 1, 1))(inputs)
            inputs = Conv2Plus1D(filters=64, kernel_size=(3, 3, 3))(inputs)
            inputs = tf.keras.layers.MaxPooling3D(pool_size=(2, 1, 1))(inputs)

            # Flatten and add dense layers
            inputs = tf.keras.layers.Flatten()(inputs)
            
        main_stream = inputs

        # Recurrent
        if self.recurrent:
            #main_stream = self.dense(units=self.units[0], activation = "tanh", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(main_stream)
            for i in range(0, len(self.units)):
                main_stream = tf.keras.layers.LSTM(units= self.units[i],
                                    return_sequences = not(i + 1 == len(self.units)),
                                    dropout = self.dropout,
                                    trainable=trainable)(main_stream)
        # Classic
        else:
            for i in range(len(self.units)):
                main_stream = tf.keras.layers.Dense(
                                    units= self.units[i],
                                    # activation='relu',
                                    activation='mish',
                                    kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                    trainable=trainable)(main_stream)
                if self.dropout > 0: main_stream = tf.keras.layers.Dropout(self.dropout)(main_stream)
        
        # Distributional & Adversarial
        if self.distributional and self.adversarial:
            action_stream = main_stream
            # action_stream = tf.keras.layers.Dense(units = 512, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(action_stream)
            action_stream = tf.keras.layers.Dense(units = 512, activation = "mish", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(action_stream)
            if self.dropout > 0: action_stream = tf.keras.layers.Dropout(self.dropout)(action_stream)
            action_stream = self.dense(units= self.nb_atoms * self.nb_actions, trainable=trainable)(action_stream)
            action_stream = tf.keras.layers.Reshape((self.nb_actions, self.nb_atoms))(action_stream)

            value_stream = main_stream
            # value_stream = tf.keras.layers.Dense(units = 512, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(value_stream)
            value_stream = tf.keras.layers.Dense(units = 512, activation = "mish", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(value_stream)
            if self.dropout > 0: value_stream = tf.keras.layers.Dropout(self.dropout)(value_stream)
            value_stream = self.dense(units = self.nb_atoms, trainable=trainable)(value_stream)

            output = AdversarialModelAgregator()({"value" :value_stream, "actions" : action_stream})
            output = tf.keras.layers.Softmax(axis= -1)(output)
        
        # Only Distributional
        elif self.distributional and not self.adversarial:
            main_stream = self.dense(units= self.nb_atoms * self.nb_actions, trainable=trainable)(main_stream)
            main_stream = tf.keras.layers.Reshape((self.nb_actions, self.nb_atoms))(main_stream)
            output = tf.keras.layers.Softmax(axis= -1)(main_stream)

        # Only Adversarial
        elif not self.distributional and self.adversarial:
            action_stream = main_stream
            # action_stream = tf.keras.layers.Dense(units = 256, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(action_stream)
            action_stream = tf.keras.layers.Dense(units = 256, activation = "mish", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(action_stream)
            action_stream = tf.keras.layers.Dropout(self.dropout)(action_stream)
            action_stream = self.dense(units= self.nb_actions, trainable=trainable)(action_stream)

            value_stream = main_stream
            # value_stream = tf.keras.layers.Dense(units = 256, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(value_stream)
            value_stream = tf.keras.layers.Dense(units = 256, activation = "mish", kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=trainable)(value_stream)
            value_stream = tf.keras.layers.Dropout(self.dropout)(value_stream)
            value_stream = self.dense(units = 1, trainable=trainable)(value_stream)

            output = AdversarialModelAgregator()({"value" :value_stream, "actions" : action_stream})[:, 0, :]
        # Classic
        else:
            output = tf.keras.layers.Dense(units=self.nb_actions, trainable=trainable)(main_stream)
        

        model =  tf.keras.models.Model(inputs = inputs, outputs = output)
        return model