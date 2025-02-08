from tensorflow.keras import layers
import tensorflow as tf
import os
import numpy as np

class Model:
    def __init__(self):
        super().__init__()

    def build_model(self,input_shape, n_classes):
        input_layer = layers.Input(shape=input_shape)

        x = layers.Conv1D(64, 3, activation='relu', padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        residual = x
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, 3, activation=None, padding='same')(x)
        x = layers.BatchNormalization()(x)

        residual = layers.Conv1D(64, 1, activation=None, padding='same')(residual)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        residual = x
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(128, 3, activation=None, padding='same')(x)
        x = layers.BatchNormalization()(x)

        residual = x
        x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(256, 3, activation=None, padding='same')(x)
        x = layers.BatchNormalization()(x)

        residual = layers.Conv1D(256, 1, activation=None, padding='same')(residual)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        x = layers.GlobalAveragePooling1D()(x)

        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output_layer = layers.Dense(n_classes, activation='softmax')(x)

        self.clf = tf.keras.Model(input_layer, output_layer)

    def predict(self, X, batch_size=32):
        stds = np.std(X, axis=-2)[:, np.newaxis, :]
        X = X / stds
        return self.clf.predict(X, batch_size=batch_size)[:, 0]

    def load(self):
        self.build_model(input_shape=(200, 2), n_classes=2)
        self.clf.load_weights(os.path.join(os.path.dirname(__file__), 'fitmodel_020.hdf5'))
