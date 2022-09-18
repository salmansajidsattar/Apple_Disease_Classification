import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def Model_Art():
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomContrast(0.5),
            layers.experimental.preprocessing.RandomRotation(0.5),
            layers.experimental.preprocessing.RandomZoom(0.5),
        ]
    )
    Model = Sequential([
        data_augmentation,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(input_shape=(300, 300, 3)),
        layers.Dense(1000, activation='relu'),
        layers.Dense(250, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(4, activation='sigmoid')
    ])

    Model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return Model
