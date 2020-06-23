import argparse
import os
import json

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.losses import BinaryCrossentropy


IMG_HEIGHT = 224
IMG_WIDTH = 224


def get_generators(training_path, validation_path, batch_size):
    training_image_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=45,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.5,
        fill_mode="constant",
        cval=0,
    )

    validation_image_generator = ImageDataGenerator(rescale=1.0 / 255,)

    training_generator = training_image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=training_path,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode="binary",
    )

    validation_generator = validation_image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=validation_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode="binary",
    )

    return training_generator, validation_generator


def create_model(learning_rate):
    model = Sequential(
        [
            Conv2D(
                16,
                3,
                padding="same",
                activation="relu",
                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
            ),
            MaxPooling2D(2, 2),
            Dropout(0.2),
            Conv2D(32, 3, padding="same", activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(64, 3, padding="same", activation="relu"),
            MaxPooling2D(2, 2),
            Dropout(0.2),
            Flatten(),
            Dense(512, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)

    parser.add_argument("--training", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))

    args, _ = parser.parse_known_args()

    training_generator, validation_generator = get_generators(
        training_path=args.training, validation_path=args.validation, batch_size=args.batch_size
    )

    model = create_model(learning_rate=args.learning_rate)

    model.fit(
        training_generator, epochs=args.epochs, validation_data=validation_generator
    )

    model_path = json.loads(os.environ["SM_TRAINING_ENV"])["model_dir"]
    model.save(os.path.join(model_path, "12345"))

