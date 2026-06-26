import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data collection", "final_data")
OUTPUT_DIR = os.path.dirname(__file__)
MODEL_NAME = "baseline_model"

SEQUENCE_LENGTH = 40
NUM_FEATURES = 6  # acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
FEATURE_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

EPOCHS = 50
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Data Loading
def load_dataset(data_dir):
    """Load all CSV files from the dataset directory structure."""
    samples = []
    labels = []
    skipped = 0

    letter_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and len(d) == 1
    ])

    print(f"Found {len(letter_dirs)} letter classes: {letter_dirs}")

    for letter in letter_dirs:
        letter_path = os.path.join(data_dir, letter)
        csv_files = sorted([f for f in os.listdir(letter_path) if f.endswith(".csv")])

        for csv_file in csv_files:
            filepath = os.path.join(letter_path, csv_file)
            try:
                df = pd.read_csv(filepath)
                data = df[FEATURE_COLS].values

                if len(data) < SEQUENCE_LENGTH:
                    # Pad short sequences with zeros
                    pad_length = SEQUENCE_LENGTH - len(data)
                    data = np.pad(data, ((0, pad_length), (0, 0)), mode="constant")
                elif len(data) > SEQUENCE_LENGTH:
                    # Truncate long sequences
                    data = data[:SEQUENCE_LENGTH]

                samples.append(data)
                labels.append(letter)

            except Exception as e:
                print(f"  Skipped {csv_file}: {e}")
                skipped += 1

    print(f"Loaded {len(samples)} samples ({skipped} skipped)")
    return np.array(samples, dtype=np.float32), np.array(labels)



# Normalisation

def normalise(X_train, X_test):
    """Per-feature z-score normalisation fitted on training data only."""
    flat = X_train.reshape(-1, NUM_FEATURES)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0 

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    np.savez(
        os.path.join(OUTPUT_DIR, "norm_params.npz"),
        mean=mean,
        std=std
    )
    print(f"Normalisation params saved (mean: {mean}, std: {std})")

    return X_train_norm, X_test_norm


# Model Architecture
def build_baseline_model(num_classes):
    """3-block 1D CNN with Global Average Pooling."""
    model = tf.keras.Sequential([
        # Input
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),

        # Block 1
        tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        # Block 2
        tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        # Block 3
        tf.keras.layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),

        # Classification head
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model



def main():
    # Load data
    print("Loading dataset...")
    X, y_labels = load_dataset(DATA_DIR)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)
    num_classes = len(label_encoder.classes_)
    y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes)

    print(f"Classes: {list(label_encoder.classes_)}")
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y_onehot.shape}")

    # Save label mapping
    np.save(os.path.join(OUTPUT_DIR, "label_classes.npy"), label_encoder.classes_)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_encoded,
    )
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # Normalise
    X_train, X_test = normalise(X_train, X_test)

    # Build model
    print("Building model...")
    model = build_baseline_model(num_classes)
    model.summary()

    # Train
    print("Training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=10,
                restore_best_weights=True,
            ),
        ],
        verbose=1,
    )

    # Evaluate
    print("Evaluation")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss:     {loss:.4f}")

    # Save Keras model
    keras_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.keras")
    model.save(keras_path)

    # Convert to TFLite 
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    # Report sizes
    keras_size = os.path.getsize(keras_path)
    tflite_size = os.path.getsize(tflite_path)

    print("Model Sizes")
    print(f"Keras model:  {keras_size / 1024:.1f} KB")
    print(f"TFLite model: {tflite_size / 1024:.1f} KB")
    print(f"\nTotal parameters: {model.count_params():,}")

    print("Parameter Breakdown")
    for layer in model.layers:
        if layer.count_params() > 0:
            print(f"  {layer.name:30s} {layer.count_params():>8,} params")

    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print(f"  - {MODEL_NAME}.keras")
    print(f"  - {MODEL_NAME}.tflite")
    print(f"  - norm_params.npz")
    print(f"  - label_classes.npy")


if __name__ == "__main__":
    main()