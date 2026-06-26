import os
import numpy as np
import pandas as pd
import tensorflow as tf


SCRIPT_DIR = os.path.dirname(__file__)
MODEL_TRAINING_DIR = os.path.join(SCRIPT_DIR, "..")
BASELINE_DIR = os.path.join(MODEL_TRAINING_DIR, "baseline")
DATA_DIR = os.path.join(MODEL_TRAINING_DIR, "..", "data collection", "final_data")
OUTPUT_DIR = SCRIPT_DIR

SEQUENCE_LENGTH = 40
NUM_FEATURES = 6
FEATURE_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_dataset(data_dir):
    samples = []
    labels = []

    letter_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and len(d) == 1
    ])

    for letter in letter_dirs:
        letter_path = os.path.join(data_dir, letter)
        csv_files = sorted([f for f in os.listdir(letter_path) if f.endswith(".csv")])

        for csv_file in csv_files:
            filepath = os.path.join(letter_path, csv_file)
            try:
                df = pd.read_csv(filepath)
                data = df[FEATURE_COLS].values

                if len(data) < SEQUENCE_LENGTH:
                    pad_length = SEQUENCE_LENGTH - len(data)
                    data = np.pad(data, ((0, pad_length), (0, 0)), mode="constant")
                elif len(data) > SEQUENCE_LENGTH:
                    data = data[:SEQUENCE_LENGTH]

                samples.append(data)
                labels.append(letter)
            except Exception as e:
                print(f"  Skipped {csv_file}: {e}")

    return np.array(samples, dtype=np.float32), np.array(labels)


def prepare_data():
    """Load data, split, and normalise using saved baseline params."""
    X, y_labels = load_dataset(DATA_DIR)

    # Encode labels
    label_classes = np.load(os.path.join(BASELINE_DIR, "label_classes.npy"), allow_pickle=True)
    label_to_idx = {label: idx for idx, label in enumerate(label_classes)}
    y_encoded = np.array([label_to_idx[l] for l in y_labels])
    num_classes = len(label_classes)
    y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes)

    # Same train/test split as baseline
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_encoded,
    )

    # Normalise using saved baseline params
    norm = np.load(os.path.join(BASELINE_DIR, "norm_params.npz"))
    mean, std = norm["mean"], norm["std"]
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test, label_classes


# Representative dataset needed for full quantisation
def make_representative_dataset(X_train):
    def representative_dataset():
        indices = np.random.choice(len(X_train), size=min(200, len(X_train)), replace=False)
        for i in indices:
            yield [X_train[i:i+1].astype(np.float32)]
    return representative_dataset



def evaluate_tflite(tflite_path, X_test, y_test):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]
    input_quant = input_details[0].get("quantization_parameters", {})
    input_scale = input_quant.get("scales", np.array([]))
    input_zp = input_quant.get("zero_points", np.array([]))

    correct = 0
    total = len(X_test)

    for i in range(total):
        sample = X_test[i:i+1].astype(np.float32)

        if input_dtype != np.float32 and len(input_scale) > 0 and input_scale[0] != 0:
            sample = (sample / input_scale[0]) + input_zp[0]
            sample = np.clip(sample, -128, 127).astype(input_dtype)

        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        predicted = np.argmax(output)
        actual = np.argmax(y_test[i])
        if predicted == actual:
            correct += 1

    return correct / total


def quantise_float16(model, output_path):
    """Float16 quantisation - weights stored as float16, computed as float32."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return output_path


def quantise_int8(model, output_path, representative_dataset_fn):
    """Full integer (int8) quantisation - weights and activations quantised."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_fn
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return output_path



def main():
    # Load data
    print("=" * 60)
    print("Loading data and baseline model...")
    print("=" * 60)
    X_train, X_test, y_train, y_test, label_classes = prepare_data()
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Classes: {len(label_classes)}")

    # Load baseline model
    model_path = os.path.join(BASELINE_DIR, "baseline_model.keras")
    model = tf.keras.models.load_model(model_path)
    print("Baseline model loaded.")

    # Baseline TFLite for reference
    baseline_tflite = os.path.join(BASELINE_DIR, "baseline_model.tflite")
    baseline_size = os.path.getsize(baseline_tflite) / 1024

    
    # 16-bit quantisation
    print("\n" + "=" * 60)
    print("16-bit (Float16) Quantisation")
    print("=" * 60)
    fp16_path = os.path.join(OUTPUT_DIR, "model_quant_fp16.tflite")
    quantise_float16(model, fp16_path)
    fp16_size = os.path.getsize(fp16_path) / 1024
    fp16_acc = evaluate_tflite(fp16_path, X_test, y_test)
    print(f"  Size:     {fp16_size:.1f} KB")
    print(f"  Accuracy: {fp16_acc * 100:.2f}%")

    # 8-bit quantisation
    print("\n" + "=" * 60)
    print("8-bit (Int8) Quantisation")
    print("=" * 60)
    int8_path = os.path.join(OUTPUT_DIR, "model_quant_int8.tflite")
    quantise_int8(model, int8_path, make_representative_dataset(X_train))
    int8_size = os.path.getsize(int8_path) / 1024
    int8_acc = evaluate_tflite(int8_path, X_test, y_test)
    print(f"  Size:     {int8_size:.1f} KB")
    print(f"  Accuracy: {int8_acc * 100:.2f}%")

    print("\n" + "=" * 60)
    print("QUANTISATION SUMMARY")
    print(f"{'Model':<25} {'Size (KB)':>10} {'Accuracy':>10} {'Size Reduction':>15}")
    print(f"{'Baseline (float32)':<25} {baseline_size:>10.1f} {'98.27%':>10} {'-':>15}")
    print(f"{'Float16':<25} {fp16_size:>10.1f} {fp16_acc*100:>9.2f}% {(1 - fp16_size/baseline_size)*100:>14.1f}%")
    print(f"{'Int8':<25} {int8_size:>10.1f} {int8_acc*100:>9.2f}% {(1 - int8_size/baseline_size)*100:>14.1f}%")

    print(f"\nFiles saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()