import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as tfk
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(__file__)
MODEL_TRAINING_DIR = os.path.join(SCRIPT_DIR, "..")
BASELINE_DIR = os.path.join(MODEL_TRAINING_DIR, "baseline")
KD_DIR = os.path.join(MODEL_TRAINING_DIR, "knowledge distillation", "new_results")
DATA_DIR = os.path.join(MODEL_TRAINING_DIR, "..", "data collection", "final_data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "new_results")

SEQUENCE_LENGTH = 40
NUM_FEATURES = 6
FEATURE_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

SPARSITY_LEVELS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

FINETUNE_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

STUDENT_CONFIGS = {
    "big": {
        "arch": [
            ("Conv1D", {"filters": 16, "kernel_size": 3, "activation": "relu", "padding": "same"}),
            ("Conv1D", {"filters": 16, "kernel_size": 3, "activation": "relu", "padding": "same"}),
            ("MaxPooling1D", {"pool_size": 2}),
            ("Conv1D", {"filters": 32, "kernel_size": 3, "activation": "relu", "padding": "same"}),
            ("Conv1D", {"filters": 32, "kernel_size": 3, "activation": "relu", "padding": "same"}),
            ("MaxPooling1D", {"pool_size": 2}),
            ("Conv1D", {"filters": 64, "kernel_size": 3, "activation": "relu", "padding": "same"}),
            ("GlobalAveragePooling1D", {}),
            ("Dropout", {"rate": 0.3}),
            ("Dense", {"units": 32, "activation": "relu"}),
            ("Dense", {"units": 26}),
        ],
    },
}


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
                    data = np.pad(data, ((0, SEQUENCE_LENGTH - len(data)), (0, 0)), mode="constant")
                elif len(data) > SEQUENCE_LENGTH:
                    data = data[:SEQUENCE_LENGTH]
                samples.append(data)
                labels.append(letter)
            except Exception as e:
                print(f"  Skipped {csv_file}: {e}")
    return np.array(samples, dtype=np.float32), np.array(labels)


def prepare_data():
    X, y_labels = load_dataset(DATA_DIR)
    label_classes = np.load(os.path.join(BASELINE_DIR, "label_classes.npy"), allow_pickle=True)
    label_to_idx = {label: idx for idx, label in enumerate(label_classes)}
    y_encoded = np.array([label_to_idx[l] for l in y_labels])
    num_classes = len(label_classes)
    y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=y_encoded,
    )
    norm = np.load(os.path.join(BASELINE_DIR, "norm_params.npz"))
    mean, std = norm["mean"], norm["std"]
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test, y_train, y_test, label_classes


def load_kd_weights(student_name):
    keras_path = os.path.join(KD_DIR, f"model_kd_{student_name}.keras")
    if not os.path.exists(keras_path):
        return None

    try:
        model = tfk.models.load_model(keras_path)
    except Exception:
        import keras
        model = keras.models.load_model(keras_path)

    layer_weights = []
    for layer in model.layers:
        w = layer.get_weights()
        if w:
            layer_weights.append(w)
    del model
    return layer_weights


def compute_filter_importance(kernel_weights):
    return np.sum(np.abs(kernel_weights), axis=(0, 1))


def compute_neuron_importance(dense_weights):
    return np.sum(np.abs(dense_weights), axis=0)


def prune_conv_layer(kernel, bias, num_to_keep, prev_keep_indices=None):
    if prev_keep_indices is not None:
        kernel = kernel[:, prev_keep_indices, :]
    importance = compute_filter_importance(kernel)
    keep_indices = np.sort(np.argsort(importance)[-num_to_keep:])
    return kernel[:, :, keep_indices], bias[keep_indices], keep_indices


def prune_dense_layer(weights, bias, num_to_keep, prev_keep_indices=None):
    if prev_keep_indices is not None:
        weights = weights[prev_keep_indices, :]
    importance = compute_neuron_importance(weights)
    keep_indices = np.sort(np.argsort(importance)[-num_to_keep:])
    return weights[:, keep_indices], bias[keep_indices], keep_indices


def structured_prune(arch, weights, sparsity, num_classes):
    pruned_arch = []
    pruned_weights_list = []
    prev_keep_indices = None
    weight_idx = 0

    for layer_type, kwargs in arch:
        if layer_type == "Conv1D":
            original_filters = kwargs["filters"]
            num_to_keep = max(1, int(original_filters * (1 - sparsity)))

            kernel, bias = weights[weight_idx]
            pruned_k, pruned_b, keep_idx = prune_conv_layer(
                kernel, bias, num_to_keep, prev_keep_indices
            )

            new_kwargs = dict(kwargs)
            new_kwargs["filters"] = num_to_keep
            pruned_arch.append((layer_type, new_kwargs))
            pruned_weights_list.append([pruned_k, pruned_b])
            prev_keep_indices = keep_idx
            weight_idx += 1

        elif layer_type == "Dense":
            w, b = weights[weight_idx]

            is_final = (kwargs.get("units") == num_classes and kwargs.get("activation") is None)

            if is_final:
                if prev_keep_indices is not None:
                    w = w[prev_keep_indices, :]
                pruned_arch.append((layer_type, kwargs))
                pruned_weights_list.append([w, b])
                prev_keep_indices = None
            else:
                original_units = kwargs["units"]
                num_to_keep = max(1, int(original_units * (1 - sparsity)))

                pruned_w, pruned_b, keep_idx = prune_dense_layer(
                    w, b, num_to_keep, prev_keep_indices
                )

                new_kwargs = dict(kwargs)
                new_kwargs["units"] = num_to_keep
                pruned_arch.append((layer_type, new_kwargs))
                pruned_weights_list.append([pruned_w, pruned_b])
                prev_keep_indices = keep_idx

            weight_idx += 1

        elif layer_type == "GlobalAveragePooling1D":
            pruned_arch.append((layer_type, kwargs))

        else:
            pruned_arch.append((layer_type, kwargs))

    return pruned_arch, pruned_weights_list


def build_model_from_arch(arch):
    layers = [tfk.layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES))]
    for layer_type, kwargs in arch:
        layers.append(getattr(tfk.layers, layer_type)(**kwargs))
    return tfk.Sequential(layers)


def apply_weights_to_model(model, pruned_weights):
    weight_layers = [l for l in model.layers if l.get_weights()]
    for layer, w in zip(weight_layers, pruned_weights):
        layer.set_weights(w)


def convert_to_tflite(model, output_path):
    export_model = tfk.Sequential([
        model,
        tfk.layers.Activation("softmax"),
    ])
    export_model.predict(
        np.zeros((1, SEQUENCE_LENGTH, NUM_FEATURES), dtype=np.float32), verbose=0
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return os.path.getsize(output_path) / 1024


def evaluate_tflite(tflite_path, X_test, y_test):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    total = len(X_test)

    for i in range(total):
        sample = X_test[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        if np.argmax(output) == np.argmax(y_test[i]):
            correct += 1

    return correct / total


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    X_train, X_test, y_train, y_test, label_classes = prepare_data()
    num_classes = len(label_classes)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Classes: {num_classes}")

    X_train_ft, X_val_ft, y_train_ft, y_val_ft = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train.argmax(axis=1)
    )

    baseline_tflite = os.path.join(BASELINE_DIR, "baseline_model.tflite")
    baseline_size = os.path.getsize(baseline_tflite) / 1024

    for student_name, config in STUDENT_CONFIGS.items():
        print(f"# Student: {student_name}")

        kd_weights = load_kd_weights(student_name)
        if kd_weights is None:
            print(f"  Skipping model_kd_{student_name}.keras not found in {KD_DIR}")
            continue

        arch = config["arch"]
        results = []

        for sparsity in SPARSITY_LEVELS:
            pct = int(sparsity * 100)
            print(f"KD {student_name} + structured pruning at {pct}%...")

            np.random.seed(RANDOM_SEED)
            tf.random.set_seed(RANDOM_SEED)

            pruned_arch, pruned_weights = structured_prune(
                arch, kd_weights, sparsity, num_classes
            )

            model = build_model_from_arch(pruned_arch)
            apply_weights_to_model(model, pruned_weights)
            total_params = model.count_params()

            model.compile(
                optimizer=tfk.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss=tfk.losses.CategoricalCrossentropy(from_logits=True),
                metrics=[tfk.metrics.CategoricalAccuracy()],
            )

            model.fit(
                X_train_ft, y_train_ft,
                epochs=FINETUNE_EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_val_ft, y_val_ft),
                callbacks=[
                    tfk.callbacks.EarlyStopping(
                        monitor="val_categorical_accuracy",
                        patience=8,
                        restore_best_weights=True,
                    ),
                ],
                verbose=0,
            )

            keras_path = os.path.join(OUTPUT_DIR, f"model_kd_{student_name}_pruned_{pct}.keras")
            model.save(keras_path)

            tflite_path = os.path.join(OUTPUT_DIR, f"model_kd_{student_name}_pruned_{pct}.tflite")
            tflite_size = convert_to_tflite(model, tflite_path)
            tflite_acc = evaluate_tflite(tflite_path, X_test, y_test)

            print(f"  Parameters: {total_params:,}")
            print(f"  Size:       {tflite_size:.1f} KB")
            print(f"  Accuracy:   {tflite_acc * 100:.2f}%")
            print(f"  Size red.:  {(1 - tflite_size / baseline_size) * 100:.1f}%")

            results.append({
                "sparsity": pct,
                "params": total_params,
                "size_kb": tflite_size,
                "accuracy": tflite_acc * 100,
                "size_reduction": (1 - tflite_size / baseline_size) * 100,
            })

        print(f"KD + STRUCTURED PRUNING SUMMARY — {student_name}")
        print(f"{'Sparsity':<10} {'Params':>10} {'Size (KB)':>10} {'Accuracy':>10} {'Size Red.':>10}")
        for r in results:
            print(f"{r['sparsity']:>3}%{'':6} {r['params']:>10,} {r['size_kb']:>10.1f} {r['accuracy']:>9.2f}% {r['size_reduction']:>9.1f}%")

    print(f"\nFiles saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()