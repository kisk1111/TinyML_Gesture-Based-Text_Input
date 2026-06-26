import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as tfk
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(__file__)
MODEL_TRAINING_DIR = os.path.join(SCRIPT_DIR, "..")
BASELINE_DIR = os.path.join(MODEL_TRAINING_DIR, "baseline")
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

BASELINE_ARCH = [
    ("Conv1D", {"filters": 32, "kernel_size": 3, "activation": "relu", "padding": "same"}),
    ("Conv1D", {"filters": 32, "kernel_size": 3, "activation": "relu", "padding": "same"}),
    ("MaxPooling1D", {"pool_size": 2}),
    ("Conv1D", {"filters": 64, "kernel_size": 3, "activation": "relu", "padding": "same"}),
    ("Conv1D", {"filters": 64, "kernel_size": 3, "activation": "relu", "padding": "same"}),
    ("MaxPooling1D", {"pool_size": 2}),
    ("Conv1D", {"filters": 128, "kernel_size": 3, "activation": "relu", "padding": "same"}),
    ("GlobalAveragePooling1D", {}),
    ("Dropout", {"rate": 0.3}),
    ("Dense", {"units": 64, "activation": "relu"}),
    ("Dense", {"units": None, "activation": "softmax"}),
]

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
    X, y_labels = load_dataset(DATA_DIR)

    label_classes = np.load(os.path.join(BASELINE_DIR, "label_classes.npy"), allow_pickle=True)
    label_to_idx = {label: idx for idx, label in enumerate(label_classes)}
    y_encoded = np.array([label_to_idx[l] for l in y_labels])
    num_classes = len(label_classes)
    y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_encoded,
    )

    norm = np.load(os.path.join(BASELINE_DIR, "norm_params.npz"))
    mean, std = norm["mean"], norm["std"]
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test, label_classes


def load_baseline_weights():
    import keras
    baseline_ref = keras.models.load_model(
        os.path.join(BASELINE_DIR, "baseline_model.keras")
    )
    layer_weights = []
    for layer in baseline_ref.layers:
        w = layer.get_weights()
        if w:
            layer_weights.append(w)
    del baseline_ref
    return layer_weights


def compute_filter_importance(kernel_weights):
    return np.sum(np.abs(kernel_weights), axis=(0, 1))


def compute_neuron_importance(dense_weights):
    return np.sum(np.abs(dense_weights), axis=0)


def prune_conv_layer(kernel, bias, num_to_keep, prev_keep_indices=None):
    if prev_keep_indices is not None:
        kernel = kernel[:, prev_keep_indices, :]

    importance = compute_filter_importance(kernel)
    keep_indices = np.argsort(importance)[-num_to_keep:]
    keep_indices = np.sort(keep_indices)

    pruned_kernel = kernel[:, :, keep_indices]
    pruned_bias = bias[keep_indices]

    return pruned_kernel, pruned_bias, keep_indices


def prune_dense_layer(weights, bias, num_to_keep, prev_keep_indices=None):
    if prev_keep_indices is not None:
        weights = weights[prev_keep_indices, :]

    importance = compute_neuron_importance(weights)
    keep_indices = np.argsort(importance)[-num_to_keep:]
    keep_indices = np.sort(keep_indices)

    pruned_weights = weights[:, keep_indices]
    pruned_bias = bias[keep_indices]

    return pruned_weights, pruned_bias, keep_indices


def structured_prune(baseline_weights, sparsity, num_classes):
    pruned_arch = []
    pruned_weights_list = []

    prev_keep_indices = None
    weight_idx = 0

    for i, (layer_type, kwargs) in enumerate(BASELINE_ARCH):
        if layer_type == "Conv1D":
            original_filters = kwargs["filters"]
            num_to_keep = max(1, int(original_filters * (1 - sparsity)))

            kernel, bias = baseline_weights[weight_idx]
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
            weights, bias = baseline_weights[weight_idx]

            is_final = (kwargs["units"] is None or kwargs["units"] == num_classes)

            if is_final:
                if prev_keep_indices is not None:
                    weights = weights[prev_keep_indices, :]
                pruned_arch.append((layer_type, {"units": num_classes, "activation": kwargs["activation"]}))
                pruned_weights_list.append([weights, bias])
                prev_keep_indices = None
            else:
                original_units = kwargs["units"]
                num_to_keep = max(1, int(original_units * (1 - sparsity)))

                pruned_w, pruned_b, keep_idx = prune_dense_layer(
                    weights, bias, num_to_keep, prev_keep_indices
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


def build_model_from_arch(arch, input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)):
    layers = [tfk.layers.Input(shape=input_shape)]

    for layer_type, kwargs in arch:
        layer_cls = getattr(tfk.layers, layer_type)
        layers.append(layer_cls(**kwargs))

    return tfk.Sequential(layers)


def apply_weights_to_model(model, pruned_weights):
    weight_layers = [l for l in model.layers if l.get_weights()]
    for layer, w in zip(weight_layers, pruned_weights):
        layer.set_weights(w)


def convert_to_tflite(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    return os.path.getsize(output_path) / 1024


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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data and baseline weights...")
    X_train, X_test, y_train, y_test, label_classes = prepare_data()
    num_classes = len(label_classes)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Classes: {num_classes}")

    baseline_weights = load_baseline_weights()
    print(f"Loaded {len(baseline_weights)} weight sets from baseline.")

    baseline_tflite = os.path.join(BASELINE_DIR, "baseline_model.tflite")
    baseline_size = os.path.getsize(baseline_tflite) / 1024

    results = []

    for sparsity in SPARSITY_LEVELS:
        pct = int(sparsity * 100)
        print(f"Structured pruning at {pct}% sparsity...")

        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)

        pruned_arch, pruned_weights = structured_prune(
            baseline_weights, sparsity, num_classes
        )

        print("  Pruned architecture:")
        for layer_type, kwargs in pruned_arch:
            if "filters" in kwargs:
                orig = next(k["filters"] for t, k in BASELINE_ARCH if t == "Conv1D" and k["filters"] >= kwargs["filters"])
                print(f"    {layer_type}: {kwargs['filters']} filters (from {orig})")
            elif "units" in kwargs and layer_type == "Dense" and kwargs.get("activation") != "softmax":
                print(f"    {layer_type}: {kwargs['units']} units (from 64)")

        model = build_model_from_arch(pruned_arch)
        apply_weights_to_model(model, pruned_weights)

        total_params = model.count_params()

        model.compile(
            optimizer=tfk.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        X_train_ft, X_val_ft, y_train_ft, y_val_ft = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train.argmax(axis=1)
        )

        model.fit(
            X_train_ft, y_train_ft,
            epochs=FINETUNE_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val_ft, y_val_ft),
            callbacks=[
                tfk.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=8,
                    restore_best_weights=True,
                ),
            ],
            verbose=0,
        )

        tflite_path = os.path.join(OUTPUT_DIR, f"model_pruned_structured_{pct}.tflite")
        tflite_size = convert_to_tflite(model, tflite_path)
        tflite_acc = evaluate_tflite(tflite_path, X_test, y_test)

        keras_path = os.path.join(OUTPUT_DIR, f"model_pruned_structured_{pct}.keras")
        model.save(keras_path)

        print(f"  Parameters:     {total_params:,}")
        print(f"  Size:           {tflite_size:.1f} KB")
        print(f"  Accuracy:       {tflite_acc * 100:.2f}%")
        print(f"  Size reduction: {(1 - tflite_size / baseline_size) * 100:.1f}%")

        results.append({
            "sparsity": pct,
            "params": total_params,
            "size_kb": tflite_size,
            "accuracy": tflite_acc * 100,
            "size_reduction": (1 - tflite_size / baseline_size) * 100,
        })

    print(f"\n{'=' * 60}")
    print("STRUCTURED PRUNING SUMMARY")
    print(f"{'Sparsity':<10} {'Params':>10} {'Size (KB)':>10} {'Accuracy':>10} {'Size Red.':>10}")
    print(f"{'Baseline':<10} {'56,922':>10} {baseline_size:>10.1f} {'98.27%':>10} {'-':>10}")
    for r in results:
        print(f"{r['sparsity']:>3}%{'':6} {r['params']:>10,} {r['size_kb']:>10.1f} {r['accuracy']:>9.2f}% {r['size_reduction']:>9.1f}%")

    print(f"\nFiles saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()