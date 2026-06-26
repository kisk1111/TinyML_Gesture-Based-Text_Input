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

TEMPERATURE = 5.0
ALPHA = 0.1
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Distiller class adapted from:
# Borup, K. (2020) Knowledge Distillation
# https://keras.io/examples/vision/knowledge_distillation/
class Distiller(tfk.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data

        teacher_pred = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_pred = self.student(x, training=True)

            student_loss = self.student_loss_fn(y, student_pred)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_pred / self.temperature, axis=1),
                tf.nn.softmax(student_pred / self.temperature, axis=1),
            ) * (self.temperature ** 2)

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, student_pred)
        results = {m.name: m.result() for m in self.metrics}
        results["student_loss"] = student_loss
        results["distillation_loss"] = distillation_loss
        return results

    def test_step(self, data):
        x, y = data
        student_pred = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, student_pred)

        self.compiled_metrics.update_state(y, student_pred)
        results = {m.name: m.result() for m in self.metrics}
        results["student_loss"] = student_loss
        return results

    def call(self, x):
        return self.student(x)


STUDENT_CONFIGS = {
    "small": [
        ("Conv1D", {"filters": 16, "kernel_size": 3, "activation": "relu", "padding": "same"}),
        ("Conv1D", {"filters": 16, "kernel_size": 3, "activation": "relu", "padding": "same"}),
        ("MaxPooling1D", {"pool_size": 2}),
        ("GlobalAveragePooling1D", {}),
        ("Dropout", {"rate": 0.3}),
        ("Dense", {"units": 16, "activation": "relu"}),
    ],
    "medium": [
        ("Conv1D", {"filters": 16, "kernel_size": 3, "activation": "relu", "padding": "same"}),
        ("Conv1D", {"filters": 16, "kernel_size": 3, "activation": "relu", "padding": "same"}),
        ("MaxPooling1D", {"pool_size": 2}),
        ("Conv1D", {"filters": 32, "kernel_size": 3, "activation": "relu", "padding": "same"}),
        ("Conv1D", {"filters": 32, "kernel_size": 3, "activation": "relu", "padding": "same"}),
        ("MaxPooling1D", {"pool_size": 2}),
        ("GlobalAveragePooling1D", {}),
        ("Dropout", {"rate": 0.3}),
        ("Dense", {"units": 32, "activation": "relu"}),
    ],
    "big": [
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
    ],
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


def load_teacher(num_classes):
    model = tfk.Sequential([
        tfk.layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
        tfk.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        tfk.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        tfk.layers.MaxPooling1D(pool_size=2),
        tfk.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        tfk.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        tfk.layers.MaxPooling1D(pool_size=2),
        tfk.layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        tfk.layers.GlobalAveragePooling1D(),
        tfk.layers.Dropout(0.3),
        tfk.layers.Dense(64, activation="relu"),
        tfk.layers.Dense(num_classes),
    ], name="teacher")

    import keras
    baseline_ref = keras.models.load_model(
        os.path.join(BASELINE_DIR, "baseline_model.keras")
    )
    src_weights = [l.get_weights() for l in baseline_ref.layers if l.get_weights()]
    dst_layers = [l for l in model.layers if l.get_weights()]
    for dst, w in zip(dst_layers, src_weights):
        dst.set_weights(w)
    del baseline_ref

    return model


def build_student(arch, num_classes, name="student"):
    layers = [tfk.layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES))]
    for layer_type, kwargs in arch:
        layers.append(getattr(tfk.layers, layer_type)(**kwargs))
    layers.append(tfk.layers.Dense(num_classes))
    return tfk.Sequential(layers, name=name)


def convert_to_tflite(student_model, num_classes, output_path):
    export_model = tfk.Sequential([
        student_model,
        tfk.layers.Activation("softmax"),
    ])
    export_model.predict(np.zeros((1, SEQUENCE_LENGTH, NUM_FEATURES), dtype=np.float32), verbose=0)

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

    print("Loading data and teacher model...")
    X_train, X_test, y_train, y_test, label_classes = prepare_data()
    num_classes = len(label_classes)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Classes: {num_classes}")

    teacher = load_teacher(num_classes)
    print("Teacher (baseline) model loaded.")

    teacher.compile(
        loss=tfk.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tfk.metrics.CategoricalAccuracy()],
    )
    _, teacher_acc = teacher.evaluate(X_test, y_test, verbose=0)
    print(f"Teacher accuracy: {teacher_acc * 100:.2f}%")

    baseline_tflite = os.path.join(BASELINE_DIR, "baseline_model.tflite")
    baseline_size = os.path.getsize(baseline_tflite) / 1024

    X_train_ft, X_val_ft, y_train_ft, y_val_ft = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train.argmax(axis=1)
    )

    results = []

    for name, arch in STUDENT_CONFIGS.items():
        print(f"Student: {name}")

        print("  Training standalone (no teacher)...")
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)

        student_scratch = build_student(arch, num_classes, name=f"student_{name}_scratch")
        total_params = student_scratch.count_params()
        print(f"  Parameters: {total_params:,}")

        student_scratch.compile(
            optimizer=tfk.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=tfk.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[tfk.metrics.CategoricalAccuracy()],
        )
        student_scratch.fit(
            X_train_ft, y_train_ft,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val_ft, y_val_ft),
            callbacks=[
                tfk.callbacks.EarlyStopping(
                    monitor="val_categorical_accuracy", patience=10,
                    restore_best_weights=True,
                ),
            ],
            verbose=1,
        )
        _, standalone_acc = student_scratch.evaluate(X_test, y_test, verbose=0)
        print(f"  Standalone accuracy: {standalone_acc * 100:.2f}%")

        print(f"\n  Distilling teacher to student (T={TEMPERATURE}, alpha={ALPHA})...")
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)

        student = build_student(arch, num_classes, name=f"student_{name}")

        distiller = Distiller(student=student, teacher=teacher)
        distiller.compile(
            optimizer=tfk.optimizers.Adam(learning_rate=LEARNING_RATE),
            metrics=[tfk.metrics.CategoricalAccuracy()],
            student_loss_fn=tfk.losses.CategoricalCrossentropy(from_logits=True),
            distillation_loss_fn=tfk.losses.KLDivergence(),
            alpha=ALPHA,
            temperature=TEMPERATURE,
        )
        distiller.fit(
            X_train_ft, y_train_ft,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val_ft, y_val_ft),
            callbacks=[
                tfk.callbacks.EarlyStopping(
                    monitor="val_categorical_accuracy", patience=10,
                    restore_best_weights=True,
                ),
            ],
            verbose=1,
        )

        tflite_path = os.path.join(OUTPUT_DIR, f"model_kd_{name}.tflite")
        tflite_size = convert_to_tflite(student, num_classes, tflite_path)
        tflite_acc = evaluate_tflite(tflite_path, X_test, y_test)

        student.save(os.path.join(OUTPUT_DIR, f"model_kd_{name}.keras"))

        print(f"\n  Results for {name}:")
        print(f"    Standalone accuracy: {standalone_acc * 100:.2f}%")
        print(f"    Distilled accuracy:  {tflite_acc * 100:.2f}%")
        print(f"    KD improvement:      {(tflite_acc - standalone_acc) * 100:+.2f}%")
        print(f"    Size:                {tflite_size:.1f} KB")
        print(f"    Size reduction:      {(1 - tflite_size / baseline_size) * 100:.1f}%")

        results.append({
            "name": name,
            "params": total_params,
            "standalone_acc": standalone_acc * 100,
            "kd_acc": tflite_acc * 100,
            "kd_improvement": (tflite_acc - standalone_acc) * 100,
            "size_kb": tflite_size,
            "size_reduction": (1 - tflite_size / baseline_size) * 100,
        })

    print(f"\n{'=' * 60}")
    print("KNOWLEDGE DISTILLATION SUMMARY")
    print(f"{'Student':<10} {'Params':>10} {'Standalone':>12} {'Distilled':>12} {'KD Gain':>10} {'Size(KB)':>10} {'Size Red.':>10}")
    print(f"{'Teacher':<10} {'56,922':>10} {teacher_acc*100:>11.2f}% {'-':>12} {'-':>10} {baseline_size:>10.1f} {'-':>10}")
    for r in results:
        print(f"{r['name']:<10} {r['params']:>10,} {r['standalone_acc']:>11.2f}% {r['kd_acc']:>11.2f}% {r['kd_improvement']:>+9.2f}% {r['size_kb']:>10.1f} {r['size_reduction']:>9.1f}%")

    print(f"\nFiles saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()