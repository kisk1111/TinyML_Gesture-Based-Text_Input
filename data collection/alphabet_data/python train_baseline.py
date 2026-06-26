#!/usr/bin/env python3
"""
Baseline 1D CNN — A-Z Gesture Classification
Run from the final_data/ folder: python train_baseline.py
"""

import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

LETTERS       = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
WINDOW_SIZE   = 40
N_FEATURES    = 6
EXPECTED_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
BATCH_SIZE    = 32
EPOCHS        = 80
RANDOM_SEED   = 42


def load_dataset(base_dir='.'):
    X, y = [], []
    for letter in LETTERS:
        folder = os.path.join(base_dir, letter)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(f for f in os.listdir(folder) if f.endswith('.csv')):
            rows = []
            with open(os.path.join(folder, fname), newline='') as f:
                for row in csv.DictReader(f):
                    try:
                        rows.append([float(row[c]) for c in EXPECTED_COLS])
                    except (KeyError, ValueError):
                        continue
            if len(rows) >= WINDOW_SIZE:
                X.append(rows[:WINDOW_SIZE])
                y.append(letter)
    return np.array(X, dtype=np.float32), np.array(y)


def build_model(n_classes):
    inputs = keras.Input(shape=(WINDOW_SIZE, N_FEATURES))

    x = layers.Conv1D(16, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.DepthwiseConv1D(3, padding='same', activation='relu')(x)
    x = layers.Conv1D(32, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading dataset...")
    X, y_str = load_dataset(base_dir)
    print(f"  {len(X)} recordings, {len(np.unique(y_str))} classes")

    le    = LabelEncoder()
    y_enc = le.fit_transform(y_str)
    n_cls = len(le.classes_)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y_enc, test_size=0.15, stratify=y_enc, random_state=RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15/0.85, stratify=y_tv, random_state=RANDOM_SEED)

    flat      = X_train.reshape(-1, N_FEATURES)
    mean      = flat.mean(axis=0)
    std       = flat.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_val_n   = (X_val   - mean) / std
    X_test_n  = (X_test  - mean) / std

    print(f"\nBuilding model...")
    model = build_model(n_cls)
    model.summary()

    print(f"\nTraining...")
    model.fit(
        X_train_n, y_train,
        validation_data=(X_val_n, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=15,
                restore_best_weights=True, verbose=0, mode='max'),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=7,
                min_lr=1e-5, verbose=0)
        ],
        verbose=1)

    _, test_acc = model.evaluate(X_test_n, y_test, verbose=0)

    save_path = os.path.join(base_dir, 'baseline_model.keras')
    model.save(save_path)
    size_kb = os.path.getsize(save_path) / 1024

    print(f"\n  Test accuracy:  {test_acc*100:.2f}%")
    print(f"  Parameters:     {model.count_params():,}")
    print(f"  Model size:     {size_kb:.2f} KB")
    print(f"  Saved:          baseline_model.keras")


if __name__ == '__main__':
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    main()