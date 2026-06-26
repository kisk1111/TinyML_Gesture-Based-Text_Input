"""
runs 10 fold cv on a model trained on the data from large slow gestures. output is saved for t test calculation
"""


import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
import sys


OUTPUT_FILE = 'ttest_accuracies_large_200.txt'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'preliminary investigation', 'hypothesis_test_data')

CLASSES = ['3', 'A', 'C', 'e', 'r']
TARGET_KEYWORD = "large"  # finds files like A_large_001


FIXED_SEQ_LEN = 200 
NUM_FEATURES = 6
BATCH_SIZE = 16
EPOCHS = 50 


#load data

X_list = []
y_list = []

for label in CLASSES:
    folder_path = os.path.join(DATA_PATH, label)
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} not found.")
        continue

    # get all csv files and sort them
    files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    
    count = 0
    for f in files:
        if TARGET_KEYWORD in os.path.basename(f):
            try:
                df = pd.read_csv(f)
                data = df.iloc[:, :NUM_FEATURES].values

                if len(data) > FIXED_SEQ_LEN:
                    data = data[:FIXED_SEQ_LEN]
                elif len(data) < FIXED_SEQ_LEN:
                    #pad with zeros if recording is short
                    padding = np.zeros((FIXED_SEQ_LEN - len(data), NUM_FEATURES))
                    data = np.vstack((data, padding))
                
                X_list.append(data)
                y_list.append(label)
                count += 1
            except Exception as e:
                print(f"Error reading {f}: {e}")
    
    print(f"Loaded {count} large files for class '{label}'")

X = np.array(X_list)
y = np.array(y_list)

if len(X) == 0:
    print("ERROR: No data loaded")
    sys.exit()


# Encode Labels (Integers for StratifiedKFold)
le = LabelEncoder()
y_enc = le.fit_transform(y)

#define model
def build_cnn_model():
    inputs = keras.Input(shape=(FIXED_SEQ_LEN, NUM_FEATURES))
    
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    

    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASSES), activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="V4_GAP_Large")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#10 fold CV
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_accuracies = []

print(f"\nStarting 10-Fold Cross Validation on 'Large' Data...")

fold_no = 1
for train_index, test_index in kfold.split(X, y_enc):
    # split
    X_train_raw, X_test_raw = X[train_index], X[test_index]
    y_train_raw, y_test_raw = y_enc[train_index], y_enc[test_index]
    
    #normalise
    scaler = StandardScaler()
    
    X_train_flat = X_train_raw.reshape(-1, NUM_FEATURES)
    X_test_flat = X_test_raw.reshape(-1, NUM_FEATURES)
    
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    X_train = X_train_flat.reshape(X_train_raw.shape)
    X_test = X_test_flat.reshape(X_test_raw.shape)
    
    y_train_cat = to_categorical(y_train_raw, num_classes=len(CLASSES))
    y_test_cat = to_categorical(y_test_raw, num_classes=len(CLASSES))
    
    #train model
    model = build_cnn_model()
    
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    #use 10% of train fold for validation to trigger early stopping
    model.fit(
        X_train, y_train_cat,
        validation_split=0.1, 
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0
    )
    
    # test
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    
    print(f"Fold {fold_no}/10 - Accuracy: {accuracy*100:.2f}%")
    fold_accuracies.append(accuracy)
    
    fold_no += 1

#save results
mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)

print(f"Average Accuracy (Large Data): {mean_acc*100:.2f}% (+/- {std_acc*100:.2f}%)")

with open(OUTPUT_FILE, "w") as f:
    f.write("Fold,Accuracy\n")
    for i, acc in enumerate(fold_accuracies):
        f.write(f"{i+1},{acc:.6f}\n")
    
    f.write("\nSummary:\n")
    f.write(f"Mean,{mean_acc:.6f}\n")
    f.write(f"StdDev,{std_acc:.6f}\n")

print(f"\nResults saved to {OUTPUT_FILE}")