"""
runs 10 fold cv on a model trained on the data from small fast gestures. output is saved for t test calculation
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


OUTPUT_FILE = 'ttest_accuracies_small_40.txt'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'preliminary investigation', 'hypothesis_test_data')

CLASSES = ['3', 'A', 'C', 'e', 'r']
TARGET_KEYWORD = "fastspeed"

FIXED_SEQ_LEN = 40 
NUM_FEATURES = 6
BATCH_SIZE = 16
EPOCHS = 50 

#load data
print(f"Loading data from: {DATA_PATH}...")

X_list = []
y_list = []

for label in CLASSES:
    folder_path = os.path.join(DATA_PATH, label)
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} not found.")
        continue

    files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    
    count = 0
    for f in files:
        if TARGET_KEYWORD in os.path.basename(f):
            try:
                df = pd.read_csv(f)
                data = df.iloc[:, :NUM_FEATURES].values
                
                #zero pad if not enough data
                if len(data) > FIXED_SEQ_LEN:
                    data = data[:FIXED_SEQ_LEN]
                elif len(data) < FIXED_SEQ_LEN:
                    padding = np.zeros((FIXED_SEQ_LEN - len(data), NUM_FEATURES))
                    data = np.vstack((data, padding))
                
                X_list.append(data)
                y_list.append(label)
                count += 1
            except Exception as e:
                print(f"Error reading {f}: {e}")
    
    print(f"Loaded {count} files for class '{label}'")

X = np.array(X_list)
y = np.array(y_list)

if len(X) == 0:
    print("ERROR: No data loaded.")
    sys.exit()

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
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#10 fold cv
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_accuracies = []

print(f"\nStarting 10-Fold Cross Validation on {len(X)} samples...")

fold_no = 1
for train_index, test_index in kfold.split(X, y_enc):
    #data split
    X_train_raw, X_test_raw = X[train_index], X[test_index]
    y_train_raw, y_test_raw = y_enc[train_index], y_enc[test_index]
    
    # normalise data
    scaler = StandardScaler()
    
    X_train_flat = X_train_raw.reshape(-1, NUM_FEATURES)
    X_test_flat = X_test_raw.reshape(-1, NUM_FEATURES)
    
    X_train_flat = scaler.fit_transform(X_train_flat) 
    X_test_flat = scaler.transform(X_test_flat)      
    
    X_train = X_train_flat.reshape(X_train_raw.shape)
    X_test = X_test_flat.reshape(X_test_raw.shape)
    
    y_train_cat = to_categorical(y_train_raw, num_classes=len(CLASSES))
    y_test_cat = to_categorical(y_test_raw, num_classes=len(CLASSES))
    
    model = build_cnn_model()
    
    #train model
    history = model.fit(
        X_train, y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0  # Silent training
    )
    
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    
    print(f"Fold {fold_no}/10 - Accuracy: {accuracy*100:.2f}%")
    fold_accuracies.append(accuracy)
    
    fold_no += 1

#save results for t test calculation
mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)

print(f"Average Accuracy: {mean_acc*100:.2f}% (+/- {std_acc*100:.2f}%)")

with open(OUTPUT_FILE, "w") as f:
    f.write("Fold,Accuracy\n")
    for i, acc in enumerate(fold_accuracies):
        f.write(f"{i+1},{acc:.6f}\n")
    
    f.write("\nSummary:\n")
    f.write(f"Mean,{mean_acc:.6f}\n")
    f.write(f"StdDev,{std_acc:.6f}\n")

print(f"\nResults saved to {OUTPUT_FILE}")