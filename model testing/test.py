"""

sends test samples to Arduino over Serial.
detects int8 models from Arduino DEBUG output and pre-quantizes data.


command:  python test.py --port COM3 --model_name "model_name"
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import serial

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TRAINING_DIR = os.path.join(SCRIPT_DIR, "model training")
BASELINE_DIR = os.path.join(MODEL_TRAINING_DIR, "baseline")
DATA_DIR = os.path.join(SCRIPT_DIR, "data collection", "final_data")

SEQUENCE_LENGTH = 40
NUM_FEATURES = 6
FEATURE_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42


def load_test_data():
    samples, labels = [], []
    letter_dirs = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and len(d) == 1
    ])
    for letter in letter_dirs:
        letter_path = os.path.join(DATA_DIR, letter)
        for csv_file in sorted(f for f in os.listdir(letter_path) if f.endswith(".csv")):
            try:
                df = pd.read_csv(os.path.join(letter_path, csv_file))
                data = df[FEATURE_COLS].values
                if len(data) < SEQUENCE_LENGTH:
                    data = np.pad(data, ((0, SEQUENCE_LENGTH - len(data)), (0, 0)), mode="constant")
                elif len(data) > SEQUENCE_LENGTH:
                    data = data[:SEQUENCE_LENGTH]
                samples.append(data)
                labels.append(letter)
            except Exception as e:
                print(f"  Skipped {csv_file}: {e}")

    X = np.array(samples, dtype=np.float32)
    y = np.array(labels)
    label_classes = np.load(os.path.join(BASELINE_DIR, "label_classes.npy"), allow_pickle=True)
    label_to_idx = {label: idx for idx, label in enumerate(label_classes)}
    y_encoded = np.array([label_to_idx[l] for l in y])

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y_encoded, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=y_encoded,
    )
    norm = np.load(os.path.join(BASELINE_DIR, "norm_params.npz"))
    X_test = (X_test - norm["mean"]) / norm["std"]
    return X_test, y_test


def send_and_wait(ser, message, timeout=10):
    ser.write((message + "\n").encode("utf-8"))
    return ser.readline().decode("utf-8", errors="ignore").strip()


def evaluate_on_device(port, baud, model_name, timeout=30):
    print("Loading test data...")
    X_test, y_test = load_test_data()
    print(f"Test samples: {len(X_test)}")

    print(f"Connecting to {port} at {baud} baud...")
    ser = serial.Serial(port, baud, timeout=timeout)

    print("Waiting for Arduino to be ready...")
    arena_used = 0
    input_type = 1
    input_scale = 1.0
    input_zp = 0
    start_time = time.time()

    while time.time() - start_time < timeout:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            print(f"  Arduino: {line}")
            if line.startswith("DEBUG:INPUT_TYPE="):
                input_type = int(line.split("=")[1])
            elif line.startswith("DEBUG:INPUT_SCALE="):
                input_scale = float(line.split("=")[1])
            elif line.startswith("DEBUG:INPUT_ZP="):
                input_zp = int(line.split("=")[1])
            elif line.startswith("READY:"):
                arena_used = int(line.split(":")[1])
                print(f"  Arena used: {arena_used} bytes ({arena_used / 1024:.1f} KB)")
                break
            elif line.startswith("ERROR:"):
                print(f"  FATAL: {line}")
                ser.close()
                return
    else:
        print("Timeout waiting for Arduino.")
        ser.close()
        return

    is_int8 = (input_type == 9)
    if is_int8:
        print(f"  INT8 model detected (scale={input_scale}, zp={input_zp})")
        print(f"  Pre-quantizing data on Python side.")
        X_quant = np.round(X_test / input_scale) + input_zp
        X_quant = np.clip(X_quant, -128, 127).astype(np.int8)
    else:
        print(f"  Float32 model detected.")

    response = send_and_wait(ser, "PING")
    if response != "PONG":
        print(f"  Communication check failed. Got: {response}")
        ser.close()
        return
    print("  Communication OK.\n")

    results = []
    correct = 0
    total = len(X_test)
    errors = 0

    for i in range(total):
        response = send_and_wait(ser, "START")
        if response != "OK":
            print(f"  Sample {i}: START failed: {response}")
            errors += 1
            continue

        timestep_ok = True
        for t in range(SEQUENCE_LENGTH):
            if is_int8:
                row = ",".join(str(int(v)) for v in X_quant[i][t])
            else:
                row = ",".join(f"{v:.6f}" for v in X_test[i][t])

            response = send_and_wait(ser, row)
            if response != "OK":
                print(f"  Sample {i}, timestep {t}: {response}")
                timestep_ok = False
                break

        if not timestep_ok:
            errors += 1
            continue

        response = send_and_wait(ser, "RUN")
        if response.startswith("RESULT:"):
            parts = response.split(":")[1].split(",")
            predicted = int(parts[0])
            latency_us = int(parts[1])
            is_correct = (predicted == y_test[i])
            if is_correct:
                correct += 1
            results.append({
                "sample": i,
                "true_class": int(y_test[i]),
                "predicted_class": predicted,
                "correct": is_correct,
                "latency_us": latency_us,
            })
        else:
            print(f"  Sample {i}: Unexpected response: {response}")
            errors += 1

        if (i + 1) % 50 == 0 or i == total - 1:
            if results:
                running_acc = correct / len(results) * 100
                avg_latency = np.mean([r["latency_us"] for r in results])
                print(f"  [{i+1}/{total}] Accuracy: {running_acc:.1f}% | "
                      f"Avg latency: {avg_latency:.0f} us | Errors: {errors}")

    send_and_wait(ser, "DONE")
    ser.close()

    if not results:
        print("No successful inferences.")
        return

    latencies = [r["latency_us"] for r in results]
    accuracy = correct / len(results) * 100

    print(f"\n{'=' * 60}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:        {accuracy:.2f}% ({correct}/{len(results)})")
    print(f"  Arena used:      {arena_used} bytes ({arena_used / 1024:.1f} KB)")
    print(f"  Avg latency:     {np.mean(latencies):.0f} us ({np.mean(latencies)/1000:.2f} ms)")
    print(f"  Min latency:     {np.min(latencies):.0f} us")
    print(f"  Max latency:     {np.max(latencies):.0f} us")
    print(f"  Std latency:     {np.std(latencies):.0f} us")
    print(f"  Transfer errors: {errors}")

    csv_path = f"eval_results_{model_name}.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\n  Results saved to {csv_path}")

    summary_path = "eval_summary.csv"
    summary_exists = os.path.exists(summary_path)
    with open(summary_path, "a") as f:
        if not summary_exists:
            f.write("model,accuracy,arena_bytes,arena_kb,avg_latency_us,avg_latency_ms,min_latency_us,max_latency_us,errors\n")
        f.write(f"{model_name},{accuracy:.2f},{arena_used},{arena_used/1024:.1f},"
                f"{np.mean(latencies):.0f},{np.mean(latencies)/1000:.2f},"
                f"{np.min(latencies):.0f},{np.max(latencies):.0f},{errors}\n")
    print(f"  Summary appended to {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--model_name", default="unknown")
    args = parser.parse_args()
    evaluate_on_device(args.port, args.baud, args.model_name)


if __name__ == "__main__":
    main()