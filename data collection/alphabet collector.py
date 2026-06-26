#!/usr/bin/env python3
"""
Alphabet IMU Data Collector
============================
Records 100 repetitions of each letter A-Z using an Arduino IMU.
Each recording captures 360 samples over 3.6 seconds.

Session structure
-----------------
  alphabet_data/
    A/
      A_001.csv ... A_100.csv
    B/
      B_001.csv ... B_100.csv
    ...
    Z/
      Z_001.csv ... Z_100.csv
  alphabet_data/progress.txt   <- auto-saved, enables resume

Records are collected in batches of 25 at a time.

"""

import serial
import serial.tools.list_ports
import time
import csv
import os
import sys
import msvcrt   

CHARACTERS         = [chr(c) for c in range(ord('A'), ord('Z') + 1)]  # A-Z
TOTAL_REPS         = 100       # recordings per letter
BATCH_SIZE         = 25        # how many reps to record per sitting
SAMPLES_PER_GESTURE = 40       # 0.4 s @ 100 Hz (40 usable samples + 20 buffer)
BAUD_RATE          = 115200
OUTPUT_DIR         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alphabet_data")


class AlphabetCollector:
    def __init__(self):
        self.arduino          = None
        self.current_char_idx = 0
        self.current_rep      = 0           
        self.progress_file    = os.path.join(OUTPUT_DIR, "progress.txt")
        self._setup_directories()
        self._load_progress()


    def _setup_directories(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for char in CHARACTERS:
            os.makedirs(os.path.join(OUTPUT_DIR, char), exist_ok=True)

    def _load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file) as f:
                lines = f.readlines()
            if len(lines) >= 2:
                self.current_char_idx = int(lines[0].strip())
                self.current_rep      = int(lines[1].strip())
            print(f"[Resumed] {CHARACTERS[self.current_char_idx]} — rep {self.current_rep + 1}/{TOTAL_REPS}")
        else:
            print("[New session] Starting from A, rep 1")

    def _save_progress(self):
        with open(self.progress_file, 'w') as f:
            f.write(f"{self.current_char_idx}\n{self.current_rep}\n")
    #arduino connection
    def connect_arduino(self) -> bool:
        ports = serial.tools.list_ports.comports()
        for port in ports:
            desc = port.description or ""
            if any(kw in desc for kw in ('Arduino', 'Nano', 'USB Serial', 'CH340', 'CP210')):
                try:
                    self.arduino = serial.Serial(port.device, BAUD_RATE, timeout=0.1)
                    time.sleep(2)   # let the board reset
                    print(f"✓ Connected to {port.device} ({desc})")
                    return True
                except Exception:
                    continue
        for port in ports:
            try:
                self.arduino = serial.Serial(port.device, BAUD_RATE, timeout=0.1)
                time.sleep(2)
                print(f"✓ Connected to {port.device} (fallback)")
                return True
            except Exception:
                continue
        return False


    def _space_pressed(self) -> bool:
        if msvcrt.kbhit():
            return msvcrt.getch() == b' '
        return False

    def _flush_kb(self):
        while msvcrt.kbhit():
            msvcrt.getch()

    #data capture

    def _read_samples(self) -> list | None:
        """
        send 'R' to the board and collect SAMPLES_PER_GESTURE CSV rows.
        """
        data = []
        self.arduino.reset_input_buffer()
        self.arduino.write(b'R')

        deadline = time.time() + 4.0   # generous timeout
        while len(data) < SAMPLES_PER_GESTURE:
            if time.time() > deadline:
                print(f"\n Timeout, only got {len(data)}/{SAMPLES_PER_GESTURE} samples")
                return None
            if self.arduino.in_waiting:
                raw = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                if not raw or 'END' in raw:
                    continue
                # Pass through Arduino cue messages directly to terminal
                if raw.startswith('>>>'):
                    print(f"\n  {raw}")
                    continue
                parts = raw.split(',')
                if len(parts) >= 7:
                    try:
                        data.append([float(p) for p in parts])
                        print(f"\r  Capturing … {len(data):>2}/{SAMPLES_PER_GESTURE}", end='', flush=True)
                    except ValueError:
                        continue

        self.arduino.reset_input_buffer()
        print()  
        return data


    def _run_one(self, char: str, rep_number: int) -> str:
        """
        Countdown → record → save.

        Returns:
            "SUCCESS"          – recording saved
            "DISCARD_PREV"     – space pressed during countdown → discard last
            "FAILED"           – capture returned no data
        """
        print(f"\n  ┌─ {char}  rep {rep_number:>3}/{TOTAL_REPS} {'─'*28}┐")
        print(f"  │  Get ready to write the letter  {char}              │")
        print(f"  │  The board will print START at 0.4 s — gesture then. │")
        print(f"  │  Sample 40 cutoff will be flagged. Stops at 60.       │")
        print(f"  └{'─'*51}┘")

        # Countdown
        self._flush_kb()
        for tick in range(3, 0, -1):
            if self._space_pressed():
                print("  [SPACE] Discarding previous recording …")
                return "DISCARD_PREV"
            print(f"  {tick} …", end=' ', flush=True)
            time.sleep(1)

        print("  GO!", flush=True)

        samples = self._read_samples()
        if not samples:
            return "FAILED"

        # Save CSV
        fname = f"{char}_{rep_number:03d}.csv"
        fpath = os.path.join(OUTPUT_DIR, char, fname)
        with open(fpath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "timestamp_us"])
            writer.writerows(samples)

        print(f"  ✓ Saved → {os.path.relpath(fpath)}")
        return "SUCCESS"


    def _discard_last(self):
        """Roll back one rep and delete its file (if it exists)."""
        rep_to_delete = self.current_rep  
                                          
        if rep_to_delete <= 0:
            print("  [!] Nothing to discard.")
            return

        self.current_rep -= 1
        char = CHARACTERS[self.current_char_idx]
        fname = f"{char}_{self.current_rep + 1:03d}.csv"
        fpath = os.path.join(OUTPUT_DIR, char, fname)

        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"  [✗] Deleted: {fname}")
        else:
            print(f"  [!] File not found (already deleted?): {fname}")

        self._save_progress()


    def _run_batch(self, char: str, batch_start: int) -> bool:
        """
        Record up to BATCH_SIZE reps for `char` starting at `batch_start`.
        Returns True to continue to next batch / letter, False to quit.
        """
        batch_end  = min(batch_start + BATCH_SIZE, TOTAL_REPS)
        batch_num  = batch_start // BATCH_SIZE + 1
        total_batches = (TOTAL_REPS + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\n{'═'*55}")
        print(f"  Letter  : {char}")
        print(f"  Batch   : {batch_num} / {total_batches}  "
              f"(reps {batch_start + 1} – {batch_end})")
        print(f"  Overall : {self.current_char_idx * TOTAL_REPS + batch_start} / "
              f"{len(CHARACTERS) * TOTAL_REPS} total recordings")
        print(f"{'═'*55}")
        print("  Press SPACE during a countdown to discard the previous recording.")
        print("  Press Ctrl+C to pause and return to the batch menu.\n")

        try:
            while self.current_rep < batch_end:
                result = self._run_one(char, self.current_rep + 1)

                if result == "SUCCESS":
                    self.current_rep += 1
                    self._save_progress()
                    if self.current_rep < batch_end:
                        print("  Resting 2 s …")

                elif result == "DISCARD_PREV":
                    self._discard_last()
                    print("  Resuming batch …")

                else:  # FAILED
                    print("  [!] Recording failed. Retrying in 3 s …")
                    time.sleep(3)

        except KeyboardInterrupt:
            print("\n  [Paused] Returning to batch menu …")

        return True  # continue


    def run(self):
        print("   Alphabet IMU Data Collector")
        print("   26 letters × 100 reps × 200 samples @ 100 Hz")

        if not self.connect_arduino():
            print("\n✗ No Arduino found. Check USB connection and try again.")
            sys.exit(1)

        while self.current_char_idx < len(CHARACTERS):
            char = CHARACTERS[self.current_char_idx]

            reps_done = self.current_rep
            if reps_done >= TOTAL_REPS:
                self.current_char_idx += 1
                self.current_rep = 0
                self._save_progress()
                continue

            batch_start = (reps_done // BATCH_SIZE) * BATCH_SIZE

            print(f"\n{'─'*55}")
            print(f"  Up next: Letter  {char}  —  "
                  f"reps {batch_start + 1}–{min(batch_start + BATCH_SIZE, TOTAL_REPS)} "
                  f"(done: {reps_done}/{TOTAL_REPS})")
            print(f"{'─'*55}")
            choice = input("  [R] Record this batch  [D] Discard last  [Q] Quit: ").strip().lower()

            if choice == 'q':
                print("\n  Progress saved. Goodbye!")
                break

            elif choice == 'd':
                self._discard_last()

            elif choice == 'r':
                self._run_batch(char, batch_start)

                if self.current_rep >= TOTAL_REPS:
                    print(f"\n  ★  All {TOTAL_REPS} recordings complete for  {char}!  ★")
                    self.current_char_idx += 1
                    self.current_rep = 0
                    self._save_progress()

            else:
                print("  Unrecognised input — please enter R, D, or Q.")

        if self.current_char_idx >= len(CHARACTERS):
            print("\n" + "★"*55)
            print("  ALL 26 LETTERS COMPLETE — dataset collection finished!")
            print("★"*55 + "\n")


if __name__ == "__main__":
    AlphabetCollector().run()