"""

host-side data collection script for the Preliminary Investigation.
communicates with the Arduino Nano 33 BLE Sense over Serial to log 6-axis IMU data.
automates the collection of Large (200 samples), Small-Slow (200 samples), 
and Small-Fastspeed (40 samples) gestures.
"""
import serial
import serial.tools.list_ports
import time
import csv
import os
import msvcrt  #used for non-blocking keyboard input

CHARACTERS = ['3', 'A', 'C', 'e', 'r']
GESTURE_SIZES = ['large', 'small', 'small_fastspeed']
REPETITIONS = 50
TOTAL_RECORDINGS = len(CHARACTERS) * len(GESTURE_SIZES) * REPETITIONS

class HypothesisTestCollector:
    def __init__(self):
        self.arduino = None
        self.current_char_idx = 0
        self.current_size_idx = 0
        self.current_rep = 0
        self.base_dir = "hypothesis_test_data"
        self.progress_file = os.path.join(self.base_dir, "progress.txt")
        self.setup_directories()
        self.load_progress()
        
    def setup_directories(self):
        if not os.path.exists(self.base_dir): os.makedirs(self.base_dir)
        for char in CHARACTERS:
            path = os.path.join(self.base_dir, char)
            if not os.path.exists(path): os.makedirs(path)
    
    def load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    self.current_char_idx = int(lines[0].strip())
                    self.current_size_idx = int(lines[1].strip())
                    self.current_rep = int(lines[2].strip())
    
    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            f.write(f"{self.current_char_idx}\n{self.current_size_idx}\n{self.current_rep}")

    def connect_arduino(self):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if any(x in port.description for x in ['Arduino', 'Nano', 'USB Serial']):
                try:
                    self.arduino = serial.Serial(port.device, 115200, timeout=0.1)
                    time.sleep(2) 
                    print(f"Connected to {port.device}")
                    return True
                except: continue
        return False

    def check_space(self):
        """Returns True if spacebar was pressed since last check."""
        if msvcrt.kbhit():
            if msvcrt.getch() == b' ': return True
        return False

    def read_fixed_samples(self, count):
        data = []
        self.arduino.reset_input_buffer()
        self.arduino.write(b'R') 
        
        start_time = time.time()
        while len(data) < count:
            if (time.time() - start_time) > 5.0: 
                print(f"\n[!] TIMEOUT: Got {len(data)}/{count}")
                return None

            if self.arduino.in_waiting > 0:
                line = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                
                if not line or "END" in line:
                    continue
                    
                parts = line.split(',')
                if len(parts) >= 7:
                    try:
                        data.append([float(p) for p in parts])
                        print(f"\rRecording: {len(data)}/{count} samples", end='')
                    except ValueError:
                        continue
        
        self.arduino.reset_input_buffer()
        return data

    def discard_last(self):
        """Backs up progress by 1 and deletes the physical file."""
        if self.current_rep > 0: 
            self.current_rep -= 1
        elif self.current_size_idx > 0:
            self.current_size_idx -= 1
            self.current_rep = REPETITIONS - 1
        
        char, size = CHARACTERS[self.current_char_idx], GESTURE_SIZES[self.current_size_idx]
        fname = f"{char}_{size}_{self.current_rep+1:03d}.csv"
        fpath = os.path.join(self.base_dir, char, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"\n[X] DELETED: {fname}")
        self.save_progress()

    def run_cycle(self, char, size, rep):
        """Single recording execution with countdown."""
        print(f"\nREADY: {char} ({size.upper()}) | Rep {rep}/{REPETITIONS}")
        for i in range(3, 0, -1):
            if self.check_space(): return "DISCARD_PREVIOUS"
            print(f"{i}...", end=' ', flush=True); time.sleep(1)
        
        print("GO!")
        
        # 40 samples for the fastspeed gestures, 200 samples for the slow ones.
        target_samples = 40 if size == 'small_fastspeed' else 200
        
        samples = self.read_fixed_samples(target_samples)
        if samples:
            fname = f"{char}_{size}_{rep:03d}.csv"
            fpath = os.path.join(self.base_dir, char, fname)
            with open(fpath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "timestamp"])
                writer.writerows(samples)
            return "SUCCESS"
        return "CANCELLED"

    def run(self):
        if not self.connect_arduino(): 
            print("Arduino not found!"); return
            
        while self.current_char_idx < len(CHARACTERS):
            char, size = CHARACTERS[self.current_char_idx], GESTURE_SIZES[self.current_size_idx]
            print(f"\n{'='*40}\nTASK: {char} - {size.upper()} (Next: {self.current_rep+1}/50)\n{'='*40}")
            mode = input("[C]ontinuous mode, [S]ingle rep, [D]iscard last, [Q]uit: ").lower()
            
            if mode == 'q': break
            if mode == 'd': self.discard_last(); continue
            
            if mode == 'c':
                print(f"\n--- Starting Continuous Mode for {char} {size} ---")
                try:
                    while self.current_rep < REPETITIONS:
                        res = self.run_cycle(char, size, self.current_rep + 1)
                        if res == "SUCCESS":
                            self.current_rep += 1
                            self.save_progress()
                            if self.current_rep < REPETITIONS:
                                print("\nPause 2s... (Press SPACE now to STOP loop)"); time.sleep(2)
                                if self.check_space(): break 
                        elif res == "DISCARD_PREVIOUS":
                            self.discard_last()
                            break # Return to main menu
                        else: # Cancelled current
                            break # Return to main menu
                except KeyboardInterrupt: pass

            elif mode == 's':
                res = self.run_cycle(char, size, self.current_rep + 1)
                if res == "SUCCESS":
                    self.current_rep += 1
                    self.save_progress()
                elif res == "DISCARD_PREVIOUS":
                    self.discard_last()

            if self.current_rep >= REPETITIONS:
                print(f"\n*** FINISHED {char} {size.upper()} ***")
                self.current_rep = 0
                self.current_size_idx += 1
                if self.current_size_idx >= len(GESTURE_SIZES):
                    self.current_size_idx = 0
                    self.current_char_idx += 1
                self.save_progress()

if __name__ == "__main__":
    HypothesisTestCollector().run()