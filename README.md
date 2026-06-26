# TinyML_Gesture-Based-Text_Input

This project investigates model compression techniques for deploying a gesture-based text input
system on the Arduino Nano 33 BLE Sense, a microcontroller with 256KB of SRAM and 1MB
of Flash. Air-writing offers a keyboard-free input method for wearable and embedded devices. A
preliminary investigation confirmed that small, fast gestures (3cm, 0.4s) achieve statistically equiv
alent classification accuracy to larger, slower gestures, validating their use for practical text input. A
dataset of 2,600 samples covering 26 uppercase characters was collected from 6-axis IMU data and
used to train a baseline 1D CNN achieving 98.27% accuracy. Structured pruning, post-training int8
quantisation, and knowledge distillation were applied individually and in combination, producing
19 compressed model variants evaluated across accuracy, model size, inference latency, and tensor
arena memory usage. Of the techniques evaluated, knowledge distillation into a student architecture
with sufficient capacity produced the most favourable performance results, with the distilled stu
dent achieving 95.96% accuracy at 67ms latency in 66.9KB, a 71% size reduction over the baseline.
Combining distillation with int8 quantisation further reduced the tensor arena footprint to 5.9KB
with 91.73% accuracy and 24ms latency, suitable for extremely memory constrained devices. The
results demonstrate the feasibility for air-writing recognition on MCUs and the importance of model
optimisation choice in TinyML contexts
