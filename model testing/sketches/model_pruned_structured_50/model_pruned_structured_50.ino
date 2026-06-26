// TFLite Micro evaluation sketch
// Auto-detects float32 vs int8 input/output tensors.
// For int8: Python sends pre-quantized integers.
// For float32: Python sends float values.

#include <Chirale_TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_data.h"

static const int SEQUENCE_LENGTH = 40;
static const int NUM_FEATURES    = 6;
static const int NUM_CLASSES     = 26;

static const int ARENA_SIZE = 96000;
alignas(16) static uint8_t tensorArena[ARENA_SIZE];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* inputTensor  = nullptr;
TfLiteTensor* outputTensor = nullptr;

static float inputBufferFloat[SEQUENCE_LENGTH][NUM_FEATURES];
static int8_t inputBufferInt8[SEQUENCE_LENGTH][NUM_FEATURES];
static int timestepCount = 0;

bool inputIsInt8 = false;
bool outputIsInt8 = false;


void setup() {
    Serial.begin(115200);
    while (!Serial);

    model = tflite::GetModel(MODEL_DATA);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR:MODEL_VERSION_MISMATCH");
        while (true);
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter staticInterpreter(
        model, resolver, tensorArena, ARENA_SIZE
    );
    interpreter = &staticInterpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR:ALLOC_TENSORS_FAILED");
        while (true);
    }

    inputTensor  = interpreter->input(0);
    outputTensor = interpreter->output(0);

    inputIsInt8 = (inputTensor->type == kTfLiteInt8);
    outputIsInt8 = (outputTensor->type == kTfLiteInt8);

    // Report tensor info so Python knows how to send data
    Serial.print("DEBUG:INPUT_TYPE=");
    Serial.println(inputTensor->type);
    if (inputIsInt8) {
        Serial.print("DEBUG:INPUT_SCALE=");
        Serial.println(inputTensor->params.scale, 8);
        Serial.print("DEBUG:INPUT_ZP=");
        Serial.println(inputTensor->params.zero_point);
    }
    Serial.print("DEBUG:OUTPUT_TYPE=");
    Serial.println(outputTensor->type);

    size_t arenaUsed = interpreter->arena_used_bytes();
    Serial.print("READY:");
    Serial.println(arenaUsed);
}


void loop() {
    if (!Serial.available()) return;

    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    if (line == "PING") { Serial.println("PONG"); return; }
    if (line == "DONE") { Serial.println("BYE");  return; }

    if (line == "START") {
        timestepCount = 0;
        Serial.println("OK");
        return;
    }

    if (line == "RUN") {
        if (timestepCount != SEQUENCE_LENGTH) {
            Serial.print("ERROR:EXPECTED_40_GOT_");
            Serial.println(timestepCount);
            timestepCount = 0;
            return;
        }

        if (inputIsInt8) {
            for (int t = 0; t < SEQUENCE_LENGTH; t++)
                for (int f = 0; f < NUM_FEATURES; f++)
                    inputTensor->data.int8[t * NUM_FEATURES + f] = inputBufferInt8[t][f];
        } else {
            for (int t = 0; t < SEQUENCE_LENGTH; t++)
                for (int f = 0; f < NUM_FEATURES; f++)
                    inputTensor->data.f[t * NUM_FEATURES + f] = inputBufferFloat[t][f];
        }

        unsigned long startUs = micros();
        TfLiteStatus status = interpreter->Invoke();
        unsigned long durationUs = micros() - startUs;
        timestepCount = 0;

        if (status != kTfLiteOk) {
            Serial.println("ERROR:INVOKE_FAILED");
            return;
        }

        int predictedClass = 0;
        if (outputIsInt8) {
            int8_t maxRaw = outputTensor->data.int8[0];
            for (int c = 1; c < NUM_CLASSES; c++) {
                if (outputTensor->data.int8[c] > maxRaw) {
                    maxRaw = outputTensor->data.int8[c];
                    predictedClass = c;
                }
            }
        } else {
            float maxVal = outputTensor->data.f[0];
            for (int c = 1; c < NUM_CLASSES; c++) {
                if (outputTensor->data.f[c] > maxVal) {
                    maxVal = outputTensor->data.f[c];
                    predictedClass = c;
                }
            }
        }

        Serial.print("RESULT:");
        Serial.print(predictedClass);
        Serial.print(",");
        Serial.println(durationUs);
        return;
    }

    // Data line: one timestep
    if (timestepCount >= SEQUENCE_LENGTH) {
        Serial.println("ERROR:BUFFER_FULL");
        return;
    }

    int idx = 0;
    int startPos = 0;
    if (inputIsInt8) {
        for (int i = 0; i <= (int)line.length() && idx < NUM_FEATURES; i++) {
            if (i == (int)line.length() || line.charAt(i) == ',') {
                int val = line.substring(startPos, i).toInt();
                if (val < -128) val = -128;
                if (val > 127) val = 127;
                inputBufferInt8[timestepCount][idx] = (int8_t)val;
                idx++;
                startPos = i + 1;
            }
        }
    } else {
        for (int i = 0; i <= (int)line.length() && idx < NUM_FEATURES; i++) {
            if (i == (int)line.length() || line.charAt(i) == ',') {
                inputBufferFloat[timestepCount][idx] = line.substring(startPos, i).toFloat();
                idx++;
                startPos = i + 1;
            }
        }
    }

    if (idx != NUM_FEATURES) {
        Serial.print("ERROR:EXPECTED_6_GOT_");
        Serial.println(idx);
        return;
    }

    timestepCount++;
    Serial.println("OK");
}
