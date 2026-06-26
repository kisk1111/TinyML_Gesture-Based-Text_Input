#include <Arduino_LSM9DS1.h>

const int SAMPLING_RATE = 100; 
const int SAMPLE_DELAY_US = 1000000 / SAMPLING_RATE; 
const int TOTAL_SAMPLES = 40; // change from 40 to 200 for slow gestures

void setup() {
  Serial.begin(115200);
  while (!Serial);
  if (!IMU.begin()) {
    while (1); 
  }
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    if (command == 'R') {
      int samplesRead = 0;
      unsigned long lastSampleTime = micros();
      unsigned long gestureStartTime = micros();

      // loop only for number of samples
      while (samplesRead < TOTAL_SAMPLES) {
        unsigned long currentTime = micros();
        
        if (currentTime - lastSampleTime >= SAMPLE_DELAY_US) {
          if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
            float ax, ay, az, gx, gy, gz;
            IMU.readAcceleration(ax, ay, az);
            IMU.readGyroscope(gx, gy, gz);
            
            Serial.print(ax * 9.80665, 4); Serial.print(",");
            Serial.print(ay * 9.80665, 4); Serial.print(",");
            Serial.print(az * 9.80665, 4); Serial.print(",");
            Serial.print(gx, 4);           Serial.print(",");
            Serial.print(gy, 4);           Serial.print(",");
            Serial.print(gz, 4);           Serial.print(",");
            Serial.println(currentTime - gestureStartTime);

            samplesRead++;
            lastSampleTime = currentTime;
          }
        }
      }
      Serial.println("END_OF_GESTURE");
    }
  }
}