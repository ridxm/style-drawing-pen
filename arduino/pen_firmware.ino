// penDNA firmware for Arduino Nano 33 BLE Sense Rev2.
// Streams 4 FSR pressures, 6 IMU channels, and a pen-down button
// over BLE notify at 100 Hz.

#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>

static const char* PEN_SERVICE_UUID = "19b10000-e8f2-537e-4f6c-d104768a1214";
static const char* PEN_CHAR_UUID    = "19b10001-e8f2-537e-4f6c-d104768a1214";

static const int FSR_PINS[4] = {A0, A1, A2, A3};
static const int BUTTON_PIN  = 2;
static const uint32_t SAMPLE_INTERVAL_US = 10000;  // 100 Hz

static const size_t PACKET_SIZE = 4 * sizeof(float)   // pressures
                                + 6 * sizeof(float)   // IMU
                                + 1;                  // pen_down byte

BLEService         penService(PEN_SERVICE_UUID);
BLECharacteristic  penChar(PEN_CHAR_UUID,
                           BLERead | BLENotify,
                           PACKET_SIZE);

uint8_t packet[PACKET_SIZE];
uint32_t nextSampleMicros = 0;

static void writeFloat(uint8_t* dst, float value) {
  memcpy(dst, &value, sizeof(float));
}

void setup() {
  Serial.begin(115200);

  for (int i = 0; i < 4; i++) {
    pinMode(FSR_PINS[i], INPUT);
  }
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  if (!IMU.begin()) {
    Serial.println("IMU init failed");
    while (1) { delay(1000); }
  }

  if (!BLE.begin()) {
    Serial.println("BLE init failed");
    while (1) { delay(1000); }
  }

  BLE.setLocalName("penDNA");
  BLE.setDeviceName("penDNA");
  BLE.setAdvertisedService(penService);
  penService.addCharacteristic(penChar);
  BLE.addService(penService);

  memset(packet, 0, PACKET_SIZE);
  penChar.writeValue(packet, PACKET_SIZE);

  BLE.advertise();
  Serial.println("penDNA advertising");
}

void loop() {
  BLEDevice central = BLE.central();
  if (!central) {
    delay(5);
    return;
  }

  Serial.print("connected: ");
  Serial.println(central.address());
  nextSampleMicros = micros();

  while (central.connected()) {
    uint32_t now = micros();
    if ((int32_t)(now - nextSampleMicros) < 0) {
      continue;
    }
    nextSampleMicros += SAMPLE_INTERVAL_US;

    float pressures[4];
    for (int i = 0; i < 4; i++) {
      int raw = analogRead(FSR_PINS[i]);
      pressures[i] = constrain(raw / 1023.0f, 0.0f, 1.0f);
    }

    float ax = 0, ay = 0, az = 0;
    float gx = 0, gy = 0, gz = 0;
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(ax, ay, az);
    }
    if (IMU.gyroscopeAvailable()) {
      IMU.readGyroscope(gx, gy, gz);
    }

    uint8_t penDown = (digitalRead(BUTTON_PIN) == LOW) ? 1 : 0;

    size_t offset = 0;
    for (int i = 0; i < 4; i++) {
      writeFloat(packet + offset, pressures[i]);
      offset += sizeof(float);
    }
    writeFloat(packet + offset, ax); offset += sizeof(float);
    writeFloat(packet + offset, ay); offset += sizeof(float);
    writeFloat(packet + offset, az); offset += sizeof(float);
    writeFloat(packet + offset, gx); offset += sizeof(float);
    writeFloat(packet + offset, gy); offset += sizeof(float);
    writeFloat(packet + offset, gz); offset += sizeof(float);
    packet[offset] = penDown;

    penChar.writeValue(packet, PACKET_SIZE);
  }

  Serial.println("disconnected");
}
