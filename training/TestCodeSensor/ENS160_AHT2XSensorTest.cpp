#include <Wire.h>
#include "ScioSense_ENS160.h"
#include <Adafruit_AHTX0.h>

// Maak sensor objecten aan
ScioSense_ENS160 ens160(ENS160_I2CADDR_1); // I²C adres 0x53 (standaard)
Adafruit_AHTX0 aht;

void setup() {
  Serial.begin(115200);
  delay(2000); // Geef de ESP32 tijd om op te starten
  Serial.println("ENS160 + AHT2X Sensor Test");
  
  Wire.begin(21, 22);
  
  if (!aht.begin()) {
    Serial.println("AHT2X sensor niet gevonden!");
    while (1);
  }
  Serial.println("AHT2X sensor geïnitialiseerd");
  
  if (!ens160.begin()) {
    Serial.println("ENS160 sensor niet gevonden!");
    while (1);
  }
  // BELANGRIJK: Zet ENS160 in standaard modus
  ens160.setMode(ENS160_OPMODE_STD);
  delay(100);
  
  Serial.println("ENS160 sensor geïnitialiseerd");
  Serial.print("ENS160 Rev: ");
  Serial.print(ens160.getMajorRev());
  Serial.print(".");
  Serial.print(ens160.getMinorRev());
  Serial.print(".");
  Serial.println(ens160.getBuild());
  
  // Wacht op eerste meting
  Serial.println("ENS160 wordt opgewarmd...");
  delay(10000);
}

void loop() {
  sensors_event_t humidity, temp;
  aht.getEvent(&humidity, &temp);
  
  ens160.measure(true);
  
  Serial.println("------ Sensor Metingen ------");
  Serial.print("Temperatuur: ");
  Serial.print(temp.temperature);
  Serial.println(" °C");
  
  Serial.print("Vochtigheid: ");
  Serial.print(humidity.relative_humidity);
  Serial.println(" %");
  
  Serial.print("eCO2: ");
  Serial.print(ens160.geteCO2());
  Serial.println(" ppm");
  
  Serial.print("TVOC: ");
  Serial.print(ens160.getTVOC());
  Serial.println(" ppb");
  
  Serial.print("AQI: ");
  Serial.println(ens160.getAQI());
  Serial.println("----------------------------");
  
  delay(2000);
}
