import time
import board
import busio

import adafruit_ens160
import adafruit_ahtx0

# I2C-bus op Raspberry Pi (SDA=GPIO2, SCL=GPIO3)
i2c = busio.I2C(board.SCL, board.SDA)

# AHT21 (temp + vocht) op 0x38
aht = adafruit_ahtx0.AHTx0(i2c)

# ENS160 (luchtkwaliteit) op 0x53 (default)
ens = adafruit_ens160.ENS160(i2c)

# Init: zorg dat ENS160 omgevingstemperatuur/vocht kent
# zodat hij betere eCO2/TVOC berekent.
def update_environment():
    temperature = aht.temperature      # °C
    humidity = aht.relative_humidity  # %
    ens.temperature_compensation = temperature
    ens.humidity_compensation = humidity
    return temperature, humidity

print("ENS160 + AHT21 test gestart...")
while True:
    # update temp/vocht naar ENS160
    temp, rh = update_environment()

    # ENS160-waarden uitlezen
    aq = ens.AQI              # IAQ index (0–3 of 1–5 afhankelijk van lib)
    tvoc = ens.TVOC                  # ppb
    eco2 = ens.ECO2                  # ppm

    print(f"Temp: {temp:.1f} °C  Humidity: {rh:.1f} %")
    print(f"AirQuality index: {aq}, TVOC: {tvoc} ppb, eCO2: {eco2} ppm")
    print("-" * 40)
    time.sleep(2)
