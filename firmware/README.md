# Firmware

`firmware/` holds optional hardware-related projects for Aeon-V1.

Current firmware project:

```text
esp32s3-auth-device/
```

Read [esp32s3-auth-device/README.md](esp32s3-auth-device/README.md) for build and usage details.

## Role In Aeon

The ESP32-S3 device is an optional hardware approval token for governed writes. It is not required for normal memory, chat, dashboard, Obsidian, or LM Studio use.

The Python integration point is:

```text
src/aeon_v1/hardware_auth_provider.py
```

## Safety Note

Hardware approval is intended to strengthen the human gate around agent-proposed writes. It does not make Aeon an autonomous executor, and it should not bypass validation or approval logic in the Python package.
