import serial
import numpy as np
import cv2

PORT = "/dev/ttyACM0"
BAUD = 921600

W, H = 320, 240
FRAME_BYTES = W * H * 2
HEADER = b"\xAA\x55\xAA\x55"

ser = serial.Serial(PORT, BAUD, timeout=None)

def read_exact(n):
    buf = bytearray()
    while len(buf) < n:
        buf += ser.read(n - len(buf))
    return bytes(buf)

def sync_to_header():
    win = bytearray()
    while True:
        win += ser.read(1)
        if len(win) > 4:
            win = win[-4:]
        if bytes(win) == HEADER:
            return

print("Syncing...")
sync_to_header()
print("Receiving frames (ESC to quit)")

while True:
    sync_to_header()
    raw = read_exact(FRAME_BYTES)

    # RGB565 (little-endian on wire) → uint16
    frame16 = np.frombuffer(raw, dtype=np.uint16).byteswap().reshape(H, W)

    # RGB565 → BGR888 (OpenCV)
    r = ((frame16 >> 11) & 0x1F).astype(np.uint8) << 3
    g = ((frame16 >> 5)  & 0x3F).astype(np.uint8) << 2
    b = ( frame16        & 0x1F).astype(np.uint8) << 3
    frame_bgr = np.dstack((b, g, r))

    cv2.imshow("OV7670 (manual reg, RGB565)", frame_bgr)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
ser.close()
