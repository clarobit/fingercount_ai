import serial
import numpy as np
import cv2

PORT = "/dev/ttyACM0"
BAUD = 921600

W, H = 320, 240
FRAME_BYTES = W * H * 2  # RGB565

ser = serial.Serial(PORT, BAUD, timeout=None)  # 블로킹으로

def read_exact(n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            continue
        buf += chunk
    return bytes(buf)

print("Receiving raw RGB565 frames (no header)...")

while True:
    raw = read_exact(FRAME_BYTES)
    print("bytes:", len(raw))

    # frame565 = np.frombuffer(raw, dtype="<u2").reshape(H, W)
    frame565 = np.frombuffer(raw, dtype=np.uint16).byteswap().reshape(H, W)

    r = ((frame565 >> 11) & 0x1F).astype(np.uint8) << 3
    g = ((frame565 >> 5)  & 0x3F).astype(np.uint8) << 2
    b = ( frame565        & 0x1F).astype(np.uint8) << 3
    frame_bgr = np.dstack((b, g, r))

    cv2.imshow("OV7670 RAW (no header)", frame_bgr)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
ser.close()
