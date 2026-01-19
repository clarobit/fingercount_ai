import serial
import numpy as np
import cv2

PORT = "/dev/ttyACM0"
BAUD = 115200
# BAUD = 921600

W, H = 320, 240
FRAME_BYTES = W * H * 2  # RGB565 (2 bytes/pixel)

# MCU 쪽에서 프레임 앞에 보내는 헤더(마커)와 동일해야 함
HEADER = b"\xAA\x55\xAA\x55"

ser = serial.Serial(PORT, BAUD, timeout=2)

def read_exact(n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            continue
        buf += chunk
    return bytes(buf)

def sync_to_header() -> None:
    """스트림에서 HEADER(4바이트)가 나올 때까지 동기 맞춤"""
    window = bytearray()
    while True:
        b = ser.read(1)
        if not b:
            continue
        window += b
        if len(window) > len(HEADER):
            window = window[-len(HEADER):]
        if bytes(window) == HEADER:
            return

print("Syncing to frame header...")
sync_to_header()
print("Receiving RGB565 frames... (ESC to quit)")

while True:
    # 매 프레임마다 먼저 헤더로 동기
    sync_to_header()

    # 그 다음 정확히 1프레임 읽기
    raw = read_exact(FRAME_BYTES)
    print(len(raw))

    # RGB565 (little-endian) -> uint16 이미지
    # frame565 = np.frombuffer(raw, dtype="<u2").reshape(H, W)
    frame565 = np.frombuffer(raw, dtype=np.uint16).byteswap().reshape(H, W)

    # RGB565 -> BGR888
    r = ((frame565 >> 11) & 0x1F).astype(np.uint8) << 3
    g = ((frame565 >> 5)  & 0x3F).astype(np.uint8) << 2
    b = ( frame565        & 0x1F).astype(np.uint8) << 3
    frame_bgr = np.dstack((b, g, r))

    cv2.imshow("OV7670 RGB565", frame_bgr)
    if cv2.waitKey(1) == 27:  # ESC
        break

cv2.destroyAllWindows()
ser.close()
