import serial

PORT = "/dev/ttyACM0"
# BAUD = 115200
BAUD = 921600


ser = serial.Serial(PORT, BAUD, timeout=1)

print("Listening...")

while True:
    line = ser.readline()
    byte_len = len(line)
    
    if line:
        print(byte_len, "bytes:", end=" ")
        try:
            print(byte_len, "bytes:", end=" ")
            print(line.decode(errors="ignore").strip())
            print(byte_len, "bytes:", end=" ")
        except Exception as e:
            print("decode error:", e)
    
        


