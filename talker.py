"""
Trans Rights are Human Rights

Short script to handle serial data over USB
"""
# SYSTEM IMPORTS
import serial

# STANDARD LIBRARY IMPORTS

# LOCAL APPLICATION IMPORTS


class Talker:
    TERMINATOR = '\r'.encode('UTF8')

    def __init__(self, com_port: str = "COM6", timeout=1):
        self.serial = serial.Serial(com_port, 115200, timeout=timeout)

    def send(self, text: str) -> None:
        line = '%s\r\f' % text
        self.serial.write(line.encode('utf-8'))
        reply = self.receive()
        reply = reply.replace('>>> ', '')  # lines after first will be prefixed by a prompt
        if reply != text:  # the line should be echoed, so the result should match
            raise ValueError(f"expected {text} got {reply}")

        return None

    def receive(self) -> str:
        line = self.serial.read_until(self.TERMINATOR)
        decoded_line = line.decode("UTF8").strip()

        return decoded_line

    def close(self) -> None:
        self.serial.close()

        return None
