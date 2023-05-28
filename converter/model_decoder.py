from enum import IntEnum
from struct import unpack
from typing import BinaryIO, Dict, List


class DataType(IntEnum):
    FALSE = 0
    TRUE = 1
    UINT8 = 2
    UINT16 = 3
    UINT32 = 4
    NUINT8 = 5
    NUINT16 = 6
    NUINT32 = 7
    FLOAT32 = 8
    STRING = 9
    UINT16_ARRAY = 10
    UINT32_ARRAY = 11
    FLOAT32_ARRAY = 12
    OBJECT = 13
    ARRAY = 14


def parse_model(f: BinaryIO) -> Dict[str, any]:
    decoder = ModelDecoder(f)
    return decoder.parse()


class ModelDecoder:
    stream: BinaryIO

    def __init__(self, f: BinaryIO) -> None:
        self.stream = f

    def parse(self):
        return self.parse_block()

    def parse_object(self) -> Dict[str, any]:
        return {
            self.parse_block(): self.parse_block() for _ in range(self.get_uint16())
        }

    def get_uint8(self) -> int:
        return unpack("<B", self.stream.read(1))[0]

    def get_uint16(self) -> int:
        return unpack("<H", self.stream.read(2))[0]

    def get_uint32(self) -> int:
        return unpack("<I", self.stream.read(4))[0]

    def get_float32(self) -> float:
        return unpack("<f", self.stream.read(4))[0]

    def parse_string(self) -> str:
        length = self.get_uint32()
        return self.stream.read(length).decode("utf-8")

    def get_uint16_array(self) -> List[int]:
        length = self.get_uint32()
        return memoryview(self.stream.read(length * 2)).cast("H")

    def get_uint32_array(self) -> List[int]:
        length = self.get_uint32()
        return memoryview(self.stream.read(length * 4)).cast("I")

    def get_float32_array(self) -> List[float]:
        length = self.get_uint32()
        return memoryview(self.stream.read(length * 4)).cast("f")

    def parse_array(self) -> List[any]:
        length = self.get_uint16()
        return [self.parse_block() for _ in range(length)]

    def parse_block(self, offset=0) -> any:
        if offset:
            self.stream.read(offset)
        typ = self.get_uint8()
        ret = None

        if typ == DataType.FALSE:
            ret = False
        elif typ == DataType.TRUE:
            ret = True
        elif typ == DataType.UINT8:
            ret = self.get_uint8()
        elif typ == DataType.UINT16:
            ret = self.get_uint16()
        elif typ == DataType.UINT32:
            ret = self.get_uint32()
        elif typ == DataType.NUINT8:
            ret = -self.get_uint8()
        elif typ == DataType.NUINT16:
            ret = -self.get_uint16()
        elif typ == DataType.NUINT32:
            ret = -self.get_uint32()
        elif typ == DataType.FLOAT32:
            ret = self.get_float32()
        elif typ == DataType.STRING:
            ret = self.parse_string()
        elif typ == DataType.UINT16_ARRAY:
            ret = self.get_uint16_array()
        elif typ == DataType.UINT32_ARRAY:
            ret = self.get_uint32_array()
        elif typ == DataType.FLOAT32_ARRAY:
            ret = self.get_float32_array()
        elif typ == DataType.OBJECT:
            ret = self.parse_object()
        elif typ == DataType.ARRAY:
            ret = self.parse_array()

        return ret
