"""Async BLE receiver for penDNA firmware packets."""

from __future__ import annotations

import asyncio
import struct
import threading
import time
from collections import deque
from typing import Deque, Optional

from bleak import BleakClient, BleakScanner

import config

PACKET_FORMAT = "<4f6fB"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

PEN_SERVICE_UUID = "19b10000-e8f2-537e-4f6c-d104768a1214"
PEN_CHAR_UUID = "19b10001-e8f2-537e-4f6c-d104768a1214"


class BLEReceiver:
    def __init__(self, buffer_size: int = 2000):
        self._buffer: Deque[dict] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._client: Optional[BleakClient] = None
        self._char_uuid: Optional[str] = None

    @property
    def connected(self) -> bool:
        return self._client is not None and self._client.is_connected

    def latest(self) -> Optional[dict]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._buffer)

    def _handle_notify(self, _sender, data: bytearray) -> None:
        if len(data) < PACKET_SIZE:
            return
        p1, p2, p3, p4, ax, ay, az, gx, gy, gz, pen_down = struct.unpack(
            PACKET_FORMAT, bytes(data[:PACKET_SIZE])
        )
        sample = {
            "t": time.time(),
            "p1": p1, "p2": p2, "p3": p3, "p4": p4,
            "ax": ax, "ay": ay, "az": az,
            "gx": gx, "gy": gy, "gz": gz,
            "pen_down": bool(pen_down),
        }
        with self._lock:
            self._buffer.append(sample)

    async def _find_device(self, timeout: float = 10.0):
        target = config.BLE_DEVICE_NAME
        devices = await BleakScanner.discover(timeout=timeout)
        for d in devices:
            if d.name and d.name == target:
                return d
        return None

    async def _pick_char_uuid(self, client: BleakClient) -> str:
        for service in client.services:
            for char in service.characteristics:
                if "notify" in char.properties and (
                    char.uuid.lower() == PEN_CHAR_UUID.lower()
                    or service.uuid.lower() == PEN_SERVICE_UUID.lower()
                ):
                    return char.uuid
        for service in client.services:
            for char in service.characteristics:
                if "notify" in char.properties:
                    return char.uuid
        raise RuntimeError("no notify characteristic found on pen")

    async def connect(self, timeout: float = 10.0) -> None:
        device = await self._find_device(timeout=timeout)
        if device is None:
            raise RuntimeError(f"pen '{config.BLE_DEVICE_NAME}' not found")
        client = BleakClient(device)
        await client.connect()
        self._client = client
        self._char_uuid = await self._pick_char_uuid(client)
        await client.start_notify(self._char_uuid, self._handle_notify)

    async def disconnect(self) -> None:
        if self._client and self._client.is_connected:
            try:
                if self._char_uuid:
                    await self._client.stop_notify(self._char_uuid)
            finally:
                await self._client.disconnect()
        self._client = None
        self._char_uuid = None

    async def run(self) -> None:
        await self.connect()
        try:
            while self.connected:
                await asyncio.sleep(0.5)
        finally:
            await self.disconnect()


async def _demo() -> None:
    print("scanning BLE devices...")
    devices = await BleakScanner.discover(timeout=5.0)
    for d in devices:
        print(f"  {d.address}  {d.name}")
    receiver = BLEReceiver()
    print(f"connecting to '{config.BLE_DEVICE_NAME}'...")
    await receiver.connect()
    print("connected. streaming for 10s:")
    start = time.time()
    last_print = 0.0
    while time.time() - start < 10.0:
        await asyncio.sleep(0.05)
        sample = receiver.latest()
        if sample and sample["t"] - last_print > 0.1:
            last_print = sample["t"]
            print(sample)
    await receiver.disconnect()


if __name__ == "__main__":
    asyncio.run(_demo())
