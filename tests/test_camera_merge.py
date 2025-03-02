import asyncio
from typing import Dict
import numpy as np
import pytest

import ironic as ir
from drivers.camera.merge import merge_on_pulse, merge_on_camera


@ir.ironic_system(output_ports=['frame'])
class MockCamera(ir.ControlSystem):
    """Mock camera that emits frames on demand"""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self._frame_queue = asyncio.Queue()

    async def emit_frame(self, frame: Dict[str, np.ndarray], timestamp: int = None):
        """Send a new frame through the output port"""
        await self.outs.frame.write(ir.Message(frame, timestamp))

    async def step(self):
        return ir.State.ALIVE


@pytest.mark.asyncio
async def test_merge_on_pulse():
    """Test merging cameras triggered by external pulse"""
    # Create mock cameras
    cam1 = MockCamera("cam1")
    cam2 = MockCamera("cam2")
    cameras = {"cam1": cam1, "cam2": cam2}

    # Create merged camera system using fps parameter
    merged = merge_on_pulse(cameras, fps=10)
    await merged.setup()

    # Create frame receiver
    received_frames = []

    async def frame_handler(message):
        received_frames.append(message.data)

    merged.outs.frame.subscribe(frame_handler)

    # Run system for a few steps to allow pulse generation
    for _ in range(5):
        await merged.step()
        await asyncio.sleep(0.1)  # Give time for pulse to trigger

    # Emit test frames from cameras
    frame1 = {"image": np.zeros((10, 10, 3))}
    frame2 = {"image": np.ones((10, 10, 3))}

    await cam1.emit_frame(frame1)
    await cam2.emit_frame(frame2)

    # Run a few more steps to allow processing
    for _ in range(3):
        await merged.step()
        await asyncio.sleep(0.1)

    assert len(received_frames) >= 1
    merged_frame = received_frames[0]
    assert "cam1.image" in merged_frame
    assert "cam2.image" in merged_frame
    np.testing.assert_array_equal(merged_frame["cam1.image"], frame1["image"])
    np.testing.assert_array_equal(merged_frame["cam2.image"], frame2["image"])


@pytest.mark.asyncio
async def test_merge_on_camera():
    """Test merging cameras triggered by main camera"""
    # Create mock cameras
    main_cam = MockCamera("main")
    ext_cam = MockCamera("ext")
    extension_cameras = {"ext": ext_cam}

    # Create merged camera system using main camera as trigger
    merged = merge_on_camera(("main", main_cam), extension_cameras)
    await merged.setup()

    # Create frame receiver
    received_frames = []

    async def frame_handler(message):
        received_frames.append(message.data)

    merged.outs.frame.subscribe(frame_handler)

    # Emit test frames
    main_frame = {"image": np.zeros((10, 10, 3))}
    ext_frame = {"image": np.ones((10, 10, 3))}

    # First emit frame from extension camera
    await ext_cam.emit_frame(ext_frame)
    await asyncio.sleep(0)  # Let event loop process messages
    assert len(received_frames) == 0  # Should not emit until main camera sends frame

    # Now emit frame from main camera
    await main_cam.emit_frame(main_frame)
    await asyncio.sleep(0)

    assert len(received_frames) == 1
    merged_frame = received_frames[0]
    assert "main.image" in merged_frame
    assert "ext.image" in merged_frame
    np.testing.assert_array_equal(merged_frame["main.image"], main_frame["image"])
    np.testing.assert_array_equal(merged_frame["ext.image"], ext_frame["image"])


@pytest.mark.asyncio
async def test_merge_missing_frame():
    """Test merging cameras when extension camera hasn't sent a frame yet"""
    main_cam = MockCamera("main")
    ext_cam = MockCamera("ext")
    extension_cameras = {"ext": ext_cam}

    merged = merge_on_camera(("main", main_cam), extension_cameras)
    await merged.setup()

    received_frames = []

    async def frame_handler(message):
        received_frames.append(message.data)

    merged.outs.frame.subscribe(frame_handler)

    # Only emit frame from main camera
    main_frame = {"image": np.zeros((10, 10, 3))}
    await main_cam.emit_frame(main_frame)
    await asyncio.sleep(0)

    # Should not emit anything when an extension camera frame is missing
    assert len(received_frames) == 0


@pytest.mark.asyncio
async def test_merge_on_camera_duplicate_name():
    """Test that using same name for main and extension camera raises error"""
    main_cam = MockCamera("cam1")
    ext_cam = MockCamera("cam1")
    extension_cameras = {"cam1": ext_cam}

    with pytest.raises(AssertionError):
        merge_on_camera(("cam1", main_cam), extension_cameras)


@pytest.mark.asyncio
async def test_merge_one_novalue_input_returns_novalue():
    """Test that merge returns NoValue if any camera returns NoValue"""
    # Create mock cameras
    cam1 = MockCamera("cam1")
    cam2 = MockCamera("cam2")
    cameras = {"cam1": cam1, "cam2": cam2}

    # Create merged camera system
    merged = merge_on_pulse(cameras, fps=20)
    await merged.setup()

    # Create frame receiver
    received_frames = []

    async def frame_handler(message):
        received_frames.append(message.data)

    merged.outs.frame.subscribe(frame_handler)

    # Emit a normal frame from cam1 and NoValue from cam2
    frame1 = {"image": np.zeros((10, 10, 3))}
    await cam1.emit_frame(frame1)
    await cam2.emit_frame(ir.NoValue)

    # Run for enough time to ensure multiple pulses
    for _ in range(3):
        await merged.step()
        await asyncio.sleep(0.1)  # Wait longer than pulse period (0.05s)

    # Should not have received any frames since one camera returned NoValue
    assert len(received_frames) == 0

    # Now emit valid frames from both cameras
    frame2 = {"image": np.ones((10, 10, 3))}
    await cam1.emit_frame(frame1)
    await cam2.emit_frame(frame2)

    # Run for enough time to get at least one frame
    received_frames.clear()
    for _ in range(3):
        await merged.step()
        await asyncio.sleep(0.1)

    # Should have received at least one merged frame
    assert len(received_frames) >= 1
    merged_frame = received_frames[0]  # Check first frame
    assert "cam1.image" in merged_frame
    assert "cam2.image" in merged_frame
