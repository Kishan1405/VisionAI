# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Comment by Bhavya

"""
## Setup

To install the dependencies for this script, run:

``` 
pip install google-genai opencv-python pyaudio pillow mss
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones. 

## Run

To run the script:

```
python Get_started_LiveAPI.py
```

The script takes a video-mode flag `--mode`, this can be "camera", "screen", or "none".
The default is "camera". To share your screen run:

```
python Get_started_LiveAPI.py --mode screen
```
"""

import asyncio
import base64
import io
import os
import sys
import traceback

import cv2
import pyaudio
import PIL.Image
import mss

import argparse
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

from google import genai


if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

SEND_SAMPLE_RATE = 24000  # was 16000
RECEIVE_SAMPLE_RATE = 24000  # was 24000, keep or increase to 48000 if supported
CHUNK_SIZE = 2048  # was 1024
FORMAT = pyaudio.paInt16  # paInt24 if supported
MODEL = "models/gemini-2.0-flash-live-001"

DEFAULT_MODE = "camera"


def prompt_api_key():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    api_key = simpledialog.askstring("Gemini API Key", "Enter your Gemini API Key:", show='*')
    root.destroy()
    if not api_key:
        messagebox.showerror("Error", "API Key is required to continue.")
        sys.exit(1)
    return api_key


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE, api_key=None):
        self.video_mode = video_mode
        self.api_key = api_key
        self.client = genai.Client(
            api_key=self.api_key,
            http_options={"api_version": "v1beta"}
        )

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        def open_camera():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return None
            return cap

        cap = await asyncio.to_thread(open_camera)
        if cap is None:
            print("Error: Could not open camera.")
            return

        try:
            while True:
                frame = await asyncio.to_thread(self._get_frame, cap)
                if frame is None:
                    print("Warning: Failed to read frame from camera.")
                    break

                await asyncio.sleep(1.0)
                await self.out_queue.put(frame)
        finally:
            cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with (
                self.client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)


def start_audio_loop(selected_mode, api_key):
    def run_loop():
        main = AudioLoop(video_mode=selected_mode, api_key=api_key)
        asyncio.run(main.run())
    threading.Thread(target=run_loop, daemon=True).start()

def launch_ui():
    # Removed prompt_api_key() usage
    root = tk.Tk()
    root.title("VisionAI Gemini Live")

    tk.Label(root, text="Select Video Mode:").pack(pady=10)
    mode_var = tk.StringVar(value=DEFAULT_MODE)
    mode_combo = ttk.Combobox(root, textvariable=mode_var, values=["camera", "screen", "none"], state="readonly")
    mode_combo.pack(pady=5)

    def on_start():
        messagebox.showinfo("Starting", f"Starting in {mode_var.get()} mode.\nClose this window to exit.")
        start_audio_loop(mode_var.get(), None)  # No API key

    start_btn = tk.Button(root, text="Start", command=on_start)
    start_btn.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    # Removed --api_key argument
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode, api_key=None)
    asyncio.run(main.run())