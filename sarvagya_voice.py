import asyncio
import pyaudio
import cv2  
import os
import time
import numpy as np
import mss
import sarvagya_tools 
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ==========================================
# ⚙️ SARVAGYA CONFIGURATION & FEATURE FLAGS
# ==========================================
ENABLE_SCREEN_CAPTURE = False  # Set to True when you are ready to use the All-Seeing Eye (Requires Xorg)

# Audio Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE_IN = 16000  
RATE_OUT = 24000 
CHUNK = 512

# ==========================================
# 🔐 INITIALIZATION
# ==========================================
load_dotenv()
my_key = os.getenv("GEMINI_API_KEY")

if not my_key:
    print("❌ ERROR: Could not find GEMINI_API_KEY! Check your .env file.")
    exit()

client = genai.Client(api_key=my_key)

# ==========================================
# 🧠 MAIN SYSTEM LOOP
# ==========================================
async def audio_video_loop():
    p = pyaudio.PyAudio()

    # Hardware Streams
    mic_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE_IN, input=True, frames_per_buffer=CHUNK)
    audio_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE_OUT, output=True)

    print("\n[ Sarvagya System Initializing... ]")
    if ENABLE_SCREEN_CAPTURE:
        print("[ 👀 Screen Capture: ONLINE ]")
    else:
        print("[ 👁️ Screen Capture: OFFLINE ]")

    # The Tools Sarvagya can use
    tools_config = [
        sarvagya_tools.update_todo,
        sarvagya_tools.create_file,
        types.Tool(google_search=types.GoogleSearch())  
    ]

    # The Master Persona
    master_prompt = (
        "You are Sarvagya, an agentic engineering mentor. "
        "You have tools: 'update_todo' to track project tasks, and 'create_file' to write code to the user's computer. "
        "Always use the native Google Search tool for live information like the weather. "
        "Never attempt to use a tool named 'search_web'. "
        "Keep responses brief, professional, and technical. "
        "If the user asks you to write code, DO NOT read the code out loud. "
        "Use the 'create_file' tool to save it directly to their machine, then tell them it is ready."
    )

    # 🌐 Connect to the Neural Network
    async with client.aio.live.connect(
        model="gemini-2.0-flash-exp", 
        config=types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            tools=tools_config,
            system_instruction=types.Content(parts=[
                types.Part.from_text(text=master_prompt)
            ])
        )
    ) as session:
        print("\n[ 🟢 Sarvagya is Listening. Say hello! ]\n")

        # ------------------------------------------
        # 🎙️ TASK 1: THE EARS (Non-Blocking Mic)
        # ------------------------------------------
        async def send_audio():
            while True:
                try:
                    data = await asyncio.to_thread(mic_stream.read, CHUNK, exception_on_overflow=False)
                    await session.send_realtime_input(
                        media=types.Blob(data=data, mime_type=f"audio/pcm;rate={RATE_IN}")
                    )
                except Exception:
                    pass 
                await asyncio.sleep(0.001)

        # ------------------------------------------
        # 🔊 TASK 2: THE MOUTH & BRAIN (UI & Tools)
        # ------------------------------------------
        async def receive_audio():
            try:
                async for chunk in session.receive():
                    
                    # Audio Playback
                    if chunk.server_content and chunk.server_content.model_turn:
                        for part in chunk.server_content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                await asyncio.to_thread(audio_stream.write, part.inline_data.data)

                    # Tool Execution
                    if chunk.tool_call:
                        for fc in chunk.tool_call.function_calls:
                            print(f"\n⟡ Sarvagya is utilizing system tool: {fc.name} ...")
                            
                            if fc.name in sarvagya_tools.tool_registry:
                                result = await asyncio.to_thread(sarvagya_tools.tool_registry[fc.name], **fc.args)
                            else:
                                result = "Error: Tool not found. Please answer using your own knowledge or native Google Search."
                            
                            await session.send_tool_response(
                                function_responses=[
                                    types.FunctionResponse(
                                        id=fc.id,
                                        name=fc.name,
                                        response={"result": str(result)}
                                    )
                                ]
                            )
            except Exception as e:
                print(f"\n[ Receive error: {e} ]")

        # ------------------------------------------
        # 👀 TASK 3: THE ALL-SEEING EYE (Screen)
        # ------------------------------------------
        async def send_screen():
            with mss.mss() as sct:
                monitor = sct.monitors[1] 
                try:
                    last_sent = 0
                    while True:
                        now = time.time()
                        if now - last_sent >= 1.0:
                            screenshot = sct.grab(monitor)
                            frame = np.array(screenshot)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                            frame_resized = cv2.resize(frame, (768, 768)) 
                            _, buffer = cv2.imencode('.jpg', frame_resized)
                            
                            await session.send_realtime_input(
                                media=types.Blob(data=buffer.tobytes(), mime_type="image/jpeg") 
                            )
                            last_sent = now
                        await asyncio.sleep(0.01) 
                except Exception as e:
                    print(f"[ Screen capture error: {e} ]")

        # ------------------------------------------
        # 🚀 SYSTEM LAUNCHER
        # ------------------------------------------
        # Dynamically build the task list based on feature flags
        active_tasks = [send_audio(), receive_audio()]
        if ENABLE_SCREEN_CAPTURE:
            active_tasks.append(send_screen())

        await asyncio.gather(*active_tasks)

    # Teardown Sequence
    mic_stream.stop_stream()
    mic_stream.close()
    audio_stream.stop_stream()
    audio_stream.close()
    p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(audio_video_loop())
    except KeyboardInterrupt:
        print("\n[ Shutting down Sarvagya safely... ]")