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

#loading the vault grabing the key 
load_dotenv()
my_key = os.getenv("GEMINI_API_KEY")

if not my_key:
    print("❌ ERROR: Could not find GEMINI_API_KEY! Check your .env file.")
    exit()

client = genai.Client(api_key=my_key)

# 1. Audio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE_IN = 16000  
RATE_OUT = 24000 
CHUNK = 512

async def audio_video_loop():
    p = pyaudio.PyAudio()

    # Open Audio Streams
    mic_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE_IN, input=True, frames_per_buffer=CHUNK)
    speaker_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE_OUT, output=True)

    print("Sarvagya is initializing systems...")

    # 2. Define Tools for the AI
    tools_config = [
        sarvagya_tools.update_todo,
        sarvagya_tools.create_file,
        types.Tool(google_search=types.GoogleSearch())  
    ]

    # Set up the Agent's Identity
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        tools=tools_config, # <-- Injected the tools here
        system_instruction=types.Content(
            parts=[types.Part.from_text(
                text=(
                    "You are Sarvagya, an agentic engineering mentor. "
                    "You have three tools: 'search_web' to find info, "
                    "'update_todo' to track project tasks, and 'create_file' to write code to the user's computer. "
                    "You can also see through the user's webcam. "
                    "Keep responses brief and technical. "
                    "If the user asks you to write code, DO NOT read the code out loud. "
                    "Use the 'create_file' tool to save it directly to their machine, then tell them it is ready."
                )
            )]
        )
    )

    async with client.aio.live.connect(model="gemini-2.5-flash-native-audio-preview-12-2025", config=config) as session:
        print("\n[🎙️👁️ Sarvagya is Online with Vision and Tools. Start speaking...]")

        # Task 1: The Ears (Microphone) - UPDATED FOR NEW SDK
        async def send_audio():
            while True:
                try:
                    data = mic_stream.read(CHUNK, exception_on_overflow=False)
                    # The new SDK requires the strictly typed 'media=types.Blob' format
                    await session.send_realtime_input(
                        media=types.Blob(
                            data=data, 
                            mime_type=f"audio/pcm;rate={RATE_IN}"
                        )
                    )
                except Exception as e:
                    print(f"Mic error: {e}")
                await asyncio.sleep(0.001)

        # Task 2: The Mouth & Brain (Speaker and Tool Handler) - UPDATED FOR NEW SDK
        async def receive_audio():
            try:
                # The new SDK yields 'chunk' objects, not raw dictionaries
                async for chunk in session.receive():
                    
                    # 1. Play spoken audio from the AI
                    if chunk.server_content and chunk.server_content.model_turn:
                        for part in chunk.server_content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                audio_stream.write(part.inline_data.data)

                    # 2. Handle Custom Python Tools (Auto-Coder & Task Tracker)
                    if chunk.tool_call:
                        for fc in chunk.tool_call.function_calls:
                            # Note: Google Search is handled internally by the API! 
                            # We only need to manually run our custom Python tools.
                            if fc.name in sarvagya_tools.tool_registry:
                                print(f"\n[🤖 Sarvagya is using tool: {fc.name}]")
                                
                                # Run the tool
                                result = sarvagya_tools.tool_registry[fc.name](**fc.args)
                                
                                # Send the result back to Gemini so it can read it
                                await session.send_tool_response(
                                    function_responses=[
                                        types.FunctionResponse(
                                            id=fc.id,
                                            name=fc.name,
                                            response={"result": result}
                                        )
                                    ]
                                )
            except Exception as e:
                print(f"Receive error: {e}")

        # Task 3: The Third Eye (Webcam)
        # Task 3: The Third Eye (Webcam) - Smooth Version
        # Task 3: The All-Seeing Eye (Screen Capture)
        async def send_screen():
            with mss.mss() as sct:
                # '1' targets your primary monitor (0 is all monitors combined)
                monitor = sct.monitors[1] 
                try:
                    last_sent = 0
                    while True:
                        now = time.time()
                        
                        # Only transmit to Gemini once every 1 second
                        if now - last_sent >= 1.0:
                            # 1. Grab the screen
                            screenshot = sct.grab(monitor)
                            
                            # 2. Convert raw pixels to an OpenCV image
                            frame = np.array(screenshot)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                            
                            # 3. Resize for Gemini's optimal vision resolution
                            frame_resized = cv2.resize(frame, (768, 768)) 
                            _, buffer = cv2.imencode('.jpg', frame_resized)
                            
                            # 4. Stream it to the AI
                            await session.send_realtime_input(
                                media=types.Blob(data=buffer.tobytes(), mime_type="image/jpeg") 
                            )
                            last_sent = now
                        
                        # Keep the async loop breathing
                        await asyncio.sleep(0.01) 
                except Exception as e:
                    print(f"Screen capture error: {e}")

        # Run all three tasks simultaneously
        await asyncio.gather(send_audio(), receive_audio(), send_screen())

    # Graceful shutdown
    mic_stream.stop_stream()
    mic_stream.close()
    speaker_stream.stop_stream()
    speaker_stream.close()
    p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(audio_video_loop())
    except KeyboardInterrupt:
        print("\nShutting down Sarvagya safely...")