import asyncio
import pyaudio
import cv2  
import os
import time
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

        # Task 1: The Ear (Microphone)
        async def send_audio():
            try:
                while True:
                    data = mic_stream.read(CHUNK, exception_on_overflow=False)
                    await session.send(input={"data": data, "mime_type": f"audio/pcm;rate={RATE_IN}"})
                    await asyncio.sleep(0.001)
            except Exception as e:
                print(f"Mic error: {e}")

        # Task 2: The Mouth (Speakers AND Tool Handler)
        async def receive_audio():
            try:
                while True: 
                    async for response in session.receive():
                        server_content = response.server_content
                        if server_content is None:
                            continue

                        # A. Handle Voice Response
                        if server_content.model_turn:
                            for part in server_content.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    speaker_stream.write(part.inline_data.data)

                        # B. Handle Tool Calls
                        if server_content.tool_call:
                            print("\n[🤖 Sarvagya is thinking and using a tool...]")
                            for call in server_content.tool_call.function_calls:
                                name = call.name
                                args = call.args
                                
                                # Execute the Python function safely
                                if name in sarvagya_tools.tool_registry:
                                    result = sarvagya_tools.tool_registry[name](**args)
                                    
                                    # Send the result back to Gemini so it knows the tool finished
                                    await session.send(
                                        input=types.LiveClientToolResponse(
                                            function_responses=[
                                                types.FunctionResponse(
                                                    name=name,
                                                    id=call.id,
                                                    response={"result": result}
                                                )
                                            ]
                                        )
                                    )
            except Exception as e:
                print(f"Speaker/Tool error: {e}")

        # Task 3: The Third Eye (Webcam)
        # Task 3: The Third Eye (Webcam) - Smooth Version
        async def send_video():
            cap = cv2.VideoCapture(0)
            try:
                last_sent = 0
                while True:
                    # Read the camera as fast as possible to clear the buffer
                    ret, frame = cap.read()
                    if not ret:
                        await asyncio.sleep(0.01)
                        continue
                    
                    # Display the smooth, real-time feed on your monitor
                    cv2.imshow("Sarvagya: Third Eye View", frame)
                    cv2.waitKey(1)
                    
                    # Only transmit to Gemini once every 1 second
                    now = time.time()
                    if now - last_sent >= 1.0:
                        frame_resized = cv2.resize(frame, (768, 768)) 
                        _, buffer = cv2.imencode('.jpg', frame_resized)
                        
                        await session.send_realtime_input(
                            media=types.Blob(data=buffer.tobytes(), mime_type="image/jpeg") 
                        )
                        last_sent = now
                    
                    # Yield control to the audio tasks without lagging the camera
                    await asyncio.sleep(0.01) 
            except Exception as e:
                print(f"Camera error: {e}")
            finally:
                cap.release()
                cv2.destroyAllWindows()

        # Run all three tasks simultaneously
        await asyncio.gather(send_audio(), receive_audio(), send_video())

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