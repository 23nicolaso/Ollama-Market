import soundfile as sf
import sounddevice as sd
from kokoro_onnx import Kokoro
import re
from threading import Thread, Lock
from market_simulator.utils.market_utils import invoke_model
import asyncio

kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
isTalking = False
talking_lock = Lock()

def tts_this(text, sentiment, importance):
    if sentiment >= 0.9:
        sentiment_str = "SOARING"
    elif sentiment > 0.5:
        sentiment_str = "up slightly"
    elif sentiment <= 0.2:
        sentiment_str = "selling off harshly"
    else:
        sentiment_str = "down slightly"
    text = f'{text}. This news is {"highly" if importance >= 9 else "not that"} important. Markets are {sentiment_str} after the news' 
    Thread(target=run_tts_queue, args=(text,)).start()

def run_tts_queue(text):
    asyncio.run(tts_queue(text))

async def tts_queue(text):
    global isTalking
    with talking_lock:
        if isTalking:
            return
        isTalking = True

    try:
        # This might need to be async if it’s an async function
        response = invoke_model(
            "You are a financial commentator for the Market Mayhem Podcast in a simulated world. "
            "Here's a news headline which just came out, make your fun, concise commentary combined with some serious analysis! "
            + text
        )
        cleaned = re.sub(r'\*.*?\*', '', response)
        double_cleaned = re.sub(r'\(.*?\)', '', cleaned)
        print(double_cleaned)

        stream = kokoro.create_stream(
            double_cleaned,
            voice='bm_george',
            speed=1.2,
            lang="en-us"
        )

        async for samples, sample_rate in stream:
            sd.play(samples, sample_rate)
            sd.wait()
    finally:
        with talking_lock:
            isTalking = False