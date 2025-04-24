import sounddevice as sd
from kokoro import KPipeline
from IPython.display import display, Audio
import re
from threading import Thread, Lock
from market_simulator.utils.market_utils import invoke_model
import numpy as np
from market_simulator.utils.market_utils import model
from market_simulator.config import LLM_MODEL

pipeline = KPipeline(lang_code='a')

# Shared state
isTalking = False
last_text = ""
tts_thread = None
stop_signal = False
thread_lock = Lock()

def tts_this(text, sentiment, importance):
    global tts_thread, stop_signal

    if sentiment >= 0.9:
        sentiment_str = "SOARING"
    elif sentiment > 0.5:
        sentiment_str = "up slightly"
    elif sentiment <= 0.2:
        sentiment_str = "selling off harshly"
    elif sentiment == 0.5:
        sentiment_str = "chopping sideways"
    else:
        sentiment_str = "down slightly"
    text = f'{text}. This news is {"highly" if importance >= 9 else "not that"} important. Markets are {sentiment_str} after the news' 
    
    with thread_lock:
        stop_signal = True  # Request current thread to stop
        if tts_thread and tts_thread.is_alive():
            tts_thread.join()  # Wait for current thread to stop

        stop_signal = False  # Reset stop signal for new thread
        tts_thread = Thread(target=run_tts_queue, args=(text,))
        tts_thread.start()


def run_tts_queue(text):
    tts_queue(text)

def stream_paragraphs(text = None):
    buffer = ""
    global last_text
    if text:
        response = model.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': 
                f"""
                You are an american ASMR financial podcast host for the podcast Market Mayhem. Comment on the latest market trends. 
                Stay engaging and entertaining.

                You just said:
                {last_text}

                Recent breaking news you haven’t yet addressed:
                {text}

                Continue your monologue naturally:
                """}],
            stream=True
        )
    else:
        response = model.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': 
                f"""
                You are an american ASMR financial podcast host for the podcast Market Mayhem. Comment on the latest market trends. 
                Stay engaging and entertaining.

                You just said:
                {last_text}

                Continue your monologue naturally:
                """}],
            stream=True
        )

    buffer = ""
    
    for chunk in response:
        content = chunk['message']['content']
        buffer += content
        
        breaks = re.split(r'\n', buffer)
        
        buffer = ""
        if not re.search(r'[!?]$', content):
            buffer = breaks.pop()
        
        for segment in breaks:
            # Skip content between asterisks
            filtered_segment = re.sub(r'\*.*?\*', '', segment)
            filtered_segment = re.sub(r'\(.*?\)', '', filtered_segment)
            if filtered_segment.strip():
                yield filtered_segment.strip()

def tts_queue(text=None):
    global isTalking, last_text, stop_signal

    try:
        isTalking = True
        for response in stream_paragraphs(text):
            if stop_signal:
                break

            if response:
                print(response)
                last_text = response
                stream = pipeline(response, voice='af_heart', speed=1.2, split_pattern=r'\n+')

                for i, (gs, ps, audio) in enumerate(stream):
                    if stop_signal:
                        sd.stop()
                        break

                    print(i, gs, ps)
                    sd.play(audio, 24000)
                    sd.wait()
    finally:
        print("done speaking")
        isTalking = False
        sd.stop()