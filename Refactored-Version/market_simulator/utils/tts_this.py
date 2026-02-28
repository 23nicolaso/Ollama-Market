import sounddevice as sd
from kokoro import KPipeline
import re
from threading import Thread, Lock
from market_simulator.utils.market_utils import invoke_model
import numpy as np
from market_simulator.utils.market_utils import model, price_history
from market_simulator.config import LLM_MODEL, TTS_LANG, get_state_string
import time
import random

pipeline = KPipeline(lang_code=TTS_LANG)

# Shared state
isTalking = False
last_text = ""
tts_thread = None
stop_signal = False
thread_lock = Lock()
person = "af_bella"
person_swap = False
first_text = True

conversation_history = []  # Store tuples like ("Emma", "Market is up...")
max_history_length = 5  # Limit to last N messages

def tts_this(text, sentiment = None, importance = None):
    global tts_thread, stop_signal

    if sentiment and importance:
        if sentiment >= 0.9:
            sentiment_str = "soar"
        elif sentiment > 0.5:
            sentiment_str = "drift up"
        elif sentiment <= 0.2:
            sentiment_str = "sell off harshly"
        elif sentiment == 0.5:
            sentiment_str = "be highly volatile"
        else:
            sentiment_str = "drift down"
        text = f'{text}. Our analysts expect markets to {sentiment_str} after the news.'
    with thread_lock:
        stop_signal = True  # Request current thread to stop
        if tts_thread and tts_thread.is_alive():
            tts_thread.join()  # Wait for current thread to stop

        stop_signal = False  # Reset stop signal for new thread
        tts_thread = Thread(target=run_tts_queue, args=(text,))
        tts_thread.start()

def run_tts_queue(text):
    start_conversation(text)

def stream_paragraphs(text = None, responding_mode = False):
    buffer = ""
    STATE_STRING = get_state_string()
    return_context = price_history["SPY"].getPercentageChange()
    return_str = "up "+ str(round(return_context,2)) + " percent for the day. " if return_context>0 else "down " + str(round(return_context,2)) + " percent for the day. "
    short_term_str = "SHORT TERM PRICE CHANGE: " + "DROPPING" if price_history["SPY"].getPriceChange(n=50) < 0 else "GOING UP"
    response_styles = [
        "Build on the idea with a concrete real-world market example",
        "Directly challenge the previous point with a counter-theory",
        "Partially agree but reframe the conclusion",
        "Compare the situation to a historical market analog",
        "Explain why this time may be different from past examples",
        "Translate macroeconomic data into market impact",
        "Break down the move using market microstructure",
        "Explain the role of liquidity and positioning",
        "Speculate on institutional positioning and intent",
        "Invent realistic retail trading flow and sentiment data",
        "Describe how retail traders are emotionally reacting",
        "Explain how social media finance discourse is shifting",
        "Compare results to analyst expectations and consensus",
        "Explain why expectations mattered more than the headline",
        "Explain why the market sold off on good news or rallied on bad news",
        "Argue the strongest bull case possible",
        "Argue the strongest bear case possible",
        "Identify what the market is mispricing or ignoring",
        "Explain how options flow and gamma exposure affect price action",
        "Speculate on hedge fund chatter and internal narratives",
        "Explain how passive flows or rebalancing influence price",
        "Discuss algorithmic trading amplification effects",
        "Frame the move within a broader market narrative",
        "Describe the psychological state of market participants",
        "Explain how recent wins or losses bias trader behavior",
        "Suggest a cautious trade expression",
        "Suggest an aggressive trade expression",
        "Suggest a contrarian trade expression",
        "Explain why staying on the sidelines may be optimal",
        "Describe risk factors that could break the thesis",
        "Identify upcoming catalysts or inflection points",
        "Predict short-term market reaction",
        "Predict medium-term market consequences",
        "Present a low-probability, high-impact scenario",
        "Respond as if debating another commentator live",
        "Summarize the thesis in a punchy on-air soundbite",
        "End with an unresolved question to maintain tension",
        "Reinterpret the same data through a different lens",
        "Explain who benefits and who loses from this move",
        "Compare this asset’s behavior to related sectors or peers",
        "Explain how monetary policy regime affects interpretation",
        "Describe how volatility itself is influencing decisions",
        "Explain how positioning could cause a squeeze or unwind",
        "Call out a popular narrative as misleading or lazy",
        "Highlight second-order and third-order effects",
        "Speculate on what smart money might do next",
        "Explain why the next data point matters more than this one"
    ]
    chosen_style = random.choice(response_styles)

    global last_text
    if responding_mode:
        personality = """Bella is an enthusiastic, slightly over-the-top market commentator who’s seen it all and loves dramatizing market moves. He loves making bold (and sometimes wrong) predictions. """
        response = model.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content':
                f"""{STATE_STRING}
                PERSONALITY:{personality}
                S&P 500 DAILY % CHANGE: {return_str}
                {short_term_str}
                The last thing you said was:
                {last_text}.
                DIALOG CONTINUATION STYLE: {chosen_style}.
                
                Continue the conversation with plain dialogue text. Do not include any stage directions, emotions in parentheses, tone indicators, or non-verbal actions.
                """
                }],
            stream=True
        )

    elif text:
        personality = """Bella is an enthusiastic, slightly over-the-top market commentator who’s seen it all and loves dramatizing market moves. He loves making bold (and sometimes wrong) predictions. """
        response = model.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': 
                f"""
                URGENT BREAKING NEWS: {text}, 
                {STATE_STRING}
                PERSONALITY:{personality}
                S&P 500 DAILY % CHANGE: {return_str}
                {short_term_str}
                DIALOG CONTINUATION STYLE: {chosen_style}.
                
                Explain the significance of the news. Do not include any stage directions, emotions in parentheses, tone indicators, or non-verbal actions.
                """
                }],
            stream=True
        )
    else:
        response = model.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': 
                f"""
                {STATE_STRING}
                PERSONALITY:{personality}
                S&P 500 DAILY % CHANGE: {return_str}
                {short_term_str}
                The last thing you said was:
                {last_text}.
                DIALOG CONTINUATION STYLE: {chosen_style}.
                
                Continue the conversation with plain dialogue text. Do not include any stage directions, emotions in parentheses, tone indicators, or non-verbal actions.
                """
                }],
            stream=True
        )

    global first_text
    buffer = ""
    if first_text:
        buffer = "Welcome to Ollama Market's favorite livestream, Market Mayhem. This is your host Bella Cartman, reporting on the news! "
        first_text = False
    if text:
        buffer += "BREAKING NEWS, "
        buffer += text
    for chunk in response:
        content = chunk['message']['content']
        cleaned = re.sub(r'\*', '', content)
        cleaned = re.sub(r'\(.*?\)', '', cleaned)
        buffer += cleaned

        breaks = re.split(r'\n', buffer)

        buffer = ""
        if not re.search(r'[!?]$', cleaned):
            buffer = breaks.pop()

        for segment in breaks:
            filtered_segment = segment.strip()
            if not filtered_segment:
                continue
            if len(filtered_segment) < 300:
                buffer = filtered_segment + ' ' + buffer
            else:
                yield filtered_segment.strip()
    
    if buffer.strip():
        yield buffer.strip()

def tts_queue(text=None, responding_mode=False):
    global isTalking, last_text, stop_signal, person_swap

    try:
        isTalking = True

        for response in stream_paragraphs(text, responding_mode):
            if stop_signal:
                break

            if response:
                last_text = response
                stream = pipeline(response, voice=person, speed=1.1, split_pattern=r'\n+')

                for _, (_, _, audio) in enumerate(stream):
                    if stop_signal:
                        sd.stop()
                        break

                    sd.play(audio, 24000)
                    sd.wait()
    finally:
        isTalking = False

def start_conversation(initial_text):
    # Initial speaker
    tts_queue(initial_text, responding_mode=False)

    # Continue the conversation loop until externally interrupted
    while not stop_signal:
        tts_queue(responding_mode=True)
