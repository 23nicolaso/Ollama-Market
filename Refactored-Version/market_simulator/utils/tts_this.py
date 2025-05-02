import sounddevice as sd
from kokoro import KPipeline
from IPython.display import display, Audio
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
person2 = "am_eric"
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
            sentiment_str = "chop sideways"
        else:
            sentiment_str = "drift down"
        text = f'{text}. Our analysts expect markets to {sentiment_str} after the news.'
        text += "" 
    else:
        text = text
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
    response_styles = [
        "Build on the idea and give a concrete market example.",
        "Challenge the previous point directly and offer a counter-theory.",
        "Introduce a new economic metaphor and explain how it applies.",
        "Summarize the discussion so far and suggest a next topic.",
        "Compare current events to historical financial trends.",
        "Give your prediction on where markets are headed."
    ]
    chosen_style = random.choice(response_styles)

    global last_text
    if responding_mode:
        cohost_name = "Bella Pearl" if not person_swap else "Eric Cartman"
        personality = """Eric, The Excitable Veteran.
            Personality: Eric is an enthusiastic, slightly over-the-top market commentator who’s seen it all and loves dramatizing market moves. He loves making bold (and sometimes wrong) predictions. Deep down, though, he’s knowledgeable and loves explaining market behavior.
            Speech style: Fast-talking, hyperbolic, peppered with jokes.
            """ if not person_swap else """Bella, The Sharp, Witty Analyst.
            Personality: Bella is sharp, witty, and tends to be a voice of reason. She’s a younger, highly skilled analyst with a bit of a sarcastic streak. 
            Speech style: Calm, confident, analytical — with a touch of dry humor and occasional savage one-liners.
        """
        response = model.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content':
                f"""{STATE_STRING}
                You are an economic commentator for the market mayhem livestream. You are {personality}. The S&P 500 is currently {return_str}.
                Your cohost, {cohost_name} just said:
                {last_text}.
                Try to be concise, and respond using the following instruction: {chosen_style}
                
                Respond only with plain dialogue text. Do not include any stage directions, emotions in parentheses, tone indicators, or non-verbal actions.
                """
                }],
            stream=True
        )

    elif text:
        cohost_name = "Bella Pearl" if not person_swap else "Eric Cartman"
        personality = """Eric, The Excitable Veteran.
            Personality: Eric is an enthusiastic, slightly over-the-top market commentator who’s seen it all. He loves making bold (and sometimes wrong) predictions. Deep down, though, he’s knowledgeable and loves teaching casual players about market behavior.
            Speech style: Fast-talking, hyperbolic, peppered with jokes.
            """ if not person_swap else """Bella, The Sharp, Witty Analyst.
            Personality: Bella is sharp, witty, and tends to be a voice of reason. She’s a younger, highly skilled analyst with a bit of a sarcastic streak. While Eric gets hyped, Bella brings in cool-headed, data-driven insights.
            Speech style: Calm, confident, analytical — with a touch of dry humor and occasional savage one-liners.
        """
        response = model.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': 
                f"""
                {STATE_STRING}
                You are an economic commentator for the market mayhem livestream. You are {personality} 
                You should explain whatever you are seeing and its implications. Market Mayhem aims to summarize all financial news and information
                to make it easy for traders to understand the markets. 
                Respond only with plain dialogue text. Do not include any stage directions, emotions in parentheses, tone indicators, or non-verbal actions.

                Here's some new content that you should talk about, start the discussion about it, and your cohost, {cohost_name}, will respond to you:
                {text}
                """
                }],
            stream=True
        )
    else:
        cohost_name = "Bella Pearl" if not person_swap else "Eric Cartman"
        response = model.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': 
                f"""
                {STATE_STRING} 
                You are an economic commentator who covers the activity in the simulated Ollama Market in your livestream. 
                Your content is analytical, and should explain whatever you are seeing and its implications.
                Respond only with plain dialogue text. Do not include any stage directions, emotions in parentheses, tone indicators, or non-verbal actions.

                Your cohost, {cohost_name} just said the following:
                {last_text}.
                
                Try to build off your cohost’s point, introduce a new angle or data point, and ask a question that invites debate or deeper exploration.
                If you disagree with their interpretation, say so and explain why.

                For additional context, the S&P 500 is currently {return_str}
                """
                }],
            stream=True
        )

    global first_text
    buffer = ""
    if first_text:
        buffer = "Welcome to Ollama Market's favorite livestream, Market Mayhem. This is your host Eric Cartman, and my cohost Bella Pearl, reporting on the news! "
        first_text = False
    if text:
        buffer += "We just recieved the news that "
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
                print(response)
                if not response == "":
                    last_text = response
                stream = pipeline(response, voice=person if person_swap else person2, speed=1.2, split_pattern=r'\n+')

                for i, (gs, ps, audio) in enumerate(stream):
                    if stop_signal:
                        sd.stop()
                        break

                    print(i, gs, ps)
                    sd.play(audio, 24000)
                    sd.wait()
    finally:
        isTalking = False
        print("done speaking")

def start_conversation(initial_text):
    global person_swap
    
    # Initial speaker
    tts_queue(initial_text, responding_mode=False)
    
    # Start the conversation loop
    while not stop_signal:
        person_swap = not person_swap
        tts_queue(responding_mode=True)
