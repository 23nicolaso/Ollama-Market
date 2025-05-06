"""
A PROTOTYPE SCRIPT FOR LLM_FUNDS.PY, NOW OBSOLETE
"""

from market_simulator.utils.market_utils import invoke_model_stream
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import inch
import re
from market_simulator.config import ASSETS

# Save this as a PDF
doc = SimpleDocTemplate("Ollama_Fund_Assessment.pdf", pagesize=LETTER,
                        rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)

# Styles
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='myHeading1', fontSize=16, fontName="Times-Bold", leading=20, spaceAfter=10, spaceBefore=10, bold=True))
styles.add(ParagraphStyle(name='myHeading2', fontSize=14, fontName="Times-Bold", leading=18, spaceAfter=8, spaceBefore=8, bold=True))
styles.add(ParagraphStyle(name='myBodyText', fontSize=11, fontName="Times-Roman", leading=14, spaceAfter=8))
styles.add(ParagraphStyle(name='myBulletList', fontSize=11, fontName="Times-Roman", leftIndent=12, leading=14, spaceAfter=6))
styles.add(ParagraphStyle(name='myItalic', fontSize=11, fontName="Times-Italic", leading=14, spaceAfter=6, italic=True))

# Function to process and convert to Paragraphs
def format_text(text):
    content = []
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            content.append(Spacer(1, 0.1 * inch))
            continue

        # Headers
        if line.startswith("## "):
            content.append(Paragraph(line[3:], styles['myHeading1']))
        elif re.match(r"^\*\*.*\*\*$", line):  # Bolded section titles
            content.append(Paragraph(f"<b>{line.strip('**')}</b>", styles['myHeading2']))
        elif re.match(r"^\*\*.*\*\*:.*", line):  # Bolded inline fields
            bold_part, rest = line.split("**: ", 1)
            content.append(Paragraph(f"<b>{bold_part.strip('**')}:</b> {rest}", styles['myBodyText']))
        elif line.startswith("* "):  # Bulleted list - now check for inline formatting inside
            # Process inline formatting within the bulleted line
            processed_line = line[2:] # Remove bullet point marker

            # Handle bold inside line
            processed_line = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", processed_line)

            # Handle italics inside line
            processed_line = re.sub(r"\*(.*?)\*", r"<i>\1</i>", processed_line)

            content.append(Paragraph("-" + processed_line, styles['myBulletList'])) # Add bullet symbol back for display
        elif re.search(r"\*.*\*", line):  # Italics inside line
            line = re.sub(r"\*(.*?)\*", r"<i>\1</i>", line)
            content.append(Paragraph(line, styles['myBodyText']))
        elif re.search(r"\*\*(.*?)\*\*", line):  # Bold inside line
            line = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
            content.append(Paragraph(line, styles['myBodyText']))
        else:
            content.append(Paragraph(line, styles['myBodyText']))
    return content

prompt = f"""You are the head analyst at Gemma Fund, a cutting-edge quant hedge fund operating in a simulated financial world.
            Your firm specializes in deriving market insights from alternative data sources (e.g., satellite imagery, shipping data, web-scraped pricing info, social sentiment analysis) 
            and applying advanced statistical learning algorithms.

            In 15 minutes the federal reserve is releasing the CPI numbers and interest rate decisions. 

            **Your task:**
            - Analyze relevant data (make it up, but cite non-traditional data sources like freight traffic, real-time grocery prices, or social media chatter).
            - Forecast the likely CPI and interest rate outcomes based on your models.
            - Explain the model’s confidence level and recent backtest behavior.

            Then, write a **positioning plan**:
            - How is your model positioning ahead of the release?
            - How will your fund react *post-release* for 3 possible outcomes:
            - CPI higher than expected
            - CPI lower than expected
            - CPI in line

            {', '.join(ASSETS)}
            Consensus is for CPI to rise 2.8% YoY and no interest rate change. 
            Your job is to find any edge beyond consensus using alternative data.
        """

prompt2 = """You are a member of the Wall Street Bets community on reddit, and you are POSITIVELY SURE that you know where
            CPI numbers will be today. Most analysts are predicting no change in interest rates, a rise of 2.8 percent YoY in CPI.
            What are your estimates? You can use questionable methods to estimate the numbers, which may or may not be accurate!
            Also please say if you are long or short the S&P 500 with options. Please use wallstreetbets slang like go to the moon, rocket, regard
        """

prompt3 = f"""You are the lead macroeconomist at Ollama Fund, a traditional macro-focused investment firm in a simulated financial world.
                Your firm makes investment decisions grounded in classical economic theory, business cycles, monetary policy frameworks, and long-term historical data patterns.

                In 15 minutes, the Federal Reserve will release CPI numbers and its interest rate decision.

                **Your task:**
                - Analyze leading macroeconomic indicators (e.g., wage growth, money supply, unemployment, productivity, output gap).
                - Predict the CPI and interest rate decision using established economic models (e.g., Taylor Rule, Phillips Curve).
                - Discuss policy expectations from the Fed and long-term implications.

                Then, write a **positioning plan**:
                - What is your baseline macro thesis going into the release?
                - How will you respond *post-release* under three cases:
                - CPI higher than expected
                - CPI lower than expected
                - CPI in line
            {', '.join(ASSETS)}
            Consensus is for CPI to rise 2.8% YoY and no interest rate change. Your job is to judge whether the macroeconomic fundamentals support or contradict this.
        """

buffer = ""
for text in invoke_model_stream(prompt3):
    print(text, end="")
    buffer += text

buffer = buffer.split('---')[0].strip()
doc.build(format_text(buffer))
print("PDF created: Ollama_Fund_Assessment.pdf")

prompt4 = f'''Right now it is time for Post-Release Positioning, there was no change in interest rates, CPI came in at 3.3 percent increase.    
            Due to limitations, the only tickers you can actually trade are:
            {', '.join(ASSETS)}

            You can call modify_position("ticker", change) to modify your position in asset by the inputted change float.
            A positive change input percentage means you are buying that percent for your portfolio. A negative change input means you are selling that percent of your portfolio. 
            Please write the calls to modify_position("ticker", change) for this specific sub-scenario.
            Please do not state the calls more than once, do not mention tickers you cannot trade. 
            <{buffer}>'''

print('\n')
buffer2 = ""
for text in invoke_model_stream(prompt4):
    print(text, end="")
    buffer2+=text

def modify_position(asset, percent):
    print("wait, amazing!", asset, ": ", str(percent))

# extract the tool call from the response
def extract_tool_call(text):
    from rapidfuzz import process
    from market_simulator.config import ASSETS
    import re

    pattern = r'modify_position\((["\'])(.*?)\1\s*,\s*([-+]?\d*\.?\d+)\)'
    matches = re.findall(pattern, text)
    for params in matches:
        # uses most similar asset ticker
        ticker = process.extractOne(params[1].upper(), ASSETS)[0]
        # if recieved as >0.5, div by 100 (as its probably as a %), otherwise keep as is (converting to decimal).
        pct_delta = float(params[2]) if (float(params[2]) < 0.5 and float(params[2]) > -0.5) else float(params[2]) / 100
        modify_position(str(ticker), pct_delta)

print(extract_tool_call(buffer2))