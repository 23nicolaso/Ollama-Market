from market_simulator.agents.executional_trader import ExecutionalTrader
from market_simulator.config import ASSETS, INITIAL_PRICES
from market_simulator.utils.market_utils import invoke_model, markets, action_queue

class LLMFund(ExecutionalTrader):
    def __init__(self, accountID, cash, firm_id):
        super().__init__(accountID, cash)
        # Start with 30% of capital spread across all assets (dollar-budget, cash deducted)
        budget_per_asset = int(cash * 0.3 / len(ASSETS))
        for asset in ASSETS:
            shares = int(budget_per_asset / INITIAL_PRICES[asset])
            if shares > 0:
                cost = shares * INITIAL_PRICES[asset]
                self.account.addPosition(asset, shares)
                self.account.addPosition("CASH", -cost)

        self.firm_id = firm_id
        self.report_txt = ""
        if firm_id == "Gemma Fund":
            self.prompt = f"""You are the head analyst at Gemma Fund, a cutting-edge quant hedge fund operating in a simulated financial world.
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
            Please explain your positioning guide as a list of TICKER:% CHANGE combos, so it can be executed on efficiently later. 

            Use only the following tickers: {', '.join(ASSETS)}
            Consensus is for CPI to rise 2.8% YoY and no interest rate change. 
            Your job is to find any edge beyond consensus using alternative data.
            """

        elif firm_id == "Ollama Fund":
            self.prompt = f"""You are the lead macroeconomist at Ollama Fund, a traditional macro-focused investment firm in a simulated financial world.
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
                Use only the following tickers: {', '.join(ASSETS)}
                Please explain your positioning guide as a list of TICKER:% CHANGE combos, so it can be executed on efficiently later.
                Consensus is for CPI to rise 2.8% YoY and no interest rate change. Your job is to judge whether the macroeconomic fundamentals support or contradict this.
                """

        else:
            self.prompt = """You are a member of the Wall Street Bets community on reddit, and you are POSITIVELY SURE that you know where
            CPI numbers will be today. Most analysts are predicting no change in interest rates, a rise of 2.8 percent YoY in CPI.
            What are your estimates? You can use questionable methods to estimate the numbers, which may or may not be accurate!
            Also please say if you are long or short the S&P 500 with options. Please use wallstreetbets slang like go to the moon, rocket, regard
            """
    
    # extract the tool call from the response
    def extract_tool_call(self, text):
        from rapidfuzz import process
        from market_simulator.config import ASSETS
        import re

        pattern = r'modify_position\((["\'])(.*?)\1\s*,\s*([-+]?\d*\.?\d+)\)'
        matches = re.findall(pattern, text)
        for params in matches:
            # identify the most similar asset ticker
            ticker = process.extractOne(params[1].upper(), ASSETS)[0]
            # if recieved outside +/-0.2, div by 100 as it should be converted into decimal (LLM said 0.2 while meaning 0.2%). 
            # BOUNDED TO +/- 20% max repositioning.
            pct_delta = min(max(-0.2, float(params[2]) if (float(params[2]) < 0.3 and float(params[2]) > -0.3) else float(params[2]) / 100),0.2)
            self.modify_position(str(ticker), pct_delta) 

    def analyzeOutcomes(self):
        from reportlab.lib.pagesizes import LETTER
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.units import inch
        import re

        print(self.firm_id, " is producing their analyst report.")
        self.report_txt = invoke_model(self.prompt)
        self.report_txt.split('---')[0].strip()

        doc = SimpleDocTemplate(f"{self.firm_id}_Fund_Assessment.pdf", pagesize=LETTER,
                        rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)

        # Styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='myHeading1', fontSize=16, fontName="Times-Bold", leading=20, spaceAfter=10, spaceBefore=10, bold=True))
        styles.add(ParagraphStyle(name='myHeading2', fontSize=14, fontName="Times-Bold", leading=18, spaceAfter=8, spaceBefore=8, bold=True))
        styles.add(ParagraphStyle(name='myBodyText', fontSize=11, fontName="Times-Roman", leading=14, spaceAfter=8))
        styles.add(ParagraphStyle(name='myBulletList', fontSize=11, fontName="Times-Roman", leftIndent=12, leading=14, spaceAfter=6))
        styles.add(ParagraphStyle(name='myItalic', fontSize=11, fontName="Times-Italic", leading=14, spaceAfter=6, italic=True))

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


        doc.build(format_text(self.report_txt))
        print(f"PDF created: {self.firm_id}_Fund_Assessment.pdf")

        call_gen_prompt = f'''Right now it is time for Pre-Release Positioning.
            Please write the calls to modify_position("ticker", change) for pre-release positioning as outlined in the report  
            Due to limitations, the only tickers you can actually trade are:
            {', '.join(ASSETS)}

            You can call modify_position("ticker", change) to modify your position in asset by the inputted change float.
            A positive change input percentage means you are buying that percent for your portfolio. A negative change input means you are selling that percent of your portfolio. 
            Please do not state the calls more than once, do not mention tickers you cannot trade. 
            <{self.report_txt}>'''

        calls = invoke_model(call_gen_prompt)
        self.extract_tool_call(calls)
        
    def modify_position(self, asset: str, percent: float):
        self.targetPosition(markets[asset], "buy" if percent > 0 else "sell", markets[asset].last_price, markets[asset].last_price, int(self.getPosition(asset) * (1+percent)), False)
        direction = "BUY" if percent > 0 else "SELL"
        action_queue.put(f"{self.firm_id}: {direction} {asset} ({percent*100:+.1f}%) @ ${markets[asset].last_price:.2f}")

    def tradeResult(self, data):
        call_gen_prompt = f'''Right now it is time for Post-Release Positioning.
            The data is in: <{data}>
            Please write the calls to modify_position to adjust your positioning as per the post-release positioning guide outlined in attached report for this CPI, interest rate decision subscenario  
            You can call modify_position("ticker", change) to modify your position in asset by the inputted change float.
            A positive change input percentage means you are buying that percent for your portfolio. A negative change input means you are selling that percent of your portfolio. 
            Please do not state the calls more than once, do not mention tickers you cannot trade. 
            Report:
            <{self.report_txt}>'''
        
        calls = invoke_model(call_gen_prompt)
        self.extract_tool_call(calls)

    def analyzeAndTradeNews(self, news):
        if self.firm_id == "Gemma Fund":
            prompt = f"""You are the head analyst at Gemma Fund, a cutting-edge quant hedge fund operating in a simulated financial world.
            Your firm specializes in deriving market insights from alternative data sources (e.g., satellite imagery, shipping data, web-scraped pricing info, social sentiment analysis) 
            and applying advanced statistical learning algorithms. Bloomberg just reported the following news headline: <{news}>.
            Your task is to analyze this news, and adjust your firm's positioning accordingly in response.
            Due to limitations, the only tickers you can actually trade are:
            {', '.join(ASSETS)}

            You can call modify_position("ticker", change) to modify your position in asset by the inputted change float.
            A positive change input percentage means you are buying that percent for your portfolio. A negative change input means you are selling that percent of your portfolio. 
            Please do not repeat calls, do not mention tickers you cannot trade. 
            """

        elif self.firm_id == "Ollama Fund":
            prompt = f"""You are the lead macroeconomist at Ollama Fund, a traditional macro-focused investment firm in a simulated financial world.
                Your firm makes investment decisions grounded in classical economic theory, business cycles, monetary policy frameworks, and long-term historical data patterns.
                Bloomberg just reported the following news headline: <{news}>.
                Your task is to analyze this news, and adjust your firm's positioning accordingly in response.
                Due to limitations, the only tickers you can actually trade are:
                {', '.join(ASSETS)}

                You can call modify_position("ticker", change) to modify your position in asset by the inputted change float.
                A positive change input percentage means you are buying that percent for your portfolio. A negative change input means you are selling that percent of your portfolio. 
                Please do not repeat calls, do not mention tickers you cannot trade. 
                """
        
        else:
            prompt = f"""You are a wallstreet bets member and this news just hit the tape <{news}>. Update your portfolio's positioning in an interesting way in response.
                        Due to limitations, the only tickers you can actually trade are:
                        {', '.join(ASSETS)}

                        You can call modify_position("ticker", change) to modify your position in asset by the inputted change float.
                        A positive change input percentage means you are buying that percent for your portfolio. A negative change input means you are selling that percent of your portfolio. 
                        Please do not repeat calls, do not mention tickers you cannot trade. 
                        """
        
        calls = invoke_model(prompt)
        self.extract_tool_call(calls)