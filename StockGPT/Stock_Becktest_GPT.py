from openai import OpenAI, OpenAIError
import yfinance as yf
import pandas as pd
import datetime as dt 
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import os

# 繞過 compiler 檢查用
class AI_strategy:
    pass

def get_openAI_client():
    file_path = os.path.join(os.getcwd(), "Data", "openai_token")
    with open(file_path, 'r') as file:
        api_key = file.read().strip()  # 去除空白字符
    client = OpenAI(api_key=api_key)
    return client

def get_reply(messages):
    try:
        client = get_openAI_client()
        response = client.chat.completions.create(model='gpt-3.5-turbo', messages=messages)
        reply = response.choices[0].message.content
    except OpenAIError as err:
        reply = f"發生 {err.type} 錯誤\n{err.message}"
    return reply

def ai_helper(df, user_msg):
    msg = [{
        "role": "system",
        "content": 
        f"As a professional code generation robot, \n\
        I require your assistance in generating Python code \n\
        based on specific user requirements. To proceed, \n\
        I will provide you with a dataframe (df) that follows the \n\
        format {df.columns}. Your task is to carefully analyze the\n\
        user's requirements and generate python code \n\
        accordingly. Please note that your response should solely \n\
        consist of the code itself,\n\
        and no additional information should be included."
    }, {
        "role": "user",
        "content":
        f"The user requirement:{user_msg} \n\
        Your task is to develop a Python function named \
        'calculate(df)'. This function should accept a dataframe as \
        its parameter. Ensure that you only utilize the columns \
        present in the dataset, specifically {df.columns}.\
        After processing, the function should return the processed \
        dataframe. Your response should strictly contain the Python \
        code for the 'calculate(df)' function \
        and exclude any unrelated content."
    }]
    reply_data = get_reply(msg)
    return reply_data

def ai_strategy(df, user_msg, add_msg="無"):
    # code_exmaple 須按照 python 的程式格式，不可為了美化而修改格式，否則將會出現 IndentationError
    code_example = '''
class AI_strategy(Strategy):
    def init(self):
        super().init()

    def next(self):
        if crossover(self.data.short_ma, self.data.long_ma):
            self.buy(size=1,
                    sl=self.data.Close[-1] * 0.90,
                    tp=self.data.Close[-1] * 1.10)
        elif crossover(self.data.long_ma, self.data.short_ma):
            self.sell(size=1,
                    sl=self.data.Close[-1] * 1.10,
                    tp=self.data.Close[-1] * 0.90)
    '''

    # 先模擬一次與 AI 的對話，透過 assistant 輸入 AI 的回覆模板，
    # 讓 AI 知道該以什麼形式來回覆，以此提高 AI 回覆的穩定性。
    msg = [
        {
            "role":"system",
            "content":"As a Python code generation bot, your task is to generate \
            code for a stock strategy based on user requirements and df. \
            Please note that your response should solely \
            consist of the code itself, \
            and no additional information should be included."
        }, {
            "role":"user",
            "content":"The user requirement:計算 ma,\n\
            The additional requirement: 請設置 10% 的停利與停損點\n\
            The df.columns =['Open', 'High', 'Low',	'Close', 'Adj Close', 'Volume', 'short_ma',	'long_ma']\n\
            Please using the crossover() function in next(self)\
            Your response should strictly contain the Python \
            code for the 'AI_strategy(Strategy)' class \
            and exclude any unrelated content."
        }, {
            "role":"assistant",
            "content":f"{code_example}"
        }, {
            "role":"user",
            "content":f"The user requirement:{user_msg}\n\
            The additional requirement:{add_msg}\n\
            The df.columns ={df.columns}\n\
            Your task is to develop a Python class named \
            'AI_strategy(Strategy)'\
            Please using the crossover() function in next(self)."
        }
    ]

    reply_data = get_reply(msg)
    return reply_data

# TODO: 將 AI 生成的 calculate(df) 執行並回傳 new_df
def ai_code_exec(df, instruction):
    code_str = ai_helper(df, instruction)
    print(code_str)

    # 將 exec 生成的 calculate 設為 local variable
    local_vars = {}
    # 執行 AI 產生的程式碼
    exec(code_str, globals(), local_vars)
    calculate = local_vars['calculate']

    new_df = df.copy()
    new_df = calculate(new_df)
    return new_df

stock_id = "2330.tw"

# 下載股票資料並讓 AI 計算指標
df = yf.download(stock_id, period='5y')
user_msg = ["MACD", "請設置 10% 的停損點和 20% 的停利點"]
new_df = ai_code_exec(df, user_msg[0])

strategy_str = ai_strategy(new_df, user_msg[0], user_msg[1])
print(strategy_str)
print("--------------------")
exec(strategy_str)
backtest = Backtest(new_df,
                    AI_strategy,
                    cash=100000,
                    commission=0.004,
                    trade_on_close=True,
                    exclusive_orders=True)
stats = backtest.run()
print(stats)