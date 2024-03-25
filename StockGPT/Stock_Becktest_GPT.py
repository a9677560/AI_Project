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
        and exclude any unrelated content.\
        At the same time, the reply content cannot include the ```Python``` format."
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

def ai_backtest(stock_id, period, user_msg, add_msg="無", print_stat = False, debug = False):
    try:
        df = yf.download(stock_id, period=period)

        # 取得指標計算程式碼
        code_str = ai_helper(df, user_msg)
        local_namespace = {}
        exec(code_str, globals(), local_namespace)
        calculate = local_namespace['calculate']
        new_df = calculate(df)

        # 取得回測策略程式碼
        strategy_str = ai_strategy(new_df, user_msg, add_msg)
        # Debug mode
        if debug == True:
            print(strategy_str)
            print("-----------------------------")

        exec(strategy_str, globals(), local_namespace)
        AIstrategy = local_namespace['AI_strategy']
        backtest = Backtest(new_df,
                            AIstrategy,
                            cash=100000,
                            commission=0.004,
                            trade_on_close=True,
                            exclusive_orders=True
                            )
        stats = backtest.run()

        if print_stat == True:
            print(stats)
            print("------------------------------")
            
        return str(stats)
    except SystemError as se:
        print(f"Syntax error occurred during code execution: {se}")
        return None
    except ValueError as ve:
        print(f"Value error occurred during code execution: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during code execution: {e}")
        return None


def backtest_analysis(*backtests):
    content_list = [f"策略 {i+1}: {report}" for i, report in enumerate(backtests)]
    content = "\n".join(content_list)
    content += "\n\n請依照以上資料給我一份約 200 字的分析報告。若有多個策略,\
                請選出最好的策略及說明原因, reply in 繁體中文"
        
    msg = [{
        "role": "system",
        "content": "你是一位專業的證券分析師,我會給你交易策略的回測績效,\
                    請幫我進行績效分析.不用詳細講解每個欄位,\
                    重點說明即可,並回答交易策略的好壞.\
                    此外,回覆的格式必須人性化. 比如數字和中文與英文和中文的間隔,\
                    數字或英文前後都必須要有空格來做分離. 比如: 分別為 0.39 和 0.58。"
    }, {
        "role": "user",
        "content": content
    }]

    reply_data = get_reply(msg)
    return reply_data

if __name__ == "__main__":
    stats1 = ai_backtest(stock_id="2330.tw", period="5y",
                         user_msg="MACD", add_msg="請設置 10% 的停損點和 20% 的停利點")
    stats2 = ai_backtest(stock_id="2330.tw", period="5y",
                         user_msg="SMA")
    stats3 = ai_backtest(stock_id="2330.tw", period="5y",
                        user_msg="RSI", add_msg="請設置 10% 的停損點與 20% 的停利點")
    
    # 回測時偶爾會出現意外狀況，故檢查
    if stats1 is None or stats2 is None or stats3 is None:
        print("回測時出現錯誤，請重新再試")
        exit()

    reply = backtest_analysis(stats1, stats2, stats3)
    print(reply)