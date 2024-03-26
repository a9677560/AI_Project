from openai import OpenAI, OpenAIError
import os

def get_openAI_client():
    file_path = os.path.join(os.getcwd(), "Data", "openai_token")
    with open(file_path, 'r') as file:
        api_key = file.read().strip()  # 去除空白字符
    client = OpenAI(api_key=api_key)
    return client

def get_reply(messages, model = 'gpt-3.5-turbo'):
    try:
        client = get_openAI_client()
        response = client.chat.completions.create(model=model, messages=messages)
        reply = response.choices[0].message.content
    except OpenAIError as err:
        reply = f"發生 {err.type} 錯誤\n{err.message}"
    return reply