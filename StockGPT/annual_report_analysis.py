import requests
import os
import pickle
from openai_interface import get_openai_api
from langchain_community.document_loaders.pdf import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.summarize import load_summarize_chain

from bs4 import BeautifulSoup

def annual_report(id, y):
    url = "https://doc.twse.com.tw/server-java/t57sb01"
    # 建立 POST 請求的表單
    data = {
        "id":"",
        "key":"",
        "step":"1",
        "co_id":id,
        "year":y,
        "seamon":"",
        "mtype":"F",
        "dtype":"F04"
    }

    try:
        response = requests.post(url, data=data)
        # 擷取檔案名稱
        link = BeautifulSoup(response.text, 'html.parser')
        link1 = link.find('a').text
        print(link1)
    except Exception as e:
        print(f"發生{e}錯誤")
    
    # 建立第二個 POST
    data2 = {
        'step':'9',
        'kind':'F',
        'co_id':id,
        'filename':link1
    }

    try:
        response = requests.post(url, data=data2)
        link = BeautifulSoup(response.text, 'html.parser')
        link1 = link.find('a')
        # 取得 PDF 連結
        link2 = link1.get('href')
        print(link2)

    except Exception as e:
        print("發生{e}錯誤")

    # 發送 get
    try:
        response = requests.get(f"https://doc.twse.com.tw{link2}")
        # 取得 PDF 資料
        with open(y + '_' + id + '.pdf', 'wb') as file:
            file.write(response.content)
        print('OK')
    except Exception as e:
        print(f"發生{e}錯誤")

def pdf_loader(file, size, overlap):
    loader = PDFPlumberLoader(file)
    doc = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap
    )
    new_doc = text_splitter.split_documents(doc)
    db = FAISS.from_documents(new_doc, OpenAIEmbeddings())
    return db

def answer_question(question, db):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個根據年報資料與上下文作回答的助手,"
                   "如果有明確數據或技術(產品)名稱可以用數據或名稱回答,"
                   "回答以繁體中文與台灣用語為主。"
                   "{context}"),
        ("human", "{question}")
    ])
    llm_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    qa = RetrievalQA.from_llm(llm=llm_model, 
                              prompt=prompt,
                              return_source_documents=True,
                              retriever = db.as_retriever(search_kwargs={'k':10}))
    result = qa(question)
    return result

if __name__ == "__main__":
    os.environ['OPENAI_API_KEY'] = get_openai_api()
    db = pdf_loader('./112_2330.pdf',500,50)

    while True:
        question = input("輸入問題:")
        if not question.strip():
            break
        result = answer_question(question, db)
        print(result['result'])
        print('_______________')