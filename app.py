from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import client
import os
from vector_db import load_vector_db
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableBranch

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

combiner = RunnableLambda(combine_docs)

persist_directory = './data_base/chroma_db'
vectordb = load_vector_db(persist_directory)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

retrieval_chain = retriever | combiner

llm = ChatOpenAI(
    openai_api_key=os.getenv('SILICON_API_KEY'),
    model='Qwen/Qwen3-8B',
    temperature=0,
    max_tokens=1024,
    streaming=False,
    openai_api_base='https://api.siliconflow.cn/v1/',
)


def f1():
    output = llm.invoke("请你自我介绍一下自己！")
    print(output)

def f2():
    prompt = """请你将由三个反引号分割的文本翻译成英文！\
text: ```{text}```
"""
    text = "我带着比身体重的行李，\
游入尼罗河底，\
经过几道闪电 看到一堆光圈，\
不确定是不是这里。\
"
    input = prompt.format(text=text)
    print(input)

def f3():
    template = "你是一个翻译助手，可以帮助我将 {input_language} 翻译成 {output_language}."
    human_template = "{text}"
    chat_prompt = ChatPromptTemplate([
        ("system", template),
        ("human", human_template),
    ])
    text = "我带着比身体重的行李，\
游入尼罗河底，\
经过几道闪电 看到一堆光圈，\
不确定是不是这里。\
"
    # messages = chat_prompt.invoke({"input_language": "中文", "output_language": "英文", "text": text})
    # print(messages)
    # output = llm.invoke(messages)
    # print(output)
    # output_parser = StrOutputParser()
    # output = output_parser.invoke(output)
    # print(output)
    chain = chat_prompt | llm | StrOutputParser()
    output = chain.invoke({"input_language": "中文", "output_language": "英文", "text": text})
    print(output)
    text = 'I carried luggage heavier than my body and dived into the bottom of the Nile River. After passing through several flashes of lightning, I saw a pile of halos, not sure if this is the place.'
    output = chain.invoke({"input_language": "英文", "output_language": "中文","text": text})
    print(output)

def f4():
    print(f"向量库中存储的数量：{vectordb._collection.count()}")
    question = "什么是prompt engineering?"
    docs = retriever.invoke(question)
    print(f"检索到的内容数：{len(docs)}")
    for i, doc in enumerate(docs):
        print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-----------------------------------------------------\n")

def f5():
    text = retrieval_chain.invoke("南瓜书是什么？")
    print(text)

def f6():
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。请你在回答的最后说“谢谢你的提问！”。
{context}
问题: {input}
"""
    prompt = PromptTemplate(template=template)
    qa_chain = (
        RunnableParallel({"context": retrieval_chain, "input": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
    question_1 = "什么是南瓜书？"
    question_2 = "Prompt Engineering for Developer是谁写的？"
    result = qa_chain.invoke(question_1)
    print("大模型+知识库后回答 question_1 的结果：")
    print(result)
    result = qa_chain.invoke(question_2)
    print("大模型+知识库后回答 question_2 的结果：")
    print(result)

def f7():
    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    messages = qa_prompt.invoke(
        {
            "input": "南瓜书是什么？",
            "chat_history": [
                ("human", "西瓜书是什么？"),
                ("ai", "西瓜书是指周志华老师的《机器学习》一书，是机器学习领域的经典入门教材之一。"),
            ],
            "context": ""
        }
    )
    for message in messages.messages:
        print(message.content)

def f8():
    condense_question_system_template = (
        "请根据聊天记录完善用户最新的问题，"
        "如果用户最新的问题不需要完善则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    retrieve_docs = RunnableBranch(
        # 分支 1: 若聊天记录中没有 chat_history 则直接使用用户问题查询向量数据库
        (lambda x: not x.get("chat_history", False), (lambda x: x['input']) | retriever),
        # 分支 2 : 若聊天记录中有 chat_history 则先让 llm 根据聊天记录完善问题再查询向量数据库
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )
    def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs["context"]) # 将 docs 改为 docs["context"]
    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough.assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    qa_history_chain = RunnablePassthrough.assign(
        context=(lambda x: x) | retrieve_docs
    ).assign(answer=qa_chain)
    # output = qa_history_chain.invoke({
    #     "input": "西瓜书是什么？",
    #     "chat_history": []
    # })
    # print(output)
    output = qa_history_chain.invoke({
        "input": "南瓜书跟它有什么关系？",
        "chat_history": [
            ("human", "西瓜书是什么？"),
            ("ai", "西瓜书是指周志华老师的《机器学习》一书，是机器学习领域的经典入门教材之一。"),
        ]
    })
    print(output)


if __name__ == '__main__':
    f8()
