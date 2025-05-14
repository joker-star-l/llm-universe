from client import OpenAIEmbeddings
from reader import load_pdf, load_markdown, split_documents
import os
from langchain_community.vectorstores import Chroma

def create_vector_db(persist_directory: str) -> Chroma:
    file_paths = []
    folder_path = './data_base/knowledge_db'    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    # print(file_paths)

    texts = []
    for file_path in file_paths:
        file_type = file_path.split('.')[-1]
        if file_type == 'pdf':
            texts.extend(load_pdf(file_path))
        elif file_type == 'md':
            texts.extend(load_markdown(file_path))

    # text = texts[1]
    # print(f"每一个元素的类型：{type(text)}.", 
    #     f"该文档的描述性数据：{text.metadata}", 
    #     f"查看该文档的内容:\n{text.page_content[0:]}", 
    #     sep="\n------\n")

    split_docs = split_documents(texts)

    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory,
    )

    return vectordb

def load_vector_db(persist_directory: str) -> Chroma:
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings(),
    )

def similarity_search(vectordb: Chroma, question: str, k: int = 3):
    sim_docs = vectordb.similarity_search(question, k=k)
    print(f"检索到的内容数：{len(sim_docs)}")
    for i, sim_doc in enumerate(sim_docs):
        print(f"检索到的第{i + 1}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")

def max_marginal_relevance_search(vectordb: Chroma, question: str, k: int = 3):
    mmr_docs = vectordb.max_marginal_relevance_search(question, k=k)
    print(f"MMR检索到的内容数：{len(mmr_docs)}")
    for i, sim_doc in enumerate(mmr_docs):
        print(f"MMR 检索到的第{i + 1}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")


if __name__ == "__main__":
    persist_directory = './data_base/chroma_db'
    if os.path.exists(persist_directory):
        print("向量数据库已经存在，直接加载")
        vectordb = load_vector_db(persist_directory)
    else:
        print("向量数据库不存在，创建新的向量数据库")
        vectordb = create_vector_db(persist_directory)

    print(f"向量库中存储的数量：{vectordb._collection.count()}")

    question = "什么是大语言模型"
    # similarity_search(vectordb, question, 3)
    max_marginal_relevance_search(vectordb, question, 3)
