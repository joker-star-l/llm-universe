import os
from typing import List
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from langchain_core.embeddings import Embeddings

_ = load_dotenv(find_dotenv())

client = OpenAI(
    base_url='https://api.siliconflow.cn/v1/',
    api_key=os.getenv('SILICON_API_KEY'),
)

def gen_messages(prompt):
    messages = [{"role": "user", "content": prompt}]
    return messages

def get_completion(prompt, model='Qwen/Qwen3-8B', temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=gen_messages(prompt),
        temperature=temperature,
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return 'generate answer failed'

def get_embedding(text, model='BAAI/bge-m3'):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    if len(response.data) > 0:
        return [d.embedding for d in response.data]
    raise ValueError('generate embedding failed')

class OpenAIEmbeddings(Embeddings):
    def __init__(self):
        self.client = client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        n = len(texts)
        batch_size = 64
        i = 0
        all_embeddings = []

        while i < n:
            embeddings = self.client.embeddings.create(
                model='BAAI/bge-m3',
                input=texts[i:min(i + batch_size, n)],
            )
            i += batch_size
            all_embeddings.extend([d.embedding for d in embeddings.data])

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

if __name__ == '__main__':
    # response = get_completion('你好')
    # print(response)

    # response = get_embedding(['猫', 'cat'])
    # print(response)

    response = OpenAIEmbeddings().embed_documents(['猫', 'cat'])
    print(response)
