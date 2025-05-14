from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredMarkdownLoader
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
OVERLAP_SIZE = 50

# RecursiveCharacterTextSplitter 将按不同的字符递归地分割(按照这个优先级["\n\n", "\n", " ", ""])，这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE
)

def load_pdf(file_path):
    return PyMuPDFLoader(file_path).load()

def load_markdown(file_path):
    return UnstructuredMarkdownLoader(file_path).load()

def split_text(text):
    return text_splitter.split_text(text)

def split_documents(documents):
    return text_splitter.split_documents(documents)

def load1():
    pdf_pages = load_pdf('./data_base/knowledge_db/pumkin_book/pumpkin_book.pdf')
    print(f"载入后的变量类型为：{type(pdf_pages)}，",  f"该 PDF 一共包含 {len(pdf_pages)} 页")
    pdf_page = pdf_pages[1]
    print(f"每一个元素的类型：{type(pdf_page)}.",
        f"该文档的描述性数据：{pdf_page.metadata}", 
        f"查看该文档的内容:\n{pdf_page.page_content}", 
        sep="\n------\n")

    pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
    pdf_page.page_content = pdf_page.page_content.replace('•', '')
    pdf_page.page_content = pdf_page.page_content.replace(' ', '')
    print(pdf_page.page_content)

def load2():
    md_pages = load_markdown('./data_base/knowledge_db/prompt_engineering/1. 简介 Introduction.md')
    print(f"载入后的变量类型为：{type(md_pages)}，",  f"该 Markdown 一共包含 {len(md_pages)} 页")
    md_page = md_pages[0]
    print(f"每一个元素的类型：{type(md_page)}.", 
        f"该文档的描述性数据：{md_page.metadata}", 
        f"查看该文档的内容:\n{md_page.page_content}", 
        sep="\n------\n")

    md_page.page_content = md_page.page_content.replace('\n\n', '\n')
    print(md_page.page_content)

def load3():
    pdf_pages = load_pdf('./data_base/knowledge_db/pumkin_book/pumpkin_book.pdf')
    split_docs = text_splitter.split_documents(pdf_pages)
    print(f"切分后的文件数量：{len(split_docs)}")
    print('split_docs[1].page_content\n', split_docs[1].page_content)
    print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")
    print('\n------\n')
    split_text = text_splitter.split_text(pdf_pages[1].page_content)
    print(f"切分后的文本数量：{len(split_text)}")
    print('split_text[1]\n', split_text[1])


if __name__ == '__main__':
    load3()
