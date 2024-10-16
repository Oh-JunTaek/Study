from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

def load_md_files(file_path): 
    loader = TextLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from the MD.")
    return documents

def split_docs(documents):
    assert len(documents) == 1  # 다수 문서일 경우 수정 필요
    assert isinstance(documents[0], Document)
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    splitted_md = markdown_splitter.split_text(documents[0].page_content)
    return splitted_md
