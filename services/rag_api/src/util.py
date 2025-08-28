from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from typing import List

# 예시 데이터 (사용자 제공 RealDictRow와 유사한 형태)
results = [
    {'openalex_id': 'W3159727696', 'title': 'SpAtten: Efficient Sparse Attention Architecture...', 'abstract': 'This paper presents an efficient sparse attention architecture...', 'published': 2023, 'authors': 'Kim et al.'},
    {'openalex_id': 'W3159727697', 'title': 'Another Paper Title', 'abstract': 'A brief overview of the paper...', 'published': 2022, 'authors': 'Lee et al.'}
]

def convert_to_documents(results: List[dict]) -> List[Document]:
    """
    RealDictRow 리스트를 Document 리스트로 변환합니다.
    """
    documents = []
    for row in results:
        # title과 abstract를 결합하여 문서의 주요 내용(page_content)으로 사용
        page_content = f"{row.get('abstract', '')}"
        
        # 나머지 필드를 메타데이터로 저장
        metadata = {
            "openalex_id": row.get('openalex_id'),
            "title": row.get('title'),
            "published": row.get('published'),
            "authors": row.get('authors'),
            "cited_by_count": row.get('cited_by_count'),
            "pdf_url": row.get('pdf_url'),
        }
        
        # Document 객체 생성 및 리스트에 추가
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)
    
    return documents

def get_last_user_query(messages: List[BaseMessage]) -> str:
    """
    Get the last user query from the messages.
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""

def set_last_user_query(messages: List[BaseMessage], query: str) -> List[BaseMessage]:
    """
    Set the last user query to the messages.
    """
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            messages[i].content = query
            break
    return messages