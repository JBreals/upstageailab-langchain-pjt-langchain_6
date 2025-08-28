from importlib import metadata
from typing import List
import re
from collections import defaultdict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_upstage import ChatUpstage
from langchain_core.prompts import MessagesPlaceholder

def format_context(context: List[Document]) -> str:
    """
    LangChain Document 리스트를 LLM 프롬프트에 적합한 단일 문자열로 변환합니다.
    
    :param context: 'title'을 metadata에 포함하는 Document 객체 리스트
    :return: 각 논문 정보가 포함된 전체 문자열
    """
    context_parts = []
    for i, doc in enumerate(context):
        # doc.metadata에서 'title'을, doc.page_content에서 초록을 가져옵니다.
        title = doc.metadata.get('title', 'No Title Provided')
        abstract = doc.page_content
        context_parts.append(f"title: {title}\nAbstract: {abstract}\n------------------\n")
    
    return "\n\n".join(context_parts)

def mock_llm_generate(question: str, context: List[Document], llm_api_key: str) -> str:
    """
    검색된 문서를 바탕으로 최종 답변을 생성하는 LLM 함수.
    논문들을 분석하여 구조화된 답변을 생성합니다.
    1. Document에는 title(논문 제목)과 content(논문 abstract 내용)가 있다.
    2. context의 길이가 0이면 "검색된 후속논문이 없다"는 내용의 답변을 반환하여라. (LLM 사용 금지)
    3. format_context 함수를 통해 context의 각 Document들을 `title: 논문 제목, abstract: 논문 abstract 내용` 으로 재구성하여 context_str 에 저장하여라.
    4. prompt를 사용하여 LangChain 문법에 따라 LLM(openai GPT-3.5 Turbo)으로부터 답변을 생성하여라.
    
    :param str question: 사용자가 입력한 프롬프트
    :param List[Document] context: 검색된 후속 연구 논문들의 리스트
    :param llm_api_key: OpenAI API 키
    :return str: 구조화된 답변 문자열
    """
    print("🤖 LLM 답변 생성 중...")
    # 2. context 리스트가 비어있는지 확인합니다.
    if not context:
        print("ℹ️ 컨텍스트가 비어있어 LLM을 호출하지 않고 기본 메시지를 반환합니다.")
        return "검색된 후속 논문이 없습니다. 다른 키워드로 검색해 보세요."
    
    # 3. context를 프롬프트에 넣기 좋은 단일 문자열로 formatting한다.
    context_str = format_context(context)

    # 4. LangChain 프롬프트 템플릿을 정의합니다.
    prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a world-class AI research assistant. Your primary goal is to provide a clear, structured, and insightful analysis of academic papers based on their titles and abstracts.
You must synthesize information from multiple sources and present it in an easy-to-digest format for researchers.
Your response must be well-organized, using Markdown for headings and lists.
"""
        ),
        (
            "human",
            """Please read the user's question and the provided follow-up papers.
**If you determine that the provided papers are relevant to the user's question, use them as your primary source for the answer. If not, answer the question based on your own knowledge and the conversation history.**

**User's Question:**
<question>
{question}
</question>

**Provided Context (Follow-up Papers):**
<context>
{context_str}
</context>

**Your Task:**
Directly answer the user's question. Synthesize the key findings from all papers in the context to support your answer. Provide a brief summary in one or two lines for each papers. Focus on what is necessary to answer the question and finally please recommend the best one to read first out of context papers.

**IMPORTANT**: Your final output and all content must be written in **Korean** except for title of the paper.
"""
        ),
    ]
)

    # LLM 모델을 초기화합니다. (GPT-3.5 Turbo 사용)
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=llm_api_key, temperature=0.2)
    llm = ChatUpstage(model="solar-pro2", api_key=llm_api_key)
    # LangChain Expression Language (LCEL)을 사용하여 체인을 구성합니다.
    # 1. 프롬프트 포맷팅 -> 2. LLM 호출 -> 3. 출력 파싱(문자열로)
    chain = prompt_template | llm | StrOutputParser()
    
    # 체인을 실행하여 답변을 생성합니다.
    answer = chain.invoke({
        "question": question,
        "context_str": context_str
    })
    print(f"\n\nanswer: {answer}\n\n")
    
    return answer

def mock_llm_generate_no_rag(question: str, llm_api_key: str) -> str:
    """
    RAG가 필요하지 않은 경우 사용하는 LLM 함수.
    LLM의 기반지식과 대화 내역들을 이용하여 답변  
    
    :param str question: 사용자가 입력한 프롬프트
    :param llm_api_key: OpenAI API 키
    :return str: 답변 문자열
    """
    print("🤖 LLM 답변 생성 중...")
    
    # 4. LangChain 프롬프트 템플릿을 정의합니다.
    prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a world-class AI research assistant. Your primary goal is to provide a clear, structured, and insightful analysis of academic papers based on their titles and abstracts.
You must synthesize information from multiple sources and present it in an easy-to-digest format for researchers.
Your response must be well-organized, using Markdown for headings and lists.
"""
        ),
        (
            "human",
            """
**User's Question:**
<question>
{question}
</question>

**Your Task:**
Directly answer the user's question.

**IMPORTANT**: Your final output and all content must be written in **Korean** except for title of the paper.
"""
        ),
    ]
)

    # LLM 모델을 초기화합니다. (GPT-3.5 Turbo 사용)
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=llm_api_key, temperature=0.2)
    llm = ChatUpstage(model="solar-pro2", api_key=llm_api_key)
    # LangChain Expression Language (LCEL)을 사용하여 체인을 구성합니다.
    # 1. 프롬프트 포맷팅 -> 2. LLM 호출 -> 3. 출력 파싱(문자열로)
    chain = prompt_template | llm | StrOutputParser()
    
    # 체인을 실행하여 답변을 생성합니다.
    answer = chain.invoke({
        "question": question
    })
    print(f"\n\nanswer: {answer}\n\n")
    
    return answer


def rag_judge(question: str, llm_api_key: str) -> str:
    """
    사용자의 쿼리를 분석하여 RAG가 필요한지 판단
    """

    prompt_template = ChatPromptTemplate.from_messages(
    [
        (
                "system",
                """You are an AI designed to determine whether a user's question requires external knowledge retrieval (RAG).

Your judgment must be based on the following criteria:
- 'RAG': The question asks for information on specific papers, data, recent research, technical terms, or facts that are not common knowledge.
- 'NO_RAG': The question is about general knowledge, greetings, personal feelings, or a summary of previous conversation content.

Your response must be exclusively 'RAG' or 'NO_RAG'. Do not include any additional explanations or text."""
            ),
            (
              "human",
              """Question: {question}

              Please judge whether the question requires external knowledge retrieval (RAG).
              """
            )
    ])

    llm = ChatUpstage(model="solar-pro2", api_key=llm_api_key)
    chain = prompt_template | llm | StrOutputParser()

    judgement = chain.invoke({
        "question": question
    })

    return judgement

