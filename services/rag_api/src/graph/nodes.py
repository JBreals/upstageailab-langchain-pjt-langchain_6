from dotenv import load_dotenv
import os

from .state import GraphState
from ..core.database import mock_db_select, mock_db_insert, mock_db_follow_up_select
from ..core.source_api import openalex_search
from ..core.retriever import UPSTAGE_API_KEY, TAVILY_SEARCH, augment_prompt
from ..core.llm import mock_llm_generate, rag_judge, mock_llm_generate_no_rag
from ..core.get_emb import get_emb_model, get_emb
from langgraph.types import interrupt
from ..util import convert_to_documents

load_dotenv()

def select_paper_node(state: GraphState):
    """
    사용자가 읽은 논문에 대한 키워드가 state['from_paper']로 전달된다.
    1. DB에서 해당 논문이 존재하는지 찾는다.
    2. 논문이 존재한다면 해당 논문의 full title을 state['sbp_title']에 저장하고 state['sbp_found]
    
    :param GraphState state: The current graph state. 
    :return dict: New state with DB query result.
    """
    print("\n--- 노드 실행: select_paper_node ---")
    query = state["initial_query"]

    paper_info = mock_db_select(query)
    print(f"paper_info: {paper_info}")

    if not state.get("is_chat_mode"):
        value = interrupt({"paper_info": paper_info})

        if paper_info and paper_info["is_sbp"]:
            return {"sbp_found": True, "sbp_title": paper_info["paper_meta"]["title"], "paper_search_result": paper_info["paper_meta"]}
        else:
            return {"sbp_found": False, "sbp_title": ""}
    else:
        return {"paper_search_result": paper_info["paper_meta"]}

    # if paper_info and paper_info["is_sbp"]:
    #     return {"sbp_found": True, "sbp_title": paper_info["paper_meta"]["title"], "paper_search_result": paper_info["paper_meta"]}
    # else:
    #     return {"sbp_found": False, "sbp_title": ""}

def web_search_node(state: GraphState):
    """
    :param state: The current graph state. 
    :return: OpenAlex 검색 결과를 state에 저장하고 논문 제목으로 초기 쿼리 업데이트
    """

    print("\n--- 노드 실행: web_search_node ---")
    query = state["initial_query"]
    paper_search_result = openalex_search(query)

    return {"paper_search_result": paper_search_result,
            "initial_query": paper_search_result["title"]} # 무한 루프 방지 위해 우선 논문 제목으로 초기 쿼리 업데이트 추후 수정 필요

def insert_paper_node(state: GraphState):
    """
    :param state: The current graph state. 
    :return: 논문 정보를 DB에 저장
    """
    print("\n--- 노드 실행: insert_paper_node ---")
    paper_info = state["paper_search_result"]

    # 논문 초록 임베딩
    emb_model = get_emb_model()
    embedding = get_emb(emb_model, [paper_info["abstract"]])
    paper_info["embedding"] = embedding[0]

    if paper_info:
        mock_db_insert(paper_info)

    return {}

def rag_judge_node(state: GraphState):
    """
    :param state: The current graph state. 
    :return: 논문 초록 임베딩과 사용자 질문 임베딩 간 유사도를 계산하여 임계값을 넘으면 논문 초록 임베딩을 사용하고, 넘지 않으면 사용자 질문 임베딩을 사용한다.
    """
    print("\n--- 노드 실행: rag_judge_node ---")
    question = get_last_user_query(state["messages"])
    judgement = rag_judge(question, os.getenv("UPSTAGE_API_KEY"))
    return {"rag_judgement": judgement}
    
def retrieve_and_select_node(state: GraphState):
    """:param state: The current graph state. :return: New state with retrieved documents."""
    print("\n--- 노드 실행: retrieve_and_select_node ---")
    last_user_query = get_last_user_query(state["messages"])
    use_prompt_augment = True
    if use_prompt_augment:
        ### prompt에서 키워드를 찾아 tavily search로 증강하고, 영어로 번역하는 함수.
        last_user_query = augment_prompt(last_user_query, UPSTAGE_API_KEY, TAVILY_SEARCH)

    paper_info = state["paper_search_result"]
    query_vec = get_emb(get_emb_model(), [state["question"]])[0]
    k = 5
    db_follow_up_docs = mock_db_follow_up_select(paper_info, query_vec, k)

    all_docs = convert_to_documents(db_follow_up_docs)
    return {"retrieved_docs": all_docs, "messages": set_last_user_query(state["messages"], last_user_query)}

def generate_answer_node(state: GraphState):
    """:param state: The current graph state. :return: New state with the final answer."""
    print("\n--- 노드 실행: generate_answer_node ---")
    question = state["question"]
    
    print(f"\n\nstate['rag_judgement']: {state['rag_judgement']}\n\n")

    if state["rag_judgement"] == "RAG":
        context = state["retrieved_docs"]
        answer = mock_llm_generate(question, context, llm_api_key = os.getenv("UPSTAGE_API_KEY"))
    else:
        answer = mock_llm_generate_no_rag(question, llm_api_key = os.getenv("UPSTAGE_API_KEY"))
    return {"answer": answer}

def should_search_web(state: GraphState) -> str:
    """:param state: The current graph state. :return: The name of the next node to call."""
    print("\n--- 조건 분기: should_search_web ---")
    if state["sbp_found"]:
        print("✅ SBP 발견! Phase 2로 진행합니다.")
        return "rag_judge"
    else:
        print("❌ SBP 미발견. 웹 검색을 시작합니다.")
        return "web_search"

def rag_condition(state: GraphState) -> str:
    """:param state: The current graph state. :return: The name of the next node to call."""
    print("\n--- 조건 분기: rag_condition ---")
    if state["rag_judgement"] == "RAG":
        return "retrieve_and_select"
    else:
        return "generate_answer"