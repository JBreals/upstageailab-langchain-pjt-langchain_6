from concurrent.futures import thread
import uuid
import os
import sys
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", "..", ".."))
sys.path.append(ROOT_DIR)

from services.rag_api.src.graph.builder import build_graph
# LangGraph app 빌드
app_builder = build_graph()
# FastAPI app 생성
app = FastAPI(
    title='RAG API Server',
    description="LangGraph 기반 RAG 워크플로우 제어 서버"
)
class Phase1Request(BaseModel):
    query: str
    thread_id: Optional[str] = None
class Phase2Request(BaseModel):
    thread_id: str
    question: str
    sbp_title: str
    history: str

@app.get("/")
async def root():
    return {"message": "RAG API Server is running"}

@app.post("/start_phase1")
async def start_phase1(request: Phase1Request):
    """Phase 1 워크플로우를 시작하고, 논문 검색 결과와 함께 thread_id를 반환합니다.
    그래프는 select_paper 노드 이후 잠시 중단됩니다.

    :param Phase1Request request: input
    """
    print(f"\n start_phase1 호출")
    # 새로운 대화를 위한 고유 ID 생성
    thread_id = str(uuid.uuid4())
    config = RunnableConfig(configurable={"thread_id":thread_id})
    inputs = {"initial_query":request.query, "thread_id": thread_id, "is_chat_mode": False}

    # .astream()을 사용하여 그래프를 실행하고 중단점까지의 결과를 수집
    for event in app_builder.stream(inputs, config = config):
        if event.get("__interrupt__", False):
            interrupt_obj = event['__interrupt__'][0]
            value_dict = interrupt_obj.value
            paper_info = value_dict.get("paper_info", None)
    
    if paper_info is None:
        return {
            'thread_id': thread_id,
            'sbp_found': False,
            'sbp_title': "Paper Not Found.",
        }
    else:
        paper_meta = paper_info.get("paper_meta", None)
        sbp_title = paper_meta.get("title", None)
        sbp_found = True
        return {
          'thread_id': thread_id,
          'sbp_found': sbp_found,
          'sbp_title': sbp_title,
        }
    
    # state = app_builder.get_state(config)
    # final_state = state.values      
    # print(f"values: {final_state}")
    # # state 에서 필요한 정보를 추출하여 클라이언트에 응답 > select_paper node 종료 이후의 GraphState를 가져온다.
    
    # return {
    #     'thread_id': thread_id,
    #     'sbp_found': final_state.get("sbp_found", False),
    #     'sbp_title': final_state.get('sbp_title', "Paper Not Found."),
    # }

@app.post("/phase1_retry")
async def phase1_retry(request: Phase1Request):
    """Phase 1 워크플로우를 재개하고, 논문 재검색 노드를 진행합니다.

    :param Phase1Request request: input
    """
    print(f"\n phase1_retry 호출")
    print(f"\nthread_id: {request.thread_id}")
    config = RunnableConfig(configurable={"thread_id":request.thread_id})
    
    resume = Command(resume={"retry": True})

    for event in app_builder.stream(resume, config):
        if event.get("__interrupt__", False):
            interrupt_obj = event['__interrupt__'][0]
            value_dict = interrupt_obj.value
            paper_info = value_dict.get("paper_info", None)
            print(f"value_dict: {value_dict}")
    
    if paper_info is None:
        return {
            'thread_id': request.thread_id,
            'sbp_found': False,
            'sbp_title': "Paper Not Found.",
        }
    else:
        paper_meta = paper_info.get("paper_meta", None)
        sbp_title = paper_meta.get("title", None)
        sbp_found = True
        return {
          'thread_id': request.thread_id,
          'sbp_found': sbp_found,
          'sbp_title': sbp_title,
        }

    # state = app_builder.get_state(config)
    # final_state = state.values      

    # return {
    #     'thread_id': request.thread_id,
    #     'sbp_found': final_state.get("sbp_found", False),
    #     'sbp_title': final_state.get('sbp_title', "Paper Not Found."),
    # }

@app.post("/start_phase2")
async def start_phase2(request: Phase2Request):
    """중단된 워크플로우를 이어받아 Phase 2를 실행하고, 최종 답변을 스트리밍으로 반환합니다.

    :param Phase2Request request: question(=message), thread_id, sbp_title 을 전달받는다.
    """
    print(f"\n start_phase2 호출\n")
    config = RunnableConfig(configurable={"thread_id": request.thread_id})

    async def stream_generator():
        # .stream()을 None 입력으로 호출하여 중단된 지점부터 실행 재개

        inputs = {
          "initial_query": request.sbp_title,
          "question": request.question,
          "sbp_title": request.sbp_title,
          "is_chat_mode": True,
          "sbp_found": True,
          "thread_id": request.thread_id,
        }
        for event in app_builder.stream(
            inputs, config, stream_mode="updates"
        ):
            if "generate_answer" in event:
                answer_chunk = event["generate_answer"]["answer"]
                # JSON 스트림 형식으로 데이터를 전송
                yield f"data: {json.dumps({'answer_chunk': answer_chunk})}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # uvicorn rag-api.src:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
