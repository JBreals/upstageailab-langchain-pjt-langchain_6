import gradio as gr
import time
from dotenv import load_dotenv
import os
import requests
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"), # 로그 파일을 지정
        logging.StreamHandler(sys.stdout) # 표준 출력으로도 보내기
    ]
)

load_dotenv(dotenv_path="../../../.env", override=True)
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

# --- Mock API Functions (향후 실제 rag-api 호출 코드로 대체) ---

def mock_phase1_api_call(paper_query: str) -> dict | None:
    """
    [Phase 1] 논문 검색 API 호출을 모의합니다. 'select_paper' 노드의 역할을 대신합니다.
    :param paper_query: 사용자가 검색한 논문 제목
    :return: DB에서 찾은 논문 정보 또는 None
    """
    print(f"📄[API Mock] Phase 1: '{paper_query}' 논문 검색 시도...")
    if "graph rag" in paper_query.lower():
        # DB에서 논문을 찾았다고 가정
        print("✅[API Mock] 'Graph RAG' 논문을 찾았습니다.")
        return {"title": "Graph RAG", "is_sbp": True, "details": "Graph RAG에 대한 상세 정보"}
    else:
        # DB에 논문이 없다고 가정
        print("❌[API Mock] 논문을 찾을 수 없습니다.")
        return None

def mock_phase2_api_call(user_prompt: str, sbp_title: str, chat_history: list) -> str:
    """
    [Phase 2] RAG 챗봇 API 호출을 모의합니다. 'retrieve_and_select' 부터 'generate_answer' 노드의 역할을 대신합니다.
    :param user_prompt: 사용자가 입력한 질문
    :param sbp_title: Phase 1에서 검색된 논문 제목
    :param chat_history: 현재까지의 대화 기록
    :return: LLM이 생성한 답변
    """
    print(f"🤖[API Mock] Phase 2: '{sbp_title}' 기반으로 '{user_prompt}'에 대한 답변 생성 시도...")
    time.sleep(1.5) # 실제 LLM이 답변을 생성하는 것처럼 보이게 하기 위함
    
    # 현재는 테스트 용 문자열.
    response = (
        f"'{sbp_title}' 논문을 기반으로 답변을 생성합니다.\n\n"
        f"후속 연구는 다음과 같습니다:\n"
        f"- 후속 논문 A (from Vector Store)\n"
        f"- 후속 논문 B (from Vector Store)\n"
        f"- 후속 논문 C (from DB)\n"
        f"- 후속 논문 D (from DB)"
    )
    
    # [수정됨] 답변을 누적하여 스트리밍
    streamed_response = ""
    for char in response:
        streamed_response += char
        yield streamed_response
        time.sleep(0.02)


# --- Gradio Event Handler Functions ---

def start_phase1(paper_query: str):
    """
    'Search' 버튼 클릭 시 실행되어 Phase 1을 시작하는 함수입니다.
    :param paper_query: 검색할 논문 제목
    :return: Phase 1 결과에 따라 업데이트될 UI 컴포넌트들의 값
    """
    try:
        response = requests.post(f"{RAG_API_URL}/start_phase1", json={"query": paper_query})
        logging.info(f"response: {response}")
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        result = response.json()
        logging.info(f"result: {result}")
        thread_id = result.get("thread_id")
        sbp_found = result.get("sbp_found")
        sbp_title = result.get("sbp_title")

        logging.info(f"thread_id: {thread_id} sbp_found: {sbp_found} sbp_title: {sbp_title}")
        
        if sbp_found:
            yield {
                thread_id_state: thread_id,
                searched_paper_state: sbp_title,
                searched_paper_output: gr.update(
                    value=f"✅ **Found Paper:** {sbp_title}\n\n이제 아래 채팅창에서 후속 연구에 대해 질문할 수 있습니다.", 
                    visible=True
                ),
                phase2_ui_container: gr.update(visible=True)
            }
        else:
          
            yield {
                thread_id_state: "",
                searched_paper_state: "",
                searched_paper_output: gr.update(
                    value=f"❌ **Paper Not Found:** '{paper_query}'\n\nDB에 해당 논문이 없습니다. 재검색합니다.", 
                    visible=True
                ),
                phase2_ui_container: gr.update(visible=False)
            }
            response = requests.post(f"{RAG_API_URL}/phase1_retry", json={"query": paper_query, "thread_id": thread_id})
            response.raise_for_status()
            result = response.json()
            thread_id = result.get("thread_id")
            sbp_found = result.get("sbp_found")
            sbp_title = result.get("sbp_title")

            yield {
                thread_id_state: thread_id,
                searched_paper_state: sbp_title,
                searched_paper_output: gr.update(
                    value=f"✅ **Found Paper:** {sbp_title}\n\n이제 아래 채팅창에서 후속 연구에 대해 질문할 수 있습니다.", 
                    visible=True
                ),
                phase2_ui_container: gr.update(visible=True)
            }
    except requests.exceptions.RequestException as e:
        error_message = f"API 호출 오류: {e}"
        print(error_message)
        yield {
            thread_id_state: "",
            searched_paper_state: "",
            searched_paper_output: gr.update(value=error_message, visible=True),
            phase2_ui_container: gr.update(visible=False)
        }

    # paper_info = mock_phase1_api_call(paper_query)
    
    # ### test를 위해 return True
    # sbp_title = "Attention is all you need."
    # return {
    #     searched_paper_state: sbp_title,
    #     searched_paper_output: gr.update(
    #         value=f"✅ **Found Paper:** {sbp_title}\n\n이제 아래 채팅창에서 후속 연구에 대해 질문할 수 있습니다.", 
    #         visible=True
    #     ),
    #     # [수정됨] Phase 2 전체 UI 컨테이너를 보이게 함
    #     phase2_ui_container: gr.update(visible=True)
    # }


def start_phase2(message: str, history: str, thread_id: str, sbp_title: str):
    """
    ChatInterface에서 채팅 입력 시 실행되어 Phase 2를 시작하는 함수입니다.
    :param message: 사용자가 입력한 메시지 (ChatInterface에 의해 자동으로 전달)
    :param history: 현재까지의 대화 기록 (ChatInterface에 의해 자동으로 전달)
    :param sbp_title: Phase 1에서 검색되어 'searched_paper_state'에 저장된 논문 제목
    :return: 챗봇의 스트리밍 응답
    """
    if not thread_id or not sbp_title:
        return "오류: 먼저 논문을 검색해주세요."
    
    try: 
        print(f"Sending data: {{'thread_id': {thread_id}, 'question': {message}, 'sbp_title': {sbp_title}}}")
        response = requests.post(
            f"{RAG_API_URL}/start_phase2",
            json={"thread_id": thread_id, "question": message, "sbp_title": sbp_title},
            stream=True
        )
        print(f"Response: {response}")
        response.raise_for_status()
        
        # for문을 통해 스트림 응답을 처리합니다.
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data:'):
                    try:
                        data_str = decoded_line[len('data:'):]
                        data = json.loads(data_str)
                        # RAG 서버의 스트리밍 청크를 직접 Gradio에 yield합니다.
                        answer_chunk = data.get("answer_chunk", "")
                        yield answer_chunk
                    except json.JSONDecodeError:
                        print(f"JSON 디코딩 오류: {decoded_line}")
                        continue
    except requests.exceptions.RequestException as e:
        yield f"API 호출 오류: {e}"


# --- Gradio UI Layout ---

with gr.Blocks(theme=gr.themes.Soft(), title="Paper Follow-up Researcher") as demo:
    thread_id_state = gr.State("")
    searched_paper_state = gr.State("")

    gr.Markdown("# 📄 Paper Follow-up Researcher")
    gr.Markdown("관심 있는 논문을 검색하고, 해당 논문의 주요 후속 연구들을 RAG를 통해 탐색하세요.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Phase 1: Search 'From Paper'")
            paper_search_input = gr.Textbox(
                label="Enter Paper Title", 
                placeholder="e.g., Graph RAG",
                info="탐색을 시작할 기준 논문의 제목을 입력하세요."
            )
            search_button = gr.Button("🔍 Search Paper")
            
            searched_paper_output = gr.Markdown(visible=False)

        with gr.Column(scale=2):
            gr.Markdown("### Phase 2: Explore Follow-up Papers")
            
            # [수정됨] Phase 2 UI 요소들을 감싸는 컨테이너 추가
            with gr.Column(visible=False) as phase2_ui_container:
                # [수정됨] ChatInterface에서 visible, interactive 인자 제거
                chat_interface = gr.ChatInterface(
                    fn=start_phase2,
                    additional_inputs=[thread_id_state, searched_paper_state],
                    type='messages',
                )
                example_prompts = gr.Examples(
                    examples=[
                        "이 논문의 주요 후속 연구들은 무엇이야?",
                        "기술적으로 가장 큰 영향을 준 후속 논문 3개를 알려줘.",
                        "이 연구의 단점을 보완한 후속 연구가 있을까?",
                    ],
                    inputs=chat_interface.textbox,
                    label="Example Prompts",
                )
    
    # --- Component Event Listeners ---
    
    search_button.click(
        fn=start_phase1,
        inputs=[paper_search_input],
        # [수정됨] outputs에 phase2_ui_container 추가
        outputs=[
            thread_id_state,
            searched_paper_state, 
            searched_paper_output, 
            phase2_ui_container
        ]
    )


if __name__ == "__main__":
    demo.launch()
