from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_FILE = Path(__file__).resolve().with_name("QA.csv")
MIN_SIMILARITY = 0.14
FALLBACK = (
    "知識庫中沒有找到足夠相近的問答。請換個方式描述，或補充學校、流程等關鍵資訊。"
    "若涉及學生安全或緊急事件，請立即聯繫學校及主管機關。"
)


@st.cache_data
def load_knowledge_base(path: str, modified: float) -> tuple[tuple[str, str], ...]:
    del modified  # Included in the cache key so edits to QA.csv reload the data.
    import csv

    rows: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"question", "answer"}.issubset(reader.fieldnames):
            raise ValueError("QA.csv 必須包含 question 與 answer 欄位。")
        for row in reader:
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if question and answer:
                rows.append((question, answer))
    if not rows:
        raise ValueError("QA.csv 沒有可用的問答資料。")
    return tuple(rows)


@st.cache_resource
def build_index(questions: tuple[str, ...]):
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 4), lowercase=True, sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(questions)
    return vectorizer, matrix


def find_answer(question: str, entries: tuple[tuple[str, str], ...], vectorizer, matrix):
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix).ravel()
    best_index = int(scores.argmax())
    score = float(scores[best_index])
    if score < MIN_SIMILARITY:
        return FALLBACK, None, score
    return entries[best_index][1], entries[best_index][0], score


st.set_page_config(page_title="教育與校事會議線上客服", page_icon="🎓")
st.title("🎓 教育與校事會議線上客服")
st.caption("輸入問題後，系統會從 QA.csv 檢索最相近的既有問答。回覆為資料庫內容，不是生成式回答。")

try:
    knowledge = load_knowledge_base(str(DATA_FILE), DATA_FILE.stat().st_mtime)
    vectorizer, matrix = build_index(tuple(question for question, _ in knowledge))
except (OSError, ValueError) as error:
    st.error(f"無法載入問答資料：{error}")
    st.stop()

st.success(f"問答資料已載入，共 {len(knowledge)} 組。您可以在下方輸入問題，或點選範例開始。")

with st.sidebar:
    st.subheader("常見問題")
    examples = [question for question, _ in knowledge[:5]]
    for index, example in enumerate(examples):
        if st.button(example, key=f"example_{index}", use_container_width=True):
            st.session_state.pending_question = example
            st.rerun()
    if st.button("清除對話", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("您好！我可以協助查詢教育相關服務及校事會議常見問題。請輸入您的問題，或選擇下方範例。")
    st.markdown("**試試這些常見問題**")
    suggestion_columns = st.columns(3)
    for index, example in enumerate(question for question, _ in knowledge[:3]):
        if suggestion_columns[index].button(example, key=f"main_example_{index}", width="stretch"):
            st.session_state.pending_question = example
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("reference"):
            st.caption(f"參考問答：{message['reference']}（文字相似度 {message['score']:.0%}）")

prompt = st.session_state.pop("pending_question", None) or st.chat_input("請輸入您的問題")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    answer, reference, score = find_answer(prompt, knowledge, vectorizer, matrix)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "reference": reference, "score": score}
    )
    st.rerun()
