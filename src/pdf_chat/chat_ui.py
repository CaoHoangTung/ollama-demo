import streamlit as st
import validators
from streamlit_chat import message

from chat_agent import ChatPDF
from pdf_uploader import PDFIndexer

st.set_page_config(page_title="ChatPDF")


def display_messages():
    st.subheader("Chat about a PDF File")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].invoke(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def handle_file_upload():
    try:
        file = st.session_state["uploaded_file"]
        if not file:
            return
        st.toast("Indexing File")
        st.session_state["indexer"].upload_file(file)
        st.toast('Indexation complete')
    except Exception as err:
        st.error(err)


def page():
    st.header("Chat about your pdf files")
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()
        st.session_state["assistant"].setup_chain()
        st.session_state["indexer"] = PDFIndexer()

    st.divider()
    st.subheader("PDF File")

    st.file_uploader(
        "PDF File to index",
        type="pdf",
        key="uploaded_file",
        accept_multiple_files=False,
        on_change=handle_file_upload
    )

    with st.expander("Indexed File"):
        files = st.session_state["indexer"].list_files()
        st.table(files)

    st.divider()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()
