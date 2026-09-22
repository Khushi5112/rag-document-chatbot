"""
RAG Document Question Answering Chatbot - Streamlit App
---------------------------------------------------------
Upload a PDF or text document, then ask questions about it in plain English.
Uses LangChain + FAISS (retrieval) + Google Gemini (free tier LLM).

Run with:  streamlit run app.py
"""

import streamlit as st
import os
import tempfile

st.set_page_config(page_title="RAG Document Chatbot", page_icon="📄", layout="wide")

st.title("📄 RAG Document Question Answering Chatbot")
st.caption("Upload a document, then ask questions about it — answers are grounded in your document using Retrieval-Augmented Generation.")

# ---------- Step 1: API key ----------
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Gemini API Key", type="password", help="Get a free key at aistudio.google.com")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.divider()
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
    build_button = st.button("Process document", type="primary", disabled=not (api_key and uploaded_file))

# ---------- Session state ----------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ---------- Build the RAG pipeline when the user clicks "Process document" ----------
if build_button and uploaded_file and api_key:
    with st.spinner("Reading document, creating embeddings, and building the search index..."):
        try:
            from langchain_community.document_loaders import PyPDFLoader, TextLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
            from langchain_community.vectorstores import FAISS
            from langchain_core.prompts import ChatPromptTemplate

            # Save uploaded file to a temp path so the loaders can read it
            suffix = "." + uploaded_file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            if suffix.lower() == ".pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path)

            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vector_store = FAISS.from_documents(chunks, embeddings)

            llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            prompt = ChatPromptTemplate.from_template("""
Answer the question using only the following context from the document.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:
""")

            def ask_question(question, retriever=retriever, prompt=prompt, llm=llm):
                docs = retriever.invoke(question)
                context = "\n\n".join([d.page_content for d in docs])
                chain = prompt | llm
                response = chain.invoke({"context": context, "question": question})
                content = response.content
                if isinstance(content, list):
                    content = "".join(block.get("text", "") for block in content if isinstance(block, dict))
                return content, docs

            st.session_state.qa_chain = ask_question
            st.session_state.doc_name = uploaded_file.name
            st.session_state.messages = []

            os.unlink(tmp_path)
            st.success(f"'{uploaded_file.name}' processed — {len(chunks)} chunks indexed. Ask a question below!")
        except Exception as e:
            st.error(f"Something went wrong while processing the document: {e}")

# ---------- Chat interface ----------
if st.session_state.qa_chain is None:
    st.info("👈 Enter your Gemini API key, upload a document, and click 'Process document' to get started.")
else:
    st.caption(f"Currently chatting with: **{st.session_state.doc_name}**")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources used"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.write(f"**Chunk {i}:** {src[:300]}...")

    question = st.chat_input("Ask a question about your document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, source_docs = st.session_state.qa_chain(question)
                    sources = [doc.page_content for doc in source_docs]

                    st.write(answer)
                    if sources:
                        with st.expander("Sources used"):
                            for i, src in enumerate(sources, 1):
                                st.write(f"**Chunk {i}:** {src[:300]}...")

                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                except Exception as e:
                    st.error(f"Error getting an answer: {e}")

st.divider()
st.caption("Built with LangChain, FAISS, Google Gemini, and Streamlit.")
