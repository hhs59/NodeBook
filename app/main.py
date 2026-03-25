import sys
import os
import logging
import shutil
import streamlit as st
import streamlit.components.v1 as components



current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.controller import NodeBookController
from app.components.graph_visualizer import render_graph

st.set_page_config(
    page_title="NodeBook | GraphRAG Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { 
        background-color: #0e1117; 
        color: #ffffff; 
    }   
    
    .stChatMessage { 
        border-radius: 10px; 
        margin-bottom: 10px; 
    }
    
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    </style>
    """, unsafe_allow_html=True)

if 'controller' not in st.session_state:
    with st.spinner("Initializing AI Engines (REBEL, BART, Chroma)... This takes a moment."):
        st.session_state.controller = NodeBookController()
    st.toast("NodeBook Engine is Ready!", icon="🚀")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "relevant_triplets" not in st.session_state:
    st.session_state.relevant_triplets = []

with st.sidebar:
    st.title("NodeBook")
    st.caption("Graph-Augmented RAG Engine")
    st.markdown("---")
    
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents", use_container_width=True):
            for uploaded_file in uploaded_files:
                save_path = os.path.join("data", "raw", uploaded_file.name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner(f"Reading & Extracting Graph from {uploaded_file.name}..."):
                    st.session_state.controller.process_document(save_path)
            
            st.success("Indexing Complete! You can now chat.")

    st.markdown("---")
    
    if st.button("🗑️ Clear All Data & Restart", use_container_width=True):
        if os.path.exists("data/db/chroma"):
            shutil.rmtree("data/db/chroma")
        
        st.session_state.controller = NodeBookController()
        st.session_state.messages = []
        st.session_state.relevant_triplets = []
        st.success("All data cleared. Ready for a new session.")
        st.rerun()

col_chat, col_graph = st.columns([1, 1], gap="large")

with col_chat:
    st.subheader("💬 Notebook Intelligence")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your knowledge...", key="nodebook_chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and synthesizing answer..."):
                response = st.session_state.controller.ask_question(prompt)
                st.markdown(response['answer'])
                
                if response.get('answer_context'):
                    with st.expander("📚 View Source Citations"):
                        for i, chunk in enumerate(response['answer_context']):
                            st.info(f"**Source {i+1}:** {chunk[:400]}...")
                
                st.session_state.relevant_triplets = response.get('relevant_nodes', [])
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})
        
        st.rerun()

with col_graph:
    st.subheader("🕸️ Knowledge Map")
    
    triplets_to_show = []
    
    if st.session_state.relevant_triplets:
        triplets_to_show = st.session_state.relevant_triplets
        st.caption("✨ Showing specific nodes related to your last question.")
        if st.button("🔄 View Full Graph", use_container_width=True):
            st.session_state.relevant_triplets = []
            st.rerun()
    elif hasattr(st.session_state.controller, 'knowledge_graph'):
        triplets_to_show = st.session_state.controller.knowledge_graph
        st.caption("Global document connections. Ask a question to filter.")

    if triplets_to_show:
        graph_html = render_graph(triplets_to_show)
        components.html(graph_html, height=700, scrolling=True)
    else:
        st.info("Upload and process documents to generate the knowledge graph.")