# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a simple standalone implementation showing rag pipeline using Nvidia AI Foundational models.
# It uses a simple Streamlit UI and one file implementation of a minimalistic RAG pipeline.

############################################
# Document Loader
############################################

import streamlit as st
import os
import csv

from datetime import datetime

st.set_page_config(layout = "wide")

import argparse
parser = argparse.ArgumentParser(description="Command line options")
parser.add_argument("runlocal")
args = parser.parse_args()
runs_local = args.runlocal and args.runlocal == "local"

from google.cloud import firestore

# Accessing Firestore
if runs_local:
    st.write("RUNNING LOCALLY")
    # Authenticate to Firestore with the JSON account key.
    db = firestore.Client.from_service_account_json("firestore-key.json")
else:
    st.write("RUNNING ON WEB")
    # Authenticate to Firestore with the project ID.
    db = firestore.Client(project="muia-faq")

USE_SIDEBAR = False # The sidebar is for developing purposes only
MODEL = "mixtral_8x7b" # "ai-llama2-70b"

DOCS_DIR = os.path.abspath("./uploaded_docs")
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

if USE_SIDEBAR:
    with st.sidebar:
        st.subheader("Add to the Knowledge Base")
        with st.form("my-form", clear_on_submit=True):
            uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:", accept_multiple_files = True)
            submitted = st.form_submit_button("Upload!")
        if uploaded_files and submitted:
            for uploaded_file in uploaded_files:
                st.success(f"File {uploaded_file.name} uploaded successfully!")
                with open(os.path.join(DOCS_DIR, uploaded_file.name),"wb") as f:
                    f.write(uploaded_file.read())

############################################
# Embedding Model and LLM
############################################

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# make sure to export your NVIDIA AI Playground key as NVIDIA_API_KEY! In fact, the key is hardwired here...
#os.environ["NVIDIA_API_KEY"] = "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"

document_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="passage")
query_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="query")

############################################
# User Feedback
############################################

# TO-DO: add a "was it useful" button such that the content of the last question and answer is stored automatically

def input_callback():
    with st.sidebar:
        st.success("¡Gracias!")
with st.sidebar:
    improvements = st.text_input(label="¿Hay algo que el bot no haya sido capaz de contestar? ¡Ayúdanos a mejorarlo!",
                                 value="",
                                 on_change=input_callback)
    time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {"Date": time_stamp,
            "Text": improvements }
    db.collection("improvements").document(time_stamp).set(data)

def storeQuery(good_or_bad):
    time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {"Date": time_stamp,
            "Question": user_input,
            "Answer": full_response }
    db.collection(f"{good_or_bad}Queries").document(time_stamp).set(data)
    st.success("¡Gracias por la retroalimentación! No dudes en hacerme otras preguntas")

def storeGood():
    storeQuery("good")

def storeBad():
    storeQuery("bad")


############################################
# Vector Database Store
############################################

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
import pickle

if USE_SIDEBAR:
    with st.sidebar:
        # Option for using an existing vector store
        use_existing_vector_store = st.radio("Use existing vector store if available", [True, False], horizontal=True)
else:
    use_existing_vector_store = True

# Path to the vector store file
vector_store_path = "vectorstore.pkl"

# Load raw documents from the directory
raw_documents = DirectoryLoader(DOCS_DIR).load()


# Check for existing vector store file
vector_store_exists = os.path.exists(vector_store_path)
vectorstore = None
if use_existing_vector_store and vector_store_exists:
    with open(vector_store_path, "rb") as f:
        vectorstore = pickle.load(f)
    if USE_SIDEBAR:
        with st.sidebar:
            st.success("Existing vector store loaded successfully.")
else:
    if USE_SIDEBAR:
        with st.sidebar:
            if raw_documents:
                with st.spinner("Splitting documents into chunks..."):
                    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                    documents = text_splitter.split_documents(raw_documents)

                with st.spinner("Adding document chunks to vector database..."):
                    vectorstore = FAISS.from_documents(documents, document_embedder)

                with st.spinner("Saving vector store"):
                    with open(vector_store_path, "wb") as f:
                        pickle.dump(vectorstore, f)
                st.success("Vector store created and saved.")
            else:
                st.warning("No documents available to process!", icon="⚠️")
    else:
        if raw_documents:
            text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            documents = text_splitter.split_documents(raw_documents)
            vectorstore = FAISS.from_documents(documents, document_embedder)
            with open(vector_store_path, "wb") as f:
                pickle.dump(vectorstore, f)

############################################
# LLM Response Generation and Chat
############################################

st.subheader("MUIA - FAQs - v0.2")

st.write("Esta aplicación pretende ser una ayuda para que puedas solucionar tus dudas sin pasar necesariamente por los emails de contacto (que siguen estando, por supuesto...)")
st.write("**DISCLAIMER**: la información que recibirás de este bot ha sido generada usando un **LLM** (Large Language Model) con **RAG** (Retrieval-Augmented Generation); en ningún caso ha de ser tomada como definitiva.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages(
    [("system", "Te llamas MUIAbot, y eres un asistente basado en IA. Siempre contestarás a las preguntas en Español y solo basándote en el contexto. Si alguna preguntas está fuera de contexto, dirás amablemente que no puedes contestar."), ("user", "{input}")]
)
user_input = st.chat_input("Escribe aquí cualquier pregunta sobre el MUIA. Intenta ser preciso.")
llm = ChatNVIDIA(model=MODEL)

chain = prompt_template | llm | StrOutputParser()

if user_input and vectorstore!=None:
    st.session_state.messages.append({"role": "user", "content": user_input})
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    context = ""
    for doc in docs:
        context += doc.page_content + "\n\n"

    augmented_user_input = "Context: " + context + "\n\nQuestion: " + user_input + "\n"

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for response in chain.stream({"input": augmented_user_input}):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    _, col2, col3, _ = st.columns(4)
    with col2:
        st.button(label="Me has sido útil",on_click=storeGood)
    with col3:
        st.button(label="No has contestado correctamente",on_click=storeBad)