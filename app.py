import os
import streamlit as st
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = " API KEY "

st.title("📄 PDF RAG Chatbot (MapReduce, Hugging Face)")

pdf_file = st.file_uploader("Bir PDF yükleyin", type="pdf")

if pdf_file:
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    # PDF yükleme
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # Metni küçük parçalara ayır
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)


    # Embeddings ve FAISS veritabanı
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    # Hugging Face LLM pipeline
    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-small",
        max_length=150,
        do_sample=False
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # Map prompt (her parça için)
    map_prompt = PromptTemplate(
        template="""
Parçayı oku ve soruyu kısa ve net cevapla.
Eğer cevap metinde yoksa 'Bilmiyorum' yaz.

Parça:
{context}

Soru: {question}
Cevap:
""",
        input_variables=["context", "question"]
    )

    # Reduce prompt (tüm parçaların özetleri için)
    combine_prompt = PromptTemplate(
        template="""
Aşağıdaki özetleri kullanarak soruyu kısa ve net şekilde yanıtla.
Eğer cevap yoksa 'Bilmiyorum' yaz.

Özetler:
{summaries}

Soru: {question}
Cevap:
""",
        input_variables=["summaries", "question"]
    )

    # map_reduce zinciri oluştur
    qa_chain = load_qa_chain(
        llm=llm,
        chain_type="map_reduce",
        question_prompt=map_prompt,
        combine_prompt=combine_prompt
    )

    # RetrievalQA zinciri
    qa = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=qa_chain
    )

    # Kullanıcı sorusu
    query = st.text_input("Sorunuzu yazın:")

    if query:
        answer = qa.run(query)
        st.write("🤖 Cevap:", answer)
