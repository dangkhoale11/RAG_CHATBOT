from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


embeddings = download_hugging_face_embeddings()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("divachatbot")

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)


PROMPT=PromptTemplate(template=template_detail, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}




model_id = 'models/vinallama-7b-chat_q5_0.gguf'
llm = CTransformers(
    model=model_id,
    model_type="llama",  # hoặc "mistral", "phi2", v.v.
    max_new_tokens=128,
    temperature=0.01,
    gpu_layers=40  # Đặt số lớp chạy trên GPU
)


qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={'k': 1}),
    memory=ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5
    ),
    combine_docs_chain_kwargs=chain_type_kwargs,
)


# ==== ROUTES ====
@app.route("/")
def index():
    return render_template('chat.html')

chat_history = []

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"].lower().strip()
    print("User:", msg)
    result = qa_chain({"question": msg, "chat_history": chat_history})
    answer = result["answer"]
    chat_history.append((msg, answer))
    print("Response:", answer)
    return str(answer)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)