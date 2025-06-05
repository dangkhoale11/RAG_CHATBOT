from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms import LlamaCpp
from dotenv import load_dotenv
from src.prompt import *
import os
import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

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
llm = LlamaCpp(
    model_path=model_id,
    n_gpu_layers=-1,  # Use -1 to offload all layers to GPU automatically
    n_ctx=2048,       # Context window
    temperature=0.3,   # Temperature for randomness
    max_tokens=128,    # Maximum tokens to generate
    verbose=True,      # Enable verbose logging
    n_batch=512,       # Batch size for prompt processing (GPU optimization)
    n_threads=8,       # Number of threads for CPU processing
    use_mmap=True,     # Use memory mapping for faster loading
    use_mlock=True,    # Lock model in memory to prevent swapping
    seed=-1,           # Random seed (-1 for random)
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