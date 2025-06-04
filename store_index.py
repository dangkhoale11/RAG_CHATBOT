from src.helper import load_data, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()
text = load_data("./data")
text_chunks = text_split(text)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("divachatbot")
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)
vectorstore.add_texts(text_chunks)
