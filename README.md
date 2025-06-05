### Retrival Chatbot for Diva beauty institute
This is a retrieval-based chatbot designed to assist users by answering questions about the Diva Beauty Institute. The chatbot utilizes document retrieval techniques to provide accurate and context-aware responses based on available institute documents.
---
# To implement and run
- Install dependencies
```bash
pip install -r requirements.txt
```
- Scrapping web text for embedding vector
```bash
python scrape_data.py
```
- Creat pinecone index and save api key in .env file

```bash
PINECONE_API_KEY=_____________
```
- Init pinecone vector db
```bash
python store_index.py
```
- Download model, creat a folder (models) and save model name from hugging_face
https://huggingface.co/vilm/vinallama-7b-chat-GGUF

- Run app
```bash
python app.py
```




