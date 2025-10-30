"""
rag_core.py
Prototype of a Retrieval-Augmented Generation (RAG) system for scientific PDFs.
Author: Nazareno Campioni
""" 

import os
import json
import numpy as np
import faiss
from itertools import chain
from pypdf import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
from openai import OpenAI

nltk.download("punkt")

os.environ["OPENAI_API_KEY"] = "enter_your_key"
openai.api_key = os.environ["OPENAI_API_KEY"]

def chunk_text(book, chunk_size=500, overlap=100):
    reader = PdfReader(book)
    chunks = []
    id = 1
    for i in range(len(reader.pages)):
      start = 0
      text = reader.pages[i].extract_text()
      text_length = len(text)
      while start < text_length:
          end = start + chunk_size
          chunk = text[start:end]
          chunks.append({"id": id, "source": book, "page": i+1, "text": chunk})
          start += chunk_size - overlap
          id += 1
    return chunks

# --- Tokenize into sentences ---
def tokenize(chunks): 
  sentences = []
  for i in range(len(chunks)):
    sentences.append(sent_tokenize(chunks[i]['text']))
  return list(chain(*sentences))
  
  print("Sentences:", sentences)
def build_index(sentences):
    print("Creating embeddings for sentences...")
    embeddings = np.vstack([get_embedding(s) for s in sentences])
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"Added {index.ntotal} vectors to FAISS index.")
    return index, embeddings


# --- Generate embeddings ---
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

# embeddings = [get_embedding(s) for s in sentences]
# embeddings = np.vstack(embeddings)

# --- Store in FAISS index ---
def store_faiss(embeddings):   
  dim = embeddings.shape[1]
  index = faiss.IndexFlatL2(dim)
  index.add(embeddings)
  print(f"Added {index.ntotal} vectors to FAISS")

# --- Generate answer ---
def generate_answer(query, sentences, k):
    """
    query: the user question
    sentences: tokenized sentences
    k: top k results from query search
    """
    query_emb = get_embedding(query).reshape(1, -1)
    D, I = index.search(query_emb, k)

    retrieved_text = list(sentences[i] for i in I.flatten())
  
    prompt = f"""
    You are a helpful librarian. Answer the following question using ONLY the information provided below.
    If the information is not in the context, say you don't know. Your personality is warm, grounded and secure.
    You make people feel at ease with your replies, but you don't actively comfort them.

    Context:
    {retrieved_text}

    Question:
    {query}

    Answer:
    """

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

