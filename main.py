import os
from openai import OpenAI
from openai.types import embedding_model
from pinecone import Pinecone
from chonkie import RecursiveChunker
from pdfminer.high_level import extract_text
from secrets import PINECONE_API_KEY, NEBIUS_API_KEY
from sentence_transformers import SentenceTransformer

index_name = "RAG-Assistant"

# initialize pinecone database for string vector embeddings
pc = Pinecone(api_key=PINECONE_API_KEY)
if not pc.has_index(index_name):
    pc.create_index(index_name)

# initialize pretrained model for embeddings
embeedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# initialize OpenAI-like client for response generation
nebiusClient = OpenAI()


def get_embedding(text):
    """
    Converts text to high-dimensional vector embeddings
    :param text: text to be converted
    :return: numerical vector representing meaning of input text
    """
    return embedding_model.encode(text)

def process_pdf(file_path):
    """
    Extracts raw text, splits it into meaningful chunks,
    converts to embeddings and then embeds them in Pinecone database
    :param file_path: the location of the pdf file
    """
    pass

def query_rag_system(query_text):
    """
    Uses a query to find most relevant text chunk from Pinecone database,
    updates openAI-like client with additional context for a more acurate
    response
    :param query_text:
    :return: AI-generated response
    """
    pass
