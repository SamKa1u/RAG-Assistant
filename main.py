from pinecone import Pinecone
import os

from secrets import PINECONE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = ""