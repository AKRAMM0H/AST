import os
import tiktoken
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Shared encoder and embedding model
enc = tiktoken.get_encoding("cl100k_base")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Neo4j config
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")

# OpenRouter config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")