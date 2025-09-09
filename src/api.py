from fastapi import FASTApi
from pydantic import BaseModel
from pinecond import Pincone
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os


load_dotenv()

# Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(host=os.getenv("PINECONE_HOST"))


# LLM model
tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_NAME"))

model = AutoModelForCausalLM.from_pretrained(
    os.getenv("MODEL_NAME"),
    device_map="auto",
    dtype="auto"
)


# FASTApi app
app = FastAPI()


# Pydantic input model
class Query(BaseModel):
    question: str
    top_k: int = 5
    max_tokens: int = 300


# Query function
def answer_query(query: str, top_k: int = 5, max_tokens: int = 300):
    
    # Pinecone search
