import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv


load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(host=os.getenv("PINECONE_HOST"))

MAX_CHARS_PER_CHUNK = 1000
TOP_K = 3


def answer_query(query: str):
    
    # Pinecone search
    results = index.search(
        namespace = os.getenv("PINECONE_NAMESPACE"),
        query = {"inputs": {"text":query}, "top_k": TOP_K},
        fields=["text", "source", "category"]
    )

    hits = results["result"]["hits"]

    if not hits:
        context = "No relevant information found in the knowledge base."
    else:
        context = "\n".join([
            hit["fields"]["text"][:MAX_CHARS_PER_CHUNK]
            for hit in hits if "fields" in hit and "text" in hit["fields"]
        ])


    # Prompt
    prompt = (
        f"Answer as if you are Ramon Emmiel Jasmin (me), in first person, casual, "
        f"friendly, and concise. Use context only, 1-2 sentences.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    response = client.responses.create(
        model=os.getenv("MODEL_NAME"),
        instructions="Answer as if you are Ramon Emmiel Jasmin in first person, in a casual and friendly tone, using your knowledge and experiences, and keep answers short and easy to understand.",
        input=prompt
    )

    return response.output_text.strip()


# Vercel handler
def handler(req):
    try:
        data = req.json()
        query = data.get("question", "")
        if not query:
            return {"error": "Missing 'question' field"}
        answer = answer_query(query)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
