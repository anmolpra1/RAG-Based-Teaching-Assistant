import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests 
import json

def create_embeddings(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]


def seconds_to_time(s):
    """Convert seconds into readable time format (e.g. 150 -> 2:30)"""
    return f"{int(s)//60}:{int(s)%60:02d}"


def find_similar_chunks(df, query, top_k=5):
    """
    Embed the user's query and find the top_k most similar chunks 
    in the  dataframe using the cosine similarity 
    """
    # Embed the incoming query 
    query_embedding = create_embeddings([query])
    if query_embedding is None:
        print("Failed to embed the query")
        return None
    
    # Reshape query vector for cosine similarity computation
    query_vec = np.array(query_embedding[0]).reshape(1, -1)

    # Stack all chunk embeddings into a 2D matrix
    embedding_matrix = np.vstack(df['embedding'].values)

    # Compute cosine similarity between query and all chunks
    similarities = cosine_similarity(query_vec, embedding_matrix)[0]

    # Add similarity scores to df, sort descending, return top_k
    result = df.copy()
    result['similarity'] = similarities
    result = result.sort_values('similarity', ascending=False).head(top_k)

    return result

def inference(prompt):
    """
    Send the prompt to llama via Ollama and stream the respone
    token by token so the user sees oyput as it's being generated.
    """
    
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream" : True # Straem response instead of waaiting for full reply
    })
    
    full_response = ""
    for line in r.iter_lines():
        if line:
            chunk = json.loads(line)
            # Print each token as it arrives
            print(chunk["response"], end="", flush=True)
            full_response += chunk["response"]
            # Stop when model signals it's done
            if chunk.get("done"):
                break
    print()  # newline after streaming finishes
    return full_response


if __name__ == "__main__":
    # Load precomputed embeddings from parquet file
    df = pd.read_parquet('embeddings.parquet')

    # Keep asking questions until user types 'quit'
    while True:
        incoming_query = input("\nAsk a Question (or 'quit' to exit): ")
        if incoming_query.lower() == 'quit':
            break

        # Find the top 5 most relevant chunks for the query
        top_chunks = find_similar_chunks(df, incoming_query, top_k=5)

        # Convert start/end timestamps from seconds to mm:ss format
        display_df = top_chunks.copy()
        display_df["start"] = display_df["start"].apply(seconds_to_time)
        display_df["end"] = display_df["end"].apply(seconds_to_time)

        # Build the prompt with retrieved chunks as context
        prompt = f'''You are a helpful assistant for the Sigma Web Development course.

You will be given video subtitle chunks with title, video number, start time, end time, and text.
Your job is ONLY to answer questions related to the course content using these chunks.

Rules:
- If the question is unrelated to web development or the course, reply ONLY with: "I can only answer questions related to the Sigma Web Development course."
- Do NOT make up information. Only use the provided chunks.
- Tell the user which video and timestamp to go to ONLY when answering a valid course-related question..
- Be conversational and helpful.

Video chunks:
{display_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
Question: "{incoming_query}"
Answer:'''

        # Save prompt to file for debugging purposes
        with open("prompt.txt", "w") as f:
            f.write(prompt)

        print("\n--- Answer ---")

        # Run inference and stream the response to terminal
        response = inference(prompt)

        # Save the final response to file for reference
        with open("response.txt", "w") as f:
            f.write(response)
    
   