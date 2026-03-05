import requests
import os
import json
import pandas as pd
from typing import List, Optional


def create_embedding(text_list: List[str]) -> Optional[List[List[float]]]:
    """Create embeddings with error handling and debugging."""
    try:
        # Ensure input is a list
        if isinstance(text_list, str):
            text_list = [text_list]
        
        payload = {
            "model": "bge-m3",
            "input": text_list
        }
        
        r = requests.post(
            "http://localhost:11434/api/embed",
            json=payload,
            timeout=60
        )
        
        # Check status code
        if r.status_code != 200:
            print(f"Error: Status code {r.status_code}")
            print(f"Response: {r.text}")
            return None
        
        response_data = r.json()
        
        # Handle different response formats
        if "embeddings" in response_data:
            return response_data["embeddings"]
        elif "embedding" in response_data:
            # Single embedding, wrap in list
            embedding = response_data["embedding"]
            return [embedding] if not isinstance(embedding[0], list) else embedding
        else:
            print(f"Unexpected response keys: {list(response_data.keys())}")
            print(f"Full response: {response_data}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Response text: {r.text}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def process_jsons(jsons_dir: str = "jsons"):
    """Process all JSON files with better error handling."""
    if not os.path.exists(jsons_dir):
        print(f"Directory '{jsons_dir}' not found!")
        return None
    
    jsons = [f for f in os.listdir(jsons_dir) if f.endswith('.json')]
    my_dicts = []
    chunk_id = 0
    
    for json_file in jsons:
        filepath = os.path.join(jsons_dir, json_file)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            if 'chunks' not in content:
                print(f"Warning: 'chunks' key not found in {json_file}")
                continue
            
            chunks = content['chunks']
            print(f"\nCreating Embeddings for {json_file} ({len(chunks)} chunks)")
            
            # Extract texts
            texts = [c.get('text', '') for c in chunks]
            
            # Filter out empty texts
            texts = [t for t in texts if t.strip()]
            
            if not texts:
                print(f"  No valid text found in {json_file}")
                continue
            
            # Create embeddings
            embeddings = create_embedding(texts)
            
            if embeddings is None:
                print(f"  Failed to create embeddings for {json_file}, skipping...")
                continue
            
            if len(embeddings) != len(texts):
                print(f"  Warning: Embedding count ({len(embeddings)}) != text count ({len(texts)})")
                continue
            
            # Add to results
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    chunk['chunk_id'] = chunk_id
                    chunk['embedding'] = embeddings[i]
                    chunk['source_file'] = json_file
                    my_dicts.append(chunk)
                    chunk_id += 1
            
            print(f"  ✓ Successfully processed {len(embeddings)} chunks")
                    
        except json.JSONDecodeError as e:
            print(f"Error decoding {json_file}: {e}")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
    
    if my_dicts:
        df = pd.DataFrame.from_records(my_dicts)
        print(f"\n{'='*60}")
        print(f"Total chunks processed: {len(df)}")
        print(f"From {len(set(df['source_file']))} files")
        print(f"{'='*60}")
        return df
    else:
        print("No data processed!")
        return None

if __name__ == "__main__":
    
    # First, test the API
    
    print("Testing Ollama API...")
    
    test_result = create_embedding(["Hello world"])
    
    if test_result is None:
        print("\n⚠️  API test failed! Check:")
        print("1. Is Ollama running? (ollama serve)")
        print("2. Is bge-m3 model available? (ollama list)")
        print("3. Try: ollama pull bge-m3")
    else:
        print(f"✓ API test successful! Embedding dimension: {len(test_result[0])}")
        
        # Process all files
        print("\nProcessing JSON files...")
        df = process_jsons("jsons")
        
        if df is not None:
            # Save results
            df.to_parquet('embeddings.parquet')
            print(f"\n✓ Saved embeddings to embeddings.parquet")
            print(f"\nDataFrame preview:")
            print(df[['chunk_id', 'text', 'source_file']].head())