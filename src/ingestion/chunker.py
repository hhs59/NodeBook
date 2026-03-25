import re

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    
    words = text.split()
    chunks =[]
    
    step = chunk_size - overlap
    
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        
        if i + chunk_size >= len(words):
            break
            
    print(f"Created {len(chunks)} chunks from the document.")
    return chunks

if __name__ == "__main__":
    dummy_10_page_text = "Artificial Intelligence is great. " * 1000  # 3000 words
    
    my_chunks = chunk_text(dummy_10_page_text, chunk_size=100, overlap=20)
    
    print(f"\nTotal chunks made: {len(my_chunks)}")
    print("\n--- PREVIEW OF CHUNK 1 ---")
    print(my_chunks[0])