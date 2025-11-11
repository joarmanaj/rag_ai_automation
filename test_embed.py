from langchain_ollama import OllamaEmbeddings

print("🚀 Testing Ollama Embedding Model...")
embedder = OllamaEmbeddings(model="nomic-embed-text")

sample_text = "Artificial Intelligence is transforming automation."
embedding = embedder.embed_query(sample_text)

print(f"✅ Embedding generated successfully! Length: {len(embedding)}")
print(f"🧩 First 10 values: {embedding[:10]}")
