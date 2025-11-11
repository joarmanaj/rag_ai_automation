
from sentence_transformers import SentenceTransformer

print("🚀 Testing Hugging Face Embedding Model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence = "This is a test sentence for embedding generation."
embedding = model.encode(sentence)

print("✅ Embedding generated successfully!")
print("🔢 Embedding vector length:", len(embedding))
print("🧠 First 5 values:", embedding[:5])

