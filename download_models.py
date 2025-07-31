from sentence_transformers import SentenceTransformer

print("Downloading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model downloaded successfully!")

# Test the model
test_embedding = model.encode("Test sentence")
print(f"Model working! Embedding shape: {test_embedding.shape}")