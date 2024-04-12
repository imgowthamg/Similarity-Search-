from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load a pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define functions for encoding texts and calculating similarity
def encode_texts(texts):
    return model.encode(texts).astype('float32')

def calculate_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Initialize FAISS index and add embeddings
def create_faiss_index(embeddings_text1):
    # Determine the dimension of the embeddings
    dimension = model.get_sentence_embedding_dimension()
    # Create a FAISS index using inner product (cosine similarity)
    index = faiss.IndexFlatIP(dimension)
    # Add the embeddings of text1 to the FAISS index
    index.add(embeddings_text1)
    return index
