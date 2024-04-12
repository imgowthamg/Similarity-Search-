from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load a pre-trained sentence transformer model (consider using transformers library)
model = SentenceTransformer("all-MiniLM-L6-v2")  # Or any suitable model from Sentence Transformers or Hugging Face

# Define functions for encoding texts and calculating similarity
def encode_texts(texts):
  """
  Encodes a list of texts using the sentence transformer model.
  Args:
      texts: A list of strings representing the texts to encode.
  Returns:
      A NumPy array of shape (n_texts, embedding_dim), where n_texts is the number of input texts
      and embedding_dim is the dimensionality of the sentence embeddings.
  """
  return model.encode(texts).astype('float32')

def calculate_similarity(embedding1, embedding2):
  """
  Calculates the cosine similarity between two sentence embeddings.
  Args:
      embedding1: A NumPy array representing the first sentence embedding.
      embedding2: A NumPy array representing the second sentence embedding.
  Returns:
      A float value representing the cosine similarity between the two embeddings.
  """
  return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Initialize FAISS index and add embeddings
def create_faiss_index(embeddings_text1):
  """
  Creates a FAISS index for efficient similarity search.
  Args:
      embeddings_text1: A NumPy array of shape (n_texts, embedding_dim) containing the sentence embeddings
                          for the first set of texts (text1).
  Returns:
      A FAISS index object for efficient similarity search.
  """
  # Determine the dimension of the embeddings
  dimension = model.get_sentence_embedding_dimension()
  # Create a FAISS index using inner product (cosine similarity)
  index = faiss.IndexFlatIP(dimension)
  # Add the embeddings of text1 to the FAISS index
  index.add(embeddings_text1)
  return index
