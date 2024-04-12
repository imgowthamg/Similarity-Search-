import streamlit as st
import pandas as pd
from model import encode_texts, calculate_similarity, create_faiss_index

# Load the dataset
df = pd.read_csv('data/DataNeuron_Text_Similarity.csv')

# Get embeddings for text1 and text2
embeddings_text1 = encode_texts(df['text1'].tolist())
embeddings_text2 = encode_texts(df['text2'].tolist())

# Create a FAISS index and add embeddings
index = create_faiss_index(embeddings_text1)

# Define the Streamlit application
st.title("Semantic Textual Similarity (STS)")

# Create input text boxes for text1 and text2
text1 = st.text_area("Enter the first paragraph (text1):")
text2 = st.text_area("Enter the second paragraph (text2):")

# Create a button for calculating similarity
if st.button("Calculate Similarity"):
    # Encode text1 and text2 into embeddings
    embedding1 = encode_texts([text1])[0]
    embedding2 = encode_texts([text2])[0]

    # Calculate similarity score
    similarity_score = calculate_similarity(embedding1, embedding2)

    # Display the similarity score
    st.write(f"Similarity score: {similarity_score:.4f}")
