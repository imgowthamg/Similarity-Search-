import streamlit as st
import pandas as pd
from model import encode_texts, calculate_similarity, create_faiss_index

def main():
    # Load the dataset
    data_path = 'data/DataNeuron_Text_Similarity.csv'
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"File not found: {data_path}")
        return

    # Pre-compute embeddings for text1 and text2 from the dataset
    embeddings_text1 = encode_texts(df['text1'].tolist())
    embeddings_text2 = encode_texts(df['text2'].tolist())

    # Create a FAISS index and add embeddings
    index = create_faiss_index(embeddings_text1)

    # Define the Streamlit application
    st.title("Semantic Textual Similarity (STS)")

    # Create input fields for text1 and text2
    with st.form(key='similarity_form'):
        text1 = st.text_area("Enter the first paragraph (text1):", key='text1')
        text2 = st.text_area("Enter the second paragraph (text2):", key='text2')
        submitted = st.form_submit_button("Calculate Similarity")

    if submitted:
        # Encode text1 and text2 into embeddings
        embedding1 = encode_texts([text1])[0]
        embedding2 = encode_texts([text2])[0]

        # Calculate similarity score
        similarity_score = calculate_similarity(embedding1, embedding2)

        # Display the similarity score
        st.success(f"Similarity score: {similarity_score:.4f}")

if __name__ == "__main__":
    main()
