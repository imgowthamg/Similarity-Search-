
import streamlit as st
from sentence_transformers import SentenceTransformer, util


# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def calculate_similarity_api(text1, text2):
    """
    Calculate the similarity score between two texts and return the score.
    """
    # Encode the input texts
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)

    # Compute cosine similarity between the embeddings
    similarity_score = util.cos_sim(embeddings1, embeddings2)

    # Return the similarity score
    return similarity_score.item()


def main():

    st.title("Semantic Textual Similarity (STS)")


    # User interface for inputting text1 and text2
    text1 = st.text_area("Enter the first paragraph (text1):")
    text2 = st.text_area("Enter the second paragraph (text2):")

    # Button to calculate similarity
    if st.button("Calculate Similarity"):
        if text1.strip() and text2.strip():
            # Calculate the similarity score
            similarity_score = calculate_similarity_api(text1, text2)

            # Display the similarity score
            st.write(f"Similarity score: {similarity_score:.4f}")
        else:
            st.warning("Please enter both sentences to calculate similarity.")

    # Define the Streamlit app title
if __name__ == "__main__":
    main()


