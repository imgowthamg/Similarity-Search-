
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
import json
app = Flask(__name__)

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
    
@app.route('/main', methods=['POST'])# Parse the request body


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
    try:
        request_data = request.get_json()
        text1 = request_data.get("text1")
        text2 = request_data.get("text2")
        
        if text1 and text2:
            # Calculate the similarity score
            similarity_score = calculate_similarity_api(text1, text2)
            response = {"similarity score": result}
            return jsonify(response)
            
            # Return the similarity score as a JSON response

        else:
            st.error("Both text1 and text2 are required in the request body.")
    except Exception as e:
        st.error(f"Error processing request: {e}")
    # Define the Streamlit app title
if __name__ == "__main__":
    main()


