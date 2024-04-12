# import streamlit as st
# import pandas as pd
# from model import encode_texts, calculate_similarity, create_faiss_index


# def main():
#     # Load the dataset
#     data_path = 'data/DataNeuron_Text_Similarity.csv'
#     try:
#         df = pd.read_csv(data_path)
#     except FileNotFoundError:
#         st.error(f"File not found: {data_path}")
#         return

#     # Pre-compute embeddings for text1 and text2 from the dataset
#     embeddings_text1 = encode_texts(df['text1'].tolist())
#     embeddings_text2 = encode_texts(df['text2'].tolist())

#     # Create a FAISS index and add embeddings
#     index = create_faiss_index(embeddings_text1)

#     # Define the Streamlit application
#     st.title("Semantic Textual Similarity (STS)")

#     # Create input fields for text1 and text2
#     with st.form(key='similarity_form'):
#         text1 = st.text_area("Enter the first paragraph (text1):", key='text1')
#         text2 = st.text_area("Enter the second paragraph (text2):", key='text2')
#         submitted = st.form_submit_button("Calculate Similarity")

#     if submitted:
#         # Encode text1 and text2 into embeddings
#         embedding1 = encode_texts([text1])[0]
#         embedding2 = encode_texts([text2])[0]

#         # Calculate similarity score
#         similarity_score = calculate_similarity(embedding1, embedding2)

#         # Display the similarity score
#         st.success(f"Similarity score: {similarity_score:.4f}")

# if __name__ == "__main__":
#     main()
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Custom CSS styles
st.markdown(
    f"""
    <style>
    body {{
        background-color: #f5f5f5; /* Light Grey */
        font-family: Arial, sans-serif;
    }}
    .stTextInput input {{
        background-color: #ffffff; /* White */
        color: #000000; /* Black */
        border: 2px solid #cccccc; /* Light Grey */
        border-radius: 5px;
        padding: 10px;
    }}
    .stTextInput label {{
        color: #000000; /* Black */
    }}
    .stButton button {{
        background-color: #008CBA; /* Dark Blue */
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
    }}
    .stSuccess {{
        background-color: #4CAF50; /* Green */
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px;
    }}
    .stWarning {{
        background-color: #FF5722; /* Orange */
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Application Title
st.title("Text Similarity Calculator")

# Text Input
text1 = st.text_area("Enter Text 1:", "", key="text1")
text2 = st.text_area("Enter Text 2:", "", key="text2")


# Text Cleaning Function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    return text

if st.button("Calculate Similarity"):
    if not text1 or not text2:
        st.warning("Please enter both texts.")
    else:
        # Clean the text
        text1 = clean_text(text1)
        text2 = clean_text(text2)

        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])

        # Calculate Cosine Similarity
        cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

        # Show Result
        st.success(f"Similarity between texts: {cosine_sim:.2f}")

# Text Suggestions
st.subheader("Text Suggestions")
st.write("Suggestions for using the text similarity calculator:")
st.write("- Use the text input boxes above to calculate similarity between two texts.")
st.write("- Click the 'Calculate Similarity' button to clean the texts.")
st.write("- Examine the results to see the similarity between the texts.")


if __name__ == "__main__":
    st.write("To run the application, use the 'Run' option from the left-side menu.")
