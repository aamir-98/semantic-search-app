import streamlit as st
import nltk
from nltk.corpus import reuters, stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import numpy as np

# Download necessary datasets
nltk.download('reuters')
nltk.download('punkt')
nltk.download('stopwords')

# Load and preprocess Reuters corpus
corpus_sentences = []
for fileid in reuters.fileids():
    raw_text = reuters.raw(fileid)
    tokenized_sentence = [word for word in word_tokenize(raw_text.lower()) if word.isalnum()]
    corpus_sentences.append(tokenized_sentence)

# Train Word2Vec model
model = Word2Vec(sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=4)

# Preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalnum() and word not in stop_words]

# Compute average embedding
def compute_embedding(tokens, word2vec_model):
    vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(word2vec_model.vector_size)

# Compute similarity
def calculate_similarity(vec1, vec2):
    if not np.any(vec1) or not np.any(vec2):  # Avoid zero vectors
        return 0.0
    return 1 - cosine(vec1, vec2)

# Retrieve top documents
def get_top_documents(query, model, documents, top_n=5):
    query_tokens = preprocess_text(query)
    query_vector = compute_embedding(query_tokens, model)
    similarity_scores = [(doc_id, calculate_similarity(query_vector, compute_embedding(doc, model))) for doc_id, doc in enumerate(documents)]
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    return similarity_scores[:top_n]

# Streamlit UI
st.title("Semantic Search with Word2Vec")
query = st.text_input("Enter search query:")
if st.button("Search"):
    if query:
        top_results = get_top_documents(query, model, corpus_sentences, top_n=5)
        st.write("### Top Results:")
        for doc_id, similarity in top_results:
            doc_preview = ' '.join(corpus_sentences[doc_id][:10]) + '...'
            st.write(f"**Document {doc_id}** (Similarity: {similarity:.4f})")
            st.write(f"{doc_preview}")
