from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# load the HuggingFace embedding model
print("Loading HuggingFace embedding model...\n")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("\nHuggingFace embedding model loaded successfully!")

# test 1: create an embedding vector for a sample text and inspect the output
sample_text = "What is the minimum CGPA required for entry bachelor in computer science?"
vector = embedding_model.embed_query(sample_text)

print(f"Text   : '{sample_text}'")
print(f"Vector : {vector[:6]}...")   # just the first 6 numbers
print(f"Length : {len(vector)} dimensions\n")


# test 2

def cosine_similarity(a,b):
    """
    Measures the angle between two vectors.
    1.0 = identical direction (same meaning)
    0.0 = perpendicular (unrelated meaning)
    """
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sentences = [
    "What is the minimum CGPA required for entry?",   # your question
    "CGPA entry requirement 2.50 senate approval",    # relevant chunk from your PDF
    "Programme learning outcomes for graduates",      # unrelated chunk
    "iLovePDF publisher document metadata",           # completely unrelated
]

print("Similarity to your question:")
print("-" * 50)
question_vector = embedding_model.embed_query(sentences[0])

for sentence in sentences[1:]:
    vec = embedding_model.embed_query(sentence)
    score = cosine_similarity(question_vector, vec)
    bar = "█" * int(score * 30)
    print(f"  {score:.3f}  {bar}")
    print(f"         '{sentence}'\n")