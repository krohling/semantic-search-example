import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer

question = sys.argv[1]

# INITIALIZE THE MODEL WE WANT TO USE FOR GENERATING EMBEDDINGS
# https://medium.com/@nils_reimers/openai-gpt-3-text-embeddings-really-a-new-state-of-the-art-in-dense-text-embeddings-6571fe3ec9d9

embedding_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)


# WE'LL USE THIS FUNCTION TO COMPARE 2 VECTORS AND MEASURE THEIR DISTANCE

def cosine_similarity(vec1, vec2):
  dot_product = np.dot(vec1, vec2)
  norm1 = np.linalg.norm(vec1)
  norm2 = np.linalg.norm(vec2)
  return dot_product / (norm1 * norm2)


# LOAD THE EMBEDDINGS WE GENERATED IN STEP 1

embeddings = None
with open("embeddings-beef.json", "r") as f:
    embeddings = json.load(f)


# GENERATE AN EMBEDDING FOR THE QUESTION WE WANT TO ASK

question_embedding = list(embedding_model.encode(question))


# COMPARE THE EMBEDDING FOR OUR QUESTION TO THE EMBEDDINGS IN OUR DOCUMENT

for e in embeddings:
    e["similarity"] = cosine_similarity(question_embedding, e["embedding"])


# SORT OUR EMBEDDINGS BY SIMILARITY

embeddings = sorted(embeddings, key=lambda x: x["similarity"], reverse=True)

for i in range(5):
    print(embeddings[i]["similarity"])
    print(embeddings[i]["text"])
    print("\n")

