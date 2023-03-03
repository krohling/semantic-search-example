import PyPDF2
import json
from sentence_transformers import SentenceTransformer


# INITIALIZE THE MODEL WE WANT TO USE FOR GENERATING EMBEDDINGS
# https://medium.com/@nils_reimers/openai-gpt-3-text-embeddings-really-a-new-state-of-the-art-in-dense-text-embeddings-6571fe3ec9d9

embedding_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)


# GENERATE EMBEDDINGS FOR EACH PAGE IN THE PDF

embeddings = []
with open('example.pdf', "rb") as f:
    pdf = PyPDF2.PdfReader(f)
    page_count = len(pdf.pages)
    print(f"Page Count: {page_count}")
    for i in range(308, 325):
        print(f"Processing Page: {i+1}")
        text = pdf.pages[i].extract_text()
        embedding = list(embedding_model.encode(text))
        embedding = json.loads(str(embedding)) # Make the Float32's json friendly
        embeddings.append({
            "text": text,
            "embedding": embedding
        })


# SAVE OUT EMBEDDINGS TO A FILE

with open("embeddings.json", "w") as file:
    output = json.dumps(embeddings)
    file.write(output)
