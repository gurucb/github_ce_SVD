import numpy as np  
import faiss  
from sentence_transformers import SentenceTransformer  
  
def read_requirements(req_file):  
    with open(req_file, 'r') as f:  
        requirements = [line.strip() for line in f]  
    return requirements  
  
def create_embeddings(requirements):  
    model = SentenceTransformer('all-MiniLM-L6-v2')  
    embeddings = model.encode(requirements)  
    return np.array(embeddings)  
  
def build_faiss_index(embeddings):  
    index = faiss.IndexFlatL2(embeddings.shape[1])  
    index.add(embeddings.astype('float32'))  
    return index  
  
def find_similar_requirements(target_req, index, requirements): 
    model = SentenceTransformer('all-MiniLM-L6-v2')  
    target_embedding = model.encode([target_req])  
    D, I = index.search(np.array(target_embedding).astype('float32'), 5)  
    return [requirements[i] for i in I[0]]  
  
# Example usage:  
  
requirements = read_requirements('Samples\\requirements.txt')  
embeddings = create_embeddings(requirements)  
index = build_faiss_index(embeddings)  
  
target_req = "The vehicle software should support voice commands for hands-free operation"  
print(find_similar_requirements(target_req, index, requirements))  