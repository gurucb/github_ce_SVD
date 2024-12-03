from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  
  
def find_similar_requirements(target_req, req_file):  
    # Read the requirements from the file  
    with open(req_file, 'r') as f:  
        requirements = [line.strip() for line in f]  
  
    # Add the target requirement to the list  
    requirements.append(target_req)  
  
    # Create a TfidfVectorizer object  
    vectorizer = TfidfVectorizer()  
  
    # Transform the requirements into a TF-IDF matrix  
    tfidf_matrix = vectorizer.fit_transform(requirements)  
  
    # Compute the cosine similarity of the target requirement with all others  
    cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])  
  
    # Get the indices of the most similar requirements  
    similar_indices = cosine_similarities[0].argsort()[:-6:-1]  
  
    # Return the most similar requirements  
    return [requirements[index] for index in similar_indices]  
  
# Example usage:  
#target_req = "The vehicle software should support voice commands for hands-free operation"  
target_req = "braking"  
print(find_similar_requirements(target_req, 'Samples\\requirements.txt'))  