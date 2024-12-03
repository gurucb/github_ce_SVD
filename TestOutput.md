# Finding Similar Requirements using Vector Search and Cosine Similarity  
  
In this project, we utilized the capabilities of vector search and cosine similarity to identify similar software requirements. The process involves encoding the requirements into a vector space using machine learning models and performing a similarity search in that vector space.   
  
The script `vectorsearch.py` takes a target requirement as input and returns the top five most similar requirements from a pre-existing list. This is achieved by encoding the target and existing requirements into embeddings, computing the cosine similarity between the target requirement and each of the existing ones, and then returning the requirements with the highest similarity scores.  
  
In our experiment, we used the `all-MiniLM-L6-v2` model from the `sentence-transformers` library to encode the requirements. This model transforms input text into a high-dimensional vector that encapsulates the semantic meaning of the text.  
  
Here is a sample output from our script:  
  
```shell  
Input "The vehicle software should support voice commands for hands-free operation"  
python3 .\\vectorsearch.py  
['REQ014: The vehicle software should support voice commands for hands-free operation.',   
'REQ029: The vehicle software should provide a virtual assistant for interaction with the driver.',   
'REQ016: The vehicle software should provide a user-friendly interface for all controls and settings.',   
'REQ001: The vehicle software shall support the processing and response to sensor data in real-time.',   
'REQ004: The vehicle software must provide real-time diagnostic information on all vehicle subsystems.']  
```


# Cosine Similarity for Finding Similar Requirements  
  
In addition to the Vector Search approach, we have implemented another method to find similar requirements using Cosine Similarity.  
  
The script `cosine_similarity.py` uses the TF-IDF (Term Frequency-Inverse Document Frequency) method to vectorize the requirements, and then calculates the cosine similarity between the vectors.  
  
When we run the script with the target requirement as "The vehicle software should support voice commands for hands-free operation", we get the following output:

```shell 
python3 .\cosine_similarity.py  
['REQ040: The vehicle hardware must include an adaptive suspension system that can be controlled via software.',   
'REQ039: The vehicle software should provide an interface for setting cruise control speeds.',   
'REQ018: The vehicle software should provide warnings when tire pressure drops below a specified level.',   
'REQ017: The vehicle hardware must include sensors for monitoring tire pressure in real-time.',   
'REQ016: The vehicle software should provide a user-friendly interface for all controls and settings.']  
```

This output shows the five requirements that are most similar to the target requirement "The vehicle software should support voice commands for hands-free operation".

In conclusion, cosine similarity provides another robust method for finding similar requirements in a large set of software requirements. By comparing the cosine similarity of the TF-IDF vectors of the requirements, we can efficiently identify requirements that are semantically similar.


