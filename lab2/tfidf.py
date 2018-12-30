documents = (
"The sky is blue",
"The sun is bright",
"The sun in the sky is bright",
"We can see the shining sun, the bright sun"
)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print(tfidf_matrix.shape)
print("tf-idf matrix for the documentsis :\n",tfidf_matrix)
from sklearn.metrics.pairwise import cosine_similarity
print("cosibne similarity matrix for the documents :\n",cosine_similarity(tfidf_matrix, tfidf_matrix))
import math
# This was already calculated on the previous step, so we just use the value
cos_sim = 0.52305744
angle_in_radians = math.acos(cos_sim)
print("angle:\n",math.degrees(angle_in_radians))
