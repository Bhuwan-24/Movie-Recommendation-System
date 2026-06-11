import joblib as jb
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from text_clean import clean

tw = 3.0
cw = 2.0
dw = 1.0
t_tfidf=jb.load('title.pkl')
c_tfidf=jb.load('category.pkl')
d_tfidf=jb.load('description.pkl')
s_v=jb.load('sparse_vectorized_matrix.pkl')
cleaned_df=jb.load('products.pkl')

def similarity_calculate(title="",category="",description="",top_similar=10):
    t_vector = t_tfidf.transform([clean(title)]) * tw
    c_vector = c_tfidf.transform([clean(category)]) * cw
    d_vector = d_tfidf.transform([clean(description)]) * dw
    ask = sp.hstack([t_vector, c_vector, d_vector],format="csr")
    similarity = cosine_similarity(ask,s_v).flatten()
    top_idx = np.argsort(similarity)[::-1][:top_similar]
    results = cleaned_df.iloc[top_idx].copy()
    results["score"] = similarity[top_idx].round(3)
    return results[["title", "main_category", "score"]]