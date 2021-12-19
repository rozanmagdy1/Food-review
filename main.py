import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
def CleanReview(review) :
    stop_words = set(stopwords.words('english'))
    review= review.replace('[^\w\s]',' ')
    review=review.lower().split()
    review=set(review)
    set_without_stopwords=review.difference(stop_words)
    list_without_stopwords=list(set_without_stopwords)
    stemmer = SnowballStemmer("english")
    review_cleaned=list()
    for word in list_without_stopwords:
        review_cleaned.append(stemmer.stem(word))
        review_cleaned = [" ".join(review_cleaned)]
    return review_cleaned
model_name = 'rf_model.pk'
vectorizer_name = 'tfidf_vectorizer.pk'
loaded_model = pickle.load(open(model_name, 'rb'))
loaded_vect =pickle.load(open(vectorizer_name, 'rb'))

def raw_test(review, model, vectorizer):
    review_c = CleanReview(review)
    embedding = vectorizer.transform(review_c)
    prediction = model.predict(embedding)
    return "Positive" if prediction == 1 else "Negative"

st.title('Amzazon Food Review')
review= st.text_input('Enter your review')
if st.button('Analyze'):
    result=raw_test(review,loaded_model,loaded_vect)
    st.write(f'{result}')