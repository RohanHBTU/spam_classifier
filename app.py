import streamlit as st
import pickle
import pandas as pd
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

#tfidf = pickle.load(open('vectorizer.pkl','rb'))
#model = pickle.load(open('model.pkl','rb'))
with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.header("Message Spam Classifier")
data = pd.read_csv("spam.csv",encoding="ISO-8859-1")
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
if st.checkbox('Show Training Dataframe'):
    data

st.subheader("Please enter your message!")
st.text("")

input = st.text_area("Enter here")

if st.button('Check Now!'):
    transformed_sms = transform_text(input)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.error("Spam")
    else:
        st.success("Not Spam")
    st.write("Thank you! I hope you liked it. ")
    st.write("Check out this Repo's [GitHub Link](https://github.com/RohanHBTU/spam-classifier)")
