from nltk.stem import WordNetLemmatizer
import re
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

lemmatizer=WordNetLemmatizer()
def clean(text):
    text=text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text=re.sub(r'[^\w\s]',"",text)
    words=text.split()
    words=[lemmatizer.lemmatize(word,pos='v')for word in words]
    return " ".join(words)