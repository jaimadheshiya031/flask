from flask import Flask, request, render_template, Response
from flask_cors import CORS, cross_origin
# from flask_ngrok import run_with_ngrok
app = Flask(__name__)
CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'
# run_with_ngrok(app)
import nltk, string 
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt') # if necessary...
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
def stem_tokens(tokens):
 return [stemmer.stem(item) for item in tokens]
'''remove punctuation, lowercase, stem'''
def normalize(text):
 return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))
vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')
def cosine_sim(text1, text2):
 tfidf = vectorizer.fit_transform([text1, text2])
 return ((tfidf * tfidf.T).A)[0,1]
# test call to ml function
# print( cosine_sim('a little bird', 'a little bird') )
# print( cosine_sim('a little bird', 'a little bird chirps') )
# print( cosine_sim('a little bird', 'a big dog barks') )

@app.route('/', methods=['GET','POST'])
def matchCaller():
 if request.method == 'POST':
    data = request.get_json()
    # test sample are keys of request body json
    print(data.get("test") )
    # print(data.get("sample") )
    # print( cosine_sim('a little bird', 'a little bird chirps') )
    result = cosine_sim(data.get("test"), data.get('sample') )
    print( result )
    return str(result)
 else:
    return "Invalid call!!"
if __name__ == "__main__":
 app.run( debug = True )
