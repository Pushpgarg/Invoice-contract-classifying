import joblib
from text_processing import TextPreprocessing
vectorizer = joblib.load('vectorizer.pkl')
classifier = joblib.load('model.pkl')
text_processor = TextPreprocessing()


text = ""

want_to_play = classifier.predict(vectorizer.transform(text_processor.text_processing("you total bill is 2000 as you bought this")))

print(want_to_play)