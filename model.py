import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t', header=None, names=['label', 'message'])

# Train model
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(df['message'], df['label'])

# Save model
joblib.dump(model, 'spam_classifier.pkl')
