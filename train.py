import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
from preprocess import clean_text


def load_data(csv_path: str):
df = pd.read_csv(csv_path)
df = df.dropna(subset=['text', 'label'])
df['text_clean'] = df['text'].astype(str).apply(clean_text)
X = df['text_clean'].values
y = df['label'].values
return X, y


def main():
csv_path = 'data/news.csv'
X, y = load_data(csv_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
clf_lr = LogisticRegression(max_iter=1000)
clf_lr.fit(X_train_tfidf, y_train)
preds_lr = clf_lr.predict(X_test_tfidf)
print(accuracy_score(y_test, preds_lr))
print(classification_report(y_test, preds_lr))
clf_nb = MultinomialNB()
clf_nb.fit(X_train_tfidf, y_train)
preds_nb = clf_nb.predict(X_test_tfidf)
print(accuracy_score(y_test, preds_nb))
print(classification_report(y_test, preds_nb))
dump(tfidf, 'models/tfidf_vectorizer.joblib')
dump(clf_lr, 'models/classifier.joblib')


if __name__ == '__main__':
main()
