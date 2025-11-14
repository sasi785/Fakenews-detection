import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
ps = PorterStemmer()


def clean_text(text: str) -> str:
if not isinstance(text, str):
return ""
text = text.lower()
text = re.sub(r'http\S+|www\S+', '', text)
text = re.sub(r'[^a-z\s]', ' ', text)
text = re.sub(r'\s+', ' ', text).strip()
tokens = [w for w in text.split() if w not in STOPWORDS]
stems = [ps.stem(w) for w in tokens]
return ' '.join(stems)
