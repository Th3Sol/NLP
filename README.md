### 1. Brown & Penn Treebank Corpus

```python
import nltk
from nltk.corpus import brown, treebank

nltk.download('brown')
nltk.download('treebank')

# Brown Corpus
print("Brown Corpus Categories:")
print(brown.categories())

print("\nSample Words from 'news' category:")
print(brown.words(categories='news')[:20])

# OR (you can skip this ↓)

# Penn Treebank Corpus
print("\nPenn Treebank Sample Words:")
print(treebank.words()[:20])
```

### 2. Sentence & Word Segmentation

```python
# Sentence segmentation
import spacy
nlp = spacy.load("en_core_web_sm") # To install: python -m spacy download en_core_web_sm
doc = nlp("I love coding. Practicing NLP every day helps.")
print("Sentences:")
for sent in doc.sents:
    print(sent)

# OR (you can skip anyone  ↑ ↓)

# Word segmentation
from nltk.tokenize import word_tokenize, RegexpTokenizer
text = "Hi! Let's go shopping."
print("\nNLTK Word Tokenize:", word_tokenize(text))

tk = RegexpTokenizer(r'\s+', gaps=True)
print("Regex Tokenize:", tk.tokenize("I Love Python"))
```

### 3. Lemmatization & Stemming

```python
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('wordnet')

words = ["running", "flies", "better", "wolves"]
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("Stemming Results:")
for w in words:
    print(w, "->", porter.stem(w))

print("\nLemmatization Results:")
for w in words:
    print(w, "->", lemmatizer.lemmatize(w))
```

### 4. Text Normalization & N-Grams

```python
import nltk, re, contractions
from nltk.tokenize import word_tokenize

nltk.download('punkt')

text = "I'm learning NLP!!! It's fun, isn't it?"
text = contractions.fix(text)                     # Expand contractions
clean_text = re.sub(r'[^a-zA-Z\s]', '', text)     # Remove special characters
tokens = word_tokenize(clean_text.lower())         # Tokenize and lowercase

print("Original Text:", text)
print("Cleaned Text:", clean_text)
print("Tokens:", tokens)
```

### 5. POS Tagging

```python
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

text = "The quick brown fox jumps over the lazy dog."
tokens = word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)

print("Tokens:", tokens)
print("\nPOS Tagging Results:")
for word, tag in pos_tags:
    print(f"{word} -> {tag}")
```
