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

# Penn Treebank Corpus
print("\nPenn Treebank Sample Words:")
print(treebank.words()[:20])
```

### 2. Sentence & Word Segmentation

```python
import spacy
from nltk.tokenize import word_tokenize, RegexpTokenizer

nlp = spacy.load("en_core_web_sm")
doc = nlp("I love coding. Practicing NLP every day helps.")
print("Sentences:")
for sent in doc.sents:
    print(sent)

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
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

nltk.download('punkt')

text = "I'm learning NLP!!! It's fun, isn't it?"
text = contractions.fix(text)
clean = re.sub(r'[^a-zA-Z\s]', '', text).lower()
tokens = word_tokenize(clean)

print("Tokens:", tokens)
print("Unigrams:", list(ngrams(tokens,1)))
print("Bigrams:", list(ngrams(tokens,2)))
print("Trigrams:", list(ngrams(tokens,3)))
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

### 6. Named Entity Recognition (NER)

```python
import spacy
nlp = spacy.load("en_core_web_sm")

text = "Barack Obama was born in Hawaii and worked for Google in the US."
doc = nlp(text)

print("Named Entities:")
for ent in doc.ents:
    print(ent.text, "->", ent.label_)
```
