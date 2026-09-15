import nltk

nltk.download("punkt_tab")  # Download the Punkt tokenizer models

text = "Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data and improve their performance over time without being explicitly programmed."

words_tokes = nltk.word_tokenize(text)  # Tokenize the text into words
print(words_tokes)  # Print the list of word tokens
