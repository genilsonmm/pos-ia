import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Machine learning é um campo da inteligência artificial que permite que computadores aprendam padrões a partir de dados.",
    "O aprendizado de máquina dá aos sistemas a capacidade de melhorar seu desempenho sem serem explicitamente programados.",
    "Em vez de seguir apenas regras fixas, o machine learning descobre relações escondidas nos dados.",
    "Esse campo combina estatística, algoritmos e poder computacional para extrair conhecimento.",
    "O objetivo é criar modelos capazes de generalizar além dos exemplos vistos no treinamento.",
    "Aplicações de machine learning vão desde recomendações de filmes até diagnósticos médicos.",
    "Os algoritmos de aprendizado de máquina transformam dados brutos em previsões úteis.",
    "Diferente de um software tradicional, o ML adapta-se conforme novos dados chegam.",
    "O aprendizado pode ser supervisionado, não supervisionado ou por reforço, dependendo do tipo de problema.",
    "Na prática, machine learning é o motor que impulsiona muitos avanços em visão computacional e processamento de linguagem natural.",
    "Mais do que encontrar padrões, o machine learning ajuda a tomar decisões baseadas em evidências.",
]


def preprocess(text):
    # Convert to lowercase
    text_lower = text.lower()
    # Tokenize the text into words
    toknes = nltk.word_tokenize(text_lower)
    # isalnum() só mantém os elementos que são alfanuméricos (ou seja, compostos apenas por letras e/ou números, sem pontuação
    return [word for word in toknes if word.isalnum()]


preprocessed_documents = [" ".join(preprocess(doc)) for doc in documents]
print(preprocessed_documents)

# TF (Term Frequency) → Frequência da palavra no documento.
# IDF (Inverse Document Frequency) → Frequência inversa nos documentos.
# Mede quão raro é o termo em um conjunto de textos.
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_documents)

print(
    "TF-IDF Matrix: 11, 106 (11 documentos, 106 palavras únicas, dimensões da matriz)"
)
print(tfidf_matrix)

query = "machine learning"


def search_tfidf(query, vectorizer, tfidf_matrix):
    query_vector = vectorizer.transform([query])
    # Calculate cosine similarity between the query and the documents (distance)
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    sorted_similarities = list(enumerate(similarities))
    results = sorted(sorted_similarities, key=lambda x: x[1], reverse=True)
    return results


search_similarities = search_tfidf(query, vectorizer, tfidf_matrix)
print(f"Top 10 documentos por score de similaridade: {query}:")

for doc_index, score in search_similarities[:10]:
    print(f"Documento {doc_index}: {documents[doc_index]}")
