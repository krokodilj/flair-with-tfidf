import itertools
import numpy as np
from flair.data import Sentence
from flair.embeddings import RoBERTaEmbeddings, BertEmbeddings, DocumentPoolEmbeddings

from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import sparse2full

import spacy
nlp = spacy.load('en_core_web_sm')

import torch

rob = RoBERTaEmbeddings()

sentences = ["goodbye world", "hello world", "destroy world"]
  
def keep_token(t):
    return (t.is_alpha and not (t.is_space or t.is_punct or t.is_stop))

def lemmatize_doc(doc):
    return [ t.lemma_ for t in doc if keep_token(t)]

docs = sentences
print("Parsing documents . . .")
parsed_docs = [lemmatize_doc(nlp(str(doc).lower())) for doc in docs]

print("Creating dictionary . . .")
# Create dictionary
dct = Dictionary(parsed_docs)
print(len(dct))

print("Creating BOW . . .")
# bow corpus
docs_corpus = [dct.doc2bow(doc) for doc in parsed_docs]

print("Initializing model . . .")
# Init tf-idf model
model_tfidf = TfidfModel(docs_corpus, id2word=dct)


test = ["destroy hello", "goodbye goodbye"]
    
sentemb = list()
for t in test:

    tokens = lemmatize_doc(nlp((str(t).lower())))
    print(tokens)

    bow = dct.doc2bow(tokens)
    print(bow)

    tfidf_vec = sparse2full(model_tfidf[[bow]][0], len(dct))
    print(tfidf_vec)

    sent = Sentence(t)

    rob.embed(sent)

    weights = list()
    embeddings = list()
    for token in sent.tokens:
        id = dct.token2id[token.text] 
        weights.append( tfidf_vec[id] )
        embeddings.append(token.get_embedding().tolist())

    weights = torch.tensor(weights)
    embeddings = torch.tensor(embeddings)
    sentemb.append(torch.matmul(weights, embeddings))