from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import sparse2full

import spacy
nlp = spacy.load('en_core_web_sm')

import numpy as np
import itertools

class TfIdfEncoder:
  
  def __init__(self, config):
    self.docs = config['docs']
    self.normalize = True
    
    self._build_model()
    
  
  def _build_model(self):    
    
    print("Parsing documents . . .")
    parsed_docs = [self.lemmatize_doc(nlp(str(doc).lower())) for doc in self.docs]

    print("Creating dictionary . . .")
    # Create dictionary 
    self.docs_dict = Dictionary(parsed_docs)
    print(len(self.docs_dict))
    
    print("Creating BOW . . .")
    # bow corpus
    docs_corpus = [self.docs_dict.doc2bow(doc) for doc in parsed_docs]

    print("Initializing model . . .")
    # Init tf-idf model
    self.model_tfidf = TfidfModel(docs_corpus, id2word=self.docs_dict)
    
    print("Setting up word vectors (GLOVE) . . .")
    # Init vector for every word in dictionary
    self.tfidf_emb_vecs = np.vstack([nlp(self.docs_dict[i]).vector for i in range(len(self.docs_dict))])

  
  def run(self, sentences, batch_size = None, **kwargs):
    """
    :param sentences: [ batch_size ]
    """
    if not batch_size:
      parsed_docs = [self.lemmatize_doc(nlp(str(doc).lower())) for doc in sentences]
      corpus = [self.docs_dict.doc2bow(doc) for doc in parsed_docs]
      vecs   = np.vstack([sparse2full(c, len(self.docs_dict)) for c in self.model_tfidf[corpus]])
      vecs = np.dot(vecs, self.tfidf_emb_vecs)
    else:
      encoded_list = list()

      for batch in self._batch(sentences, batch_size):
        parsed_docs = [self.lemmatize_doc(nlp(str(doc).lower())) for doc in batch]
        corpus = [self.docs_dict.doc2bow(doc) for doc in parsed_docs]
        enc   = np.vstack([sparse2full(c, len(self.docs_dict)) for c in self.model_tfidf[corpus]])
        encoded_list.append( np.dot(enc, self.tfidf_emb_vecs))

      vecs =  np.array(list(itertools.chain(*encoded_list)))
      
        
    return self._normalize(vecs) if self.normalize else vecs
  
  def _normalize(self, x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)
  
  def keep_token(self, t):
    return (t.is_alpha and not (t.is_space or t.is_punct or t.is_stop))

  def lemmatize_doc(self, doc):
    return [ t.lemma_ for t in doc if self.keep_token(t)]
    
  def _batch(self, iterable, n):
    """
    :param iterable: a list if things to be splitted into batches
    :param n: number of things per batch
    """
    l = len(iterable)
    for ndx in range(0, l, n):
      yield iterable[ndx:min(ndx + n, l)]