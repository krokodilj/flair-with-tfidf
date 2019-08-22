from flair.embeddings import DocumentPoolEmbeddings, RoBERTaEmbeddings
from flair.data import Sentence

from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import sparse2full


import spacy
nlp = spacy.load('en_core_web_sm')



roberta = RoBERTaEmbeddings()

docs = [
    "Beethoven was the grandson of Ludwig van Beethoven (1712–1773), a musician from the town of Mechelen in the Austrian Duchy of Brabant (in what is now the Flemish region of Belgium) who had moved to Bonn at the age of 21.",
    "Ludwig was employed as a bass singer at the court of Clemens August, Archbishop-Elector of Cologne, eventually rising to become, in 1761, Kapellmeister (music director) and thereafter the pre-eminent musician in Bonn.",
    "The portrait he commissioned of himself towards the end of his life remained displayed in his grandson's rooms as a talisman of his musical heritage.[8] Ludwig had one son, Johann (1740–1792), who worked as a tenor in the same musical establishment and gave keyboard and violin lessons to supplement his income",
    "Johann married Maria Magdalena Keverich in 1767; she was the daughter of Johann Heinrich Keverich (1701–1751), who had been the head chef at the court of the Archbishopric of Trier."
    ]
print(len(docs))

def keep_token(t):
    return (t.is_alpha and not (t.is_space or t.is_punct or t.is_stop))

def lemmatize_doc(doc):
    return [ t.lemma_ for t in doc if keep_token(t)]

print("Parsing documents . . .")
parsed_docs = [lemmatize_doc(nlp(str(doc).lower())) for doc in docs]

print("Creating dictionary . . .")
dct = Dictionary(parsed_docs)
print(len(dct))

print("Creating BOW . . .")
docs_corpus = [dct.doc2bow(doc) for doc in parsed_docs]

print("Initializing model . . .")
model_tfidf = TfidfModel(docs_corpus, id2word=dct)


pool = DocumentPoolEmbeddings(
    embeddings = [roberta],
    pooling = 'tfidf_weighted',
    # pooling='mean',
    tfidf_model = model_tfidf,
    dictionary = dct
)

test = lemmatize_doc(nlp(str("The portrait was employed in the Austrian rooms.").lower()))

print(f"test sentence: {test}")
sent = Sentence(' '.join(test))

pool.embed(sent)

print(sent.embedding.shape)