from flair.models import SequenceTagger
from flair.data import Sentence

tagger = SequenceTagger.load('ner')
sentence = Sentence('George Washington went to Washington .')

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence.get_spans('ner'))

