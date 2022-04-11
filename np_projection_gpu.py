from align_reader import *
import spacy
from spacy.tokens import Doc
import pickle
import multiprocessing as mp

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


def clean_alignments(target_list):
    if len(target_list) == 1:
        return target_list

    last_index = target_list[0]
    for index in target_list[1:]:
        if index != last_index+1:
            return None
        last_index = index

    return target_list


user_path = '/mounts/work/weissweiler/deepcase/final_data/spacy_3.2_trf/'
alignment_reader = AlignReader()

verses_in_all_langs = pickle.load(open('/mounts/Users/student/weissweiler/deepcase/code_for_publishing/verses_in_all_langs.p', 'rb'))

ignore_eng_editions = ['engt','enga','eng3', 'eng7']
ignore_langs = ['plt3','kat0','bre0','kor5','tur2','grc8','grc3']
eng_editions = [alignment_reader.lang_prf_map[lang] for lang in alignment_reader.all_langs if lang[:3] == 'eng' and alignment_reader.lang_prf_map[lang] not in ignore_eng_editions]
target_langs_with_number = [alignment_reader.lang_prf_map[lang] for lang in alignment_reader.all_langs if lang[:3] != 'eng' and alignment_reader.lang_prf_map[lang] not in ignore_langs]
target_langs_without_number = [code_with_number[:3] for code_with_number in target_langs_with_number]


eng_sents = {eng_edition: alignment_reader.get_text_for_lang(eng_edition) for eng_edition in eng_editions}

spacy_engine = spacy.load('en_core_web_trf')
spacy_engine.remove_pipe('ner')
spacy_engine.remove_pipe('lemmatizer')
spacy_engine.tokenizer = WhitespaceTokenizer(spacy_engine.vocab)

all_alignments = {(eng_edition, target_lang[:3]):alignment_reader.get_verse_alignment(verses_in_all_langs, eng_edition, target_lang) for eng_edition in eng_editions for target_lang in target_langs_with_number}

index_to_verse_num = list(sorted(verses_in_all_langs))
verse_num_to_index = {verse_num:index for index, verse_num in enumerate(index_to_verse_num)}

all_docs = {}
for eng_edition in eng_editions:
    eng_edition_texts = [eng_sents[eng_edition][verse_num] for verse_num in index_to_verse_num]
    print(eng_edition)
    if eng_edition == "engi":
        eng_edition_texts[verse_num_to_index["43017017"]] = ' '.join(eng_edition_texts[verse_num_to_index["43017017"]].split())
    eng_edition_docs = spacy_engine.pipe(eng_edition_texts, batch_size=2000)
    verse_num_to_doc = {index_to_verse_num[i]:doc for i, doc in enumerate(eng_edition_docs)}
    all_docs[eng_edition] = verse_num_to_doc

counter = mp.Value('i', 0)
def process_verse(verse_num):
    
    global counter
    with counter.get_lock():
        counter.value += 1
    print(counter.value)

    final_nps_verse = {}

    for eng_edition in eng_editions:

        final_nps_verse[eng_edition] = {}
           
        eng_text = eng_sents[eng_edition][verse_num]

        if verse_num == "43017017" and eng_edition == "engi":
            eng_text = " ".join(eng_text.split())
        
        doc = all_docs[eng_edition][verse_num]

        for chunk in doc.noun_chunks:

            for target_lang in target_langs_without_number:
                
                alignments = all_alignments[(eng_edition, target_lang)][verse_num]

                aligned_with_chunk_set = set([alignment[1] for alignment in alignments if chunk.start <= alignment[0] < chunk.end])

                aligned_with_chunk = tuple(sorted(aligned_with_chunk_set))

                if target_lang not in final_nps_verse[eng_edition]:
                    final_nps_verse[eng_edition][target_lang] = set()

                final_nps_verse[eng_edition][target_lang].add(aligned_with_chunk)
                   
    
    for eng_edition in eng_editions:
        doc = all_docs[eng_edition][verse_num]

        final_nps_verse[eng_edition][eng_edition] = set()

        for chunk in doc.noun_chunks:
            chunk_indices = tuple(range(chunk.start, chunk.end))
            final_nps_verse[eng_edition][eng_edition].add(chunk_indices)


    final_nps[verse_num] = final_nps_verse
        


manager = mp.Manager()
final_nps = manager.dict()
pool = mp.Pool(mp.cpu_count())
pool.map(process_verse, [verse_num for verse_num in verses_in_all_langs])

pool.close()
pool.join()

f = open(user_path + 'final_nps.p', 'wb')
pickle.dump(dict(final_nps), f)
f.close()
