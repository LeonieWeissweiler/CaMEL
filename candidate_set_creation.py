import pickle
from align_reader import *
import numpy as np
from collections import Counter
import multiprocessing as mp
from tqdm import tqdm
import numpy as np


user_path = '/mounts/work/weissweiler/deepcase/final_data/normal_bible/'
final_nps = pickle.load(open(user_path + 'final_nps.p', 'rb'))
alignment_reader = AlignReader()
verses_in_all_langs = pickle.load(open('verses_in_all_langs.p', 'rb'))

ignore_eng_editions = ['engt','enga','eng3', 'eng7']
ignore_langs = ['plt3','kat0','bre0','kor5','tur2','grc8','grc3']
eng_editions = [alignment_reader.lang_prf_map[lang] for lang in alignment_reader.all_langs if lang[:3] == 'eng' and alignment_reader.lang_prf_map[lang] not in ignore_eng_editions]
target_langs_with_number = [alignment_reader.lang_prf_map[lang] for lang in alignment_reader.all_langs if lang[:3] != 'eng' and alignment_reader.lang_prf_map[lang] not in ignore_langs]
target_langs_without_number = [code_with_number[:3] for code_with_number in target_langs_with_number]
texts = {lang: alignment_reader.get_text_for_lang(lang) for lang in target_langs_with_number}

def get_ngrams(word):
    ngrams = set()
    word = '$' + word.lower() + '$'
    word_length = len(word)
    for first_i in range(word_length):
        for second_i in range(first_i+1, word_length+1):

            if first_i == 1:
                continue
            if second_i == word_length-1:
                continue

            ngram = word[first_i:second_i]
            if ngram != '$':
                ngrams.add(ngram)

    return ngrams


def process_lang(lang):
    inside_word_editions = Counter()
    outside_word_editions = Counter()
    edition_inside_sets = {}
    edition_outside_sets = {}

    word_freqs_inside = Counter()
    word_freqs_outside = Counter()

    ngram_freqs_inside = Counter()
    ngram_freqs_outside = Counter()

    for verse_num in tqdm(verses_in_all_langs):
        for eng_edition in eng_editions:
            if lang in final_nps[verse_num][eng_edition]:
                if eng_edition not in edition_inside_sets:
                    edition_inside_sets[eng_edition] = set()
                    edition_outside_sets[eng_edition] = set()

                verse_lang_nps = final_nps[verse_num][eng_edition][lang]
                verse_text = texts[lang][verse_num]
                verse_list = verse_text.split()

                np_bools = np.array([any([True if word_i in np else False for np in verse_lang_nps]) for word_i in range(len(verse_list))])

                for word, in_np in zip(verse_list, np_bools):
                    if in_np:
                        word_freqs_inside[word] += 1
                        if word not in edition_inside_sets[eng_edition]:
                            edition_inside_sets[eng_edition].add(word)
                            inside_word_editions[word] += 1

                    else:
                        word_freqs_outside[word] += 1
                        if word not in edition_outside_sets[eng_edition]:
                            edition_outside_sets[eng_edition].add(word)
                            outside_word_editions[word] += 1

    total_inside_freqs = sum(word_freqs_inside.values())
    total_outside_freqs = sum(word_freqs_outside.values())

    words_inside = {k for k,v in word_freqs_inside.items() if v/total_inside_freqs > word_freqs_outside[k]/total_outside_freqs}
    words_outside = {k for k,v in word_freqs_inside.items() if v/total_inside_freqs < word_freqs_outside[k]/total_outside_freqs}

    for word in words_inside:
        new_ngrams = get_ngrams(word)
        for ngram in new_ngrams:
            ngram_freqs_inside[ngram] += 1           

    for word in words_outside:
        new_ngrams = get_ngrams(word)
        for ngram in new_ngrams:
            ngram_freqs_outside[ngram] += 1
               
    pickle.dump((ngram_freqs_inside,len(words_inside)), open(user_path + 'ngram_freqs_inside/ngram_freqs_inside_' + lang + '.p', 'wb'))
    pickle.dump((ngram_freqs_outside, len(words_outside)), open(user_path + 'ngram_freqs_outside/ngram_freqs_outside_' + lang + '.p', 'wb'))


manager = mp.Manager()
pool = mp.Pool(mp.cpu_count())
pool.map(process_lang, [lang for lang in target_langs_without_number])
pool.close()
pool.join()