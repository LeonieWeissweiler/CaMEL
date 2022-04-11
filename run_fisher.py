import pickle
from scipy.stats import fisher_exact
from collections import Counter
from tqdm import tqdm
import os
from align_reader import *
import multiprocessing as mp

user_path = '/mounts/work/weissweiler/deepcase/final_data/itermax_bible/'
alignment_reader = AlignReader()


ignore_langs = ['plt3','kat0','bre0','kor5','tur2','grc8','grc3']
# langs = [alignment_reader.lang_prf_map[lang][:3] for lang in alignment_reader.all_langs if lang[:3] != 'eng' and alignment_reader.lang_prf_map[lang] not in ignore_langs]
langs = [alignment_reader.lang_prf_map[lang] for lang in alignment_reader.all_langs if lang[:3] != 'eng' and alignment_reader.lang_prf_map[lang] not in ignore_langs]

def process_lang(lang):
    ngrams_with_fisher_lang = {}

    if not os.path.isfile(user_path + 'ngram_freqs_inside/ngram_freqs_inside_' + lang + '.p'):
        del ngrams_with_fisher[lang]
        print(lang, "not found")
        return
    
    ngram_freqs_inside, total_inside = pickle.load(open(user_path + 'ngram_freqs_inside/ngram_freqs_inside_' + lang + '.p', 'rb'))
    ngram_freqs_outside, total_outside = pickle.load(open(user_path + 'ngram_freqs_outside/ngram_freqs_outside_' + lang + '.p', 'rb'))

    ngram_freqs_inside = Counter(ngram_freqs_inside)
    ngram_freqs_outside = Counter(ngram_freqs_outside)

    all_ngrams = ngram_freqs_inside.keys() | ngram_freqs_outside.keys()
    
    for ngram in tqdm(all_ngrams):

        freq_inside = ngram_freqs_inside[ngram]
        freq_outside = ngram_freqs_outside[ngram]
        
        ngram_np = freq_inside
        not_ngram_np = total_inside - freq_inside
        ngram_not_np = freq_outside
        not_ngram_not_np = total_outside - freq_outside

        oddsratio, pvalue = fisher_exact([[ngram_np, ngram_not_np], [not_ngram_np, not_ngram_not_np]])
        
        ngrams_with_fisher_lang[ngram] = (freq_inside, oddsratio, pvalue)
                
    ngrams_with_fisher[lang] = ngrams_with_fisher_lang
    

manager = mp.Manager()
ngrams_with_fisher = manager.dict({lang: {} for lang in langs})
pool = mp.Pool(mp.cpu_count())

pool.map(process_lang, [lang for lang in langs])
pool.close()
pool.join()


pickle.dump(dict(ngrams_with_fisher), open(user_path + 'ngrams_with_fisher.p', 'wb'))