from align_reader import *
import pickle
import numpy as np
from sklearn.manifold import TSNE

user_path = '/mounts/work/weissweiler/deepcase/final_data/spacy_3.2_trf/'
alignment_reader = AlignReader()
reference_english_version = "engq"

verses_in_all_langs = pickle.load(open('/mounts/Users/student/weissweiler/deepcase/code_for_publishing/verses_in_all_langs.p', 'rb'))

ignore_langs = ['plt3','kat0','bre0','kor5','tur2','grc8','grc3']
target_langs_with_number = [alignment_reader.lang_prf_map[lang] for lang in alignment_reader.all_langs if lang[:3] != 'eng' and alignment_reader.lang_prf_map[lang] not in ignore_langs]
target_langs_without_number = [code_with_number[:3] for code_with_number in target_langs_with_number]

all_nps = pickle.load(open(user_path + "final_nps.p", "rb"))


all_sents = {lang[:3]: alignment_reader.get_text_for_lang(lang) for lang in target_langs_with_number + [reference_english_version]}
all_sents[reference_english_version] = alignment_reader.get_text_for_lang(reference_english_version)

all_alignments = {lang[:3]: alignment_reader.get_verse_alignment(verses_in_all_langs, reference_english_version, lang) for lang in target_langs_with_number if lang != reference_english_version}

verse_dict = {}
langword_to_id = {}
np_i = 0

for verse_num in verses_in_all_langs:
    for eng_np in all_nps[verse_num][reference_english_version][reference_english_version]:
        eng_list = all_sents[reference_english_version][verse_num].split()
        eng_np_text = [eng_word for eng_i, eng_word in enumerate(eng_list) if eng_i in eng_np]
        verse_dict[np_i] = {}
        verse_dict[np_i]["eng"] = eng_np_text
        for word_in_sentence in eng_np_text:          
            transformed_word = "eng|" + word_in_sentence
            if transformed_word not in langword_to_id:
                langword_to_id[transformed_word] = len(langword_to_id) 
        
        for lang in target_langs_without_number:
            alignments = all_alignments[lang][verse_num]
            lang_aligned = tuple([alignment[1] for alignment in alignments if alignment[0] in eng_np])
            
            if len(lang_aligned) == 0:
                continue

            aligned_with_chunk = tuple(sorted(lang_aligned))
                
            candidate_nps = all_nps[verse_num][reference_english_version][lang]
            
            if aligned_with_chunk in candidate_nps:
                lang_list = all_sents[lang][verse_num].split()
                aligned_list = [lang_list[i] for i in aligned_with_chunk]
                verse_dict[np_i][lang] = aligned_list
                for word_in_sentence in aligned_list:          
                    transformed_word = lang + "|" + word_in_sentence
                    if transformed_word not in langword_to_id:
                        langword_to_id[transformed_word] = len(langword_to_id)
        np_i += 1

# for verse_num in verses_in_all_langs:
#     for eng_np in all_nps[verse_num][reference_english_version]:
#         eng_list = all_sents[reference_english_version][verse_num].split()
#         eng_np_text = [eng_word for eng_i, eng_word in enumerate(eng_list) if eng_i in eng_np]
#         for word_in_sentence in eng_np_text:          
#             transformed_word = "eng|" + word_in_sentence
#             if transformed_word not in langword_to_id:
#                 langword_to_id[transformed_word] = len(langword_to_id) 
        
#         for lang in target_langs_without_number:
#             if lang == reference_english_version:
#                 continue
#             alignments = all_alignments[lang][verse_num]
#             lang_aligned = tuple([alignment[1] for alignment in alignments if alignment[0] in eng_np])
            
#             if len(lang_aligned) == 0:
#                 continue

#             aligned_with_chunk = tuple(sorted(lang_aligned))
                
#             candidate_nps = all_nps[verse_num][lang]
            
#             if aligned_with_chunk in candidate_nps:
#                 lang_list = all_sents[lang][verse_num].split()
#                 aligned_list = [lang_list[i] for i in aligned_with_chunk]
#                 for word_in_sentence in aligned_list:          
#                     transformed_word = lang + "|" + word_in_sentence
#                     if transformed_word not in langword_to_id:
#                         langword_to_id[transformed_word] = len(langword_to_id) 


# for verse_num in verses_in_all_langs:
#     for eng_np in all_nps[verse_num][reference_english_version]:
#         verse_dict[np_i] = {}
#         eng_list = all_sents[reference_english_version][verse_num].split()
#         eng_np_text = [eng_word for eng_i, eng_word in enumerate(eng_list) if eng_i in eng_np]
#         verse_dict[np_i][reference_english_version] = eng_np_text
        
#         for lang in target_langs_without_number:
#             if lang == reference_english_version:
#                 continue
#             alignments = all_alignments[lang][verse_num]
#             lang_aligned = tuple([alignment[1] for alignment in alignments if alignment[0] in eng_np])
            
#             if len(lang_aligned) == 0:
#                 continue

#             aligned_with_chunk = tuple(sorted(lang_aligned))
                
#             candidate_nps = all_nps[verse_num][lang]
            
#             if aligned_with_chunk in candidate_nps:
#                 lang_list = all_sents[lang][verse_num].split()
#                 aligned_list = [lang_list[i] for i in aligned_with_chunk]
#                 verse_dict[np_i][lang] = aligned_list
#         np_i += 1

matrix = np.zeros((np_i, len(langword_to_id)))
for np_id, lang_dict in verse_dict.items():
       for lang, words in lang_dict.items():
            for word in words:
                transformed = lang + "|" + word
                word_id = langword_to_id[transformed]
                matrix[np_id, word_id] += 1


pickle.dump((matrix, verse_dict, langword_to_id), open(user_path + "matrix_small.p", "wb"))

matrix_tsne = TSNE(n_components=2, perplexity=30).fit_transform(matrix)
pickle.dump((matrix_tsne, verse_dict, langword_to_id), open(user_path + "matrix_tsne.p", "wb"))

