#!/usr/bin/python3
from align_reader import *
import pickle

ignore_eng = ['eng1', 'eng3', 'eng7', 'enga', 'engd', 'engl', 'engs', 'engt', 'engw']
ignore_target_langs = ['plt3','kat0','bre0','kor5','tur2','grc8','grc3'] 
if __name__ == '__main__':
	alignment_reader = AlignReader()
	verse_sets = []

	for lang in sorted(alignment_reader.all_langs):
		lang_code = ar.lang_prf_map[lang]
		if lang_code in ignore_eng or lang_code in ignore_target_langs:
			continue
		verses = ar.get_text_for_lang(lang_code)
		verse_nums = set(verses.keys())
		verse_sets.append(verse_nums)
		print(lang_code, len(verse_nums), len(set.intersection(*verse_sets)))

	verses_in_all_langs = set.intersection(*verse_sets)

	pickle.dump(verses_in_all_langs, open('verses_in_all_langs.p', 'wb'))