# 🐫 Code and Data for CaMEL: Case Marker Extraction without Labels

This repository contains the code for [Code and Data for CaMEL: Case Marker Extraction without Labels](https://doi.org/10.48550/arxiv.2203.10010)

## 🔧 Pipeline

### Required Data

Running this code requires the Parallel Bible Corpus, aligned with SimAlign. It can easily be modified to work with a different parallel dataset with alignments by editing np_projection.py or np_projection_gpu.py

### Required Packages

This pipeline requires spacy, numpy, json, pickle, scikitlearn, and optionally matplotlib to reproduce the figure from the paper. Using np_projection_gpu.py to use spacy's transformer-based pipeline additionally requires huggingface-transformers.

### Commands

For the entire pipeline, run:

```
python3 np_projection_gpu.py;
python3 candidate_set_creation.py;
python3 run_fisher.py;
python3 prepare_tsne.py;
```

The final set of case markers can then be produced with hyperparameter_search.ipynb and the map of ngrams for Polish and Latin can be created with create_case_marker_map.ipynb

## 📄 Silver Standard

We provide our silver standard in silver_standard.p 
If you want to re-run the creation procedure, you can download the [UniMorph data](https://unimorph.github.io/) and run create_silver_standard.ipynb

## 📕 Citation

If you want to make use of the code in this repository or the resulting data, please cite the following paper:

@misc{https://doi.org/10.48550/arxiv.2203.10010,
  url = {https://arxiv.org/abs/2203.10010},
  author = {Weissweiler, Leonie and Hofmann, Valentin and Sabet, Masoud Jalili and Schütze, Hinrich},
  title = {CaMEL: Case Marker Extraction without Labels}, 
  year = {2022},
}
