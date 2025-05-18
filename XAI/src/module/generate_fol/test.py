
from nl2fol import (
    nl_to_fol_new,
    predicate_nl_extractor
)
import os, torch
from icecream import ic
from typing import Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    pipeline,
)
import re
from typing import List
os.environ["HF_TOKEN"] = "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"  
import spacy
from collections import Counter
nlp = spacy.load("en_core_web_sm")

import re
import argparse
import yaml
from sentence_transformers import SentenceTransformer, util
import os 
import json
import time
import numpy as np
from nltk.parse.corenlp import CoreNLPParser, CoreNLPDependencyParser
from nltk.tree import ParentedTree
import nltk

import subprocess
import os
mapping_model = SentenceTransformer("all-MiniLM-L6-v2")
pipeline_llm = pipeline(
    model="/workspace/hf_model/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=2048 
)
pipeline_llm.tokenizer.pad_token     = pipeline_llm.tokenizer.eos_token
fol_converter = nl_to_fol_new(pipeline_llm)
extractor     = predicate_nl_extractor(pipeline_llm, mapping_model, threshold=0.7)

premises   =  [
            "All students have completed all quizzes.",
            "If x finishes all quizzes, then x is eligible for the final exam.",
            "If x completes all course requirements, then x is eligible for the final exam.",
            "If x has not completed all course requirements, then x has not completed all quizzes.",
            "If x has not finished all quizzes, then x is not eligible for the final exam."
        ]
conclusion = [
            "Which of the following can be inferred about students based on the premises?\nA. All students are eligible for the final exam.\nB. All students have not completed course requirements.\nC. No students are eligible for the final exam.\nD. Some students complete quizzes but are not eligible.",
        ]
prem_pred_dic, pred_list, sub_dic, sub_list = extractor.extract(premises, conclusion)
print('prem_pred_dic: ',prem_pred_dic)
print('pred_list: ',pred_list)
print('sub_dic: ',sub_dic)
print('sub_list: ',sub_list)
dic_pred = fol_converter.convert(pred_list)
fol_official  = fol_converter.convert_premise_to_fol(
    premise_nl_list   = premises,
    premise_nl_pred_dict   = prem_pred_dic,
    dic_predicates    = dic_pred,
    premise_nl_subject= sub_dic
)


print(fol_official)
