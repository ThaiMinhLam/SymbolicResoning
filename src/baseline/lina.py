import json
import torch
import argparse
import numpy as np
import pandas as pd
from icecream import ic

from dotenv import load_dotenv, dotenv_values 
from tqdm import tqdm

# %--- LangChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder,
)
import sys
sys.path.append("/data/npl/ViInfographicCaps/Contest/final_contest/XAI")
from utils import load_yml, load_llm
from src.dataloader import XAIDataset, load_dataloader 
from src.chat_agent import ChatAgent
from module.lina import PARAPHRASE_PROMPT
load_dotenv()

'''
    LINA MODEL Explain:
    -  Apply hypothetical-deductive method (LLM + FOL) on "question"
    
    HAVING 2 MODULES:
    ----
    -  Information Extraction: (1)
        + 
            > (FOL-context, question, options) -> Key information extraction and transformation
            <  Reasoning Tuple - (LS, NL, H) - (logical statements, natural language information, hypotheses)
        
        +   
            Convert (logical statement) -> FOL (with rules) 
            Tránh được các trường hợp ko học được các FOL rút ra từ FOL có sẵn (Avoid Information Loss)
        
        + Logical Statement: Context là một đoạn dài, rút ngắn đoạn text đó:
            Categorizeed based-on the ease of translation to FOL
            Convert to FOL, annotated by LS = [ls1, ...]

    

    ----
    -  LLM-driven Symbolic Reasoning:
        +   Take (1) as input -> Resoning -> Answer
        +   Take (1) -> LLM step-by-step reasoning using hypothesis H -> Get the relevant (LS, NL) -> Answer Ci
        +   Supervisor check for error in reasoning process and adjust C or reset C = H, decide:
            >   Continue: C is conflict with H
            >   Stop    : C supported by (LS or NL - Proving H)
        +   If
            > Continue: Update H' = C
            > Continue reasoning untils reach final answer
        +   Reasoning -> Check xem câu đã reason có conflict hay không  (Không) -> Tiếp tục reason đến khi Done 
                                                                        (Có)    -> Stop                                                 
'''

def get_args():
    parser = argparse.ArgumentParser(description="Load model config and run something")
    
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--device', type=str, required=True, default='cuda:0', help='Path to YAML config file')
    
    return parser.parse_args()


class ChatAgentCoT(ChatAgent):
    def __init__(self, args):
        super().__init__(args)



def main(config):
    print("Load dataset")
    # Load dataset
    train_dataset = XAIDataset(config['data']['train'], num_samples='all')
    val_dataset = XAIDataset(config['data']['val'], num_samples=100)
    test_dataset = XAIDataset(config['data']['test'], num_samples=100)

    train_dataloader = load_dataloader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
    )
    val_dataloader = load_dataloader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=True,
    )
    test_dataloader = load_dataloader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
    ) 

    print("Load Chat Agent")
    # Load ChatAgent
    chat_agent = ChatAgentCoT(config)
    fewshot_examples = train_dataset.data[:2]
    cot_prompt = chat_agent.make_prompt(fewshot_examples)

    print("Chain example")
    # Inference one sample:
    for list_id, list_q_id, list_premises, list_fol_premises, list_conclusion, list_reasoning, list_judgement in train_dataloader:
        input_values = {
            'q_question': list_conclusion[0], # Take the first element in batch
            'q_premises': list_premises[0],  # Take the first element in batch
        }
        
        results = chat_agent.inference(
            prompt=cot_prompt,
            input_values=input_values,
        )

        print(results['text'])
        break

if __name__=="__main__":
    args = get_args()
    config = load_yml(args.config)
    main(config)
