import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from icecream import ic
from pprint import pprint
import time

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
from utils import load_yml, load_llm, load_json, save_json
from src.dataloader import XAIDataset, load_dataloader 
from src.chat_agent import ChatAgent, Prompt
from src.module.reasoning import REDUCE_AND_MATCHING_PREDICATE_PROMPT_CONVERT_FOL2FOL
load_dotenv()

'''
    Matching predicate from context to question, and then extract predicate of question and trackback context 
       
'''

def get_args():
    parser = argparse.ArgumentParser(description="Load model config and run something")
    
    parser.add_argument('--file_path', type=str, required=True, help='Path to Reasoning Json File')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--device', type=int, required=True, default='cuda:0', help='Path to YAML config file')
    
    return parser.parse_args()


class ReasoningDataset(XAIDataset):
    

    def __init__(self, annotation_path, num_samples='all'):
        super().__init__(annotation_path, num_samples)

    def sampling(self, num_samples):
        """
            numsamples: str or int
                "all": select all
                int: select number 
        """
        samples = []
        num_records = 0
        data = self.annotation
        for id in tqdm(range(len(data))):
            item_value = data[id]
            premises = ' '.join(item_value['premises-NL'])
            fol_premises = '.'.join(item_value['premises-FOL'])
            questions = item_value['questions']
            answers = item_value['answers']
            reasonings = item_value['explanation']
            logic_program_predicates = item_value['logic_program_predicates']
            logic_program_premises = item_value['logic_program_premises']
            logic_program_predicate_LLM = item_value['logic_program_predicate_LLM']
            llm_fol = item_value['LLM-FOL']

            # Create samples
            for q_id, (question, answer, reasoning) in enumerate(zip(questions, answers, reasonings)):
                sub_questions = question.split(', and')
                for sub_question in sub_questions:
                    sample_item = {
                        'id': id,
                        'q_id': q_id,
                        'premises': premises,
                        'fol_premises': fol_premises,
                        'conclusion': sub_question.strip(),
                        'reasoning': reasoning,
                        'answer': answer,
                        'logic_program_predicates': logic_program_predicates, 
                        'logic_program_premises': logic_program_premises,
                        'logic_program_predicate_LLM': logic_program_predicate_LLM, 
                        'llm_fol': llm_fol, 
                    }
                    samples.append(sample_item)
                num_records += 1
            

                if num_samples != "all" and num_records >= num_samples:
                    return samples
        return samples


class ChatAgentReduceWithPremise(ChatAgent):
    def __init__(self, model, config):
        super().__init__(model, config)

    def make_prompt(self, lp_predicates_list, lp_premises_list, question):
        # PROMPT TEMPLATE
        llama2_chat_prompt_template = """
            <s>[INST] <<SYS>>
            ### Instruction:
            {instruct_prompt}

            <</SYS>>
            ### Question
            {user_question} [/INST]
        """

        lp_predicates_prompt_template = """
            **Predicate**: {predicate} ::: {nl_explain}
        """

        lp_premises_prompt_template = """
            **FOL**: {fol}
        """

        # Logic Program example
        lp_predicates_samples = [{
            "predicate": predicate,
            "nl_explain": nl_explain
        } for predicate, nl_explain in parse_logic_program(lp_predicates_list)]
        
        lp_premises_samples = [{
            "fol": fol,
        } for fol, nl_explain in parse_logic_program(lp_premises_list)]


        # Input Context
        lp_predicates_samples_obj = Prompt(
            template=lp_predicates_prompt_template,
            input_variables=["predicate", "nl_explain"]
        )
        lp_predicates_samples_obj.create_fewshot_template(
            lp_predicates_samples,
            prefix="List of **Predicates**")
        lp_predicates_samples_prompt = lp_predicates_samples_obj.get_prompt({})

        lp_premises_samples_obj = Prompt(
            template=lp_premises_prompt_template,
            input_variables=["fol"]
        )
        lp_premises_samples_obj.create_fewshot_template(
            lp_premises_samples,
            prefix="List of **FOLs**")
        lp_premises_samples_prompt = lp_premises_samples_obj.get_prompt({})
        
        # INSTRUCT PROMPT
        BACKGROUND_PROMPT = REDUCE_AND_MATCHING_PREDICATE_PROMPT_CONVERT_FOL2FOL()
        # FINAL PROMPT
        final_prompt_obj = Prompt(
            template=llama2_chat_prompt_template,
            input_variables=['instruct_prompt']
        )
        final_prompt_obj.create_prompt_template()
        final_prompt = final_prompt_obj.get_prompt({
            'instruct_prompt': BACKGROUND_PROMPT,
            'user_question': f'{lp_predicates_samples_prompt}\n{lp_premises_samples_prompt}',
        })
        return final_prompt


def parse_logic_program(logic_programs: list):
    '''
        Yield: predicate, natural languages
    '''
    for logic_program in logic_programs:
        pairs = logic_program.split(':::')
        predicate = pairs[0].strip()
        nl = pairs[1].strip() if len(pairs) == 2 else None
        yield predicate, nl


def reducing(config, device):
    # Load dataset path
    print("Load dataset")
    reasoning_dataset = ReasoningDataset(config['file_path'], num_samples='all')
    
    # Load ChatAgent
    model = load_llm(
        model_id=config['model_id'],
        config=config['model_config'],
        model_type=config['model_type'],
        device=device,
    )

    print("Chain example")
    chat_agent_reducing = ChatAgentReduceWithPremise(model, config)
    # chat_agent_reasoning_without_premises = ChatAgentReduce(model, config)
    
    for i in range(len(reasoning_dataset)):
        if i != 1:
            continue

        # Logic Programs
        logic_program_predicates = reasoning_dataset[i]['logic_program_predicate_LLM']
        logic_program_premises = reasoning_dataset[i]['llm_fol']

        lp_predicates_list = logic_program_predicates
        lp_premises_list = logic_program_premises
        
        # Input question
        question = reasoning_dataset[i]['conclusion']
        
        # final_prompt = chat_agent_reasoning.make_prompt(
        final_prompt = chat_agent_reducing.make_prompt(
            lp_predicates_list=lp_predicates_list,
            lp_premises_list=lp_premises_list,
            question=question
        )
        # ic(final_prompt)
        reasoning_results = chat_agent_reducing.inference_direct(
            prompt=final_prompt,
        )
        ic(reasoning_results['text'].split("### Answer Response:")[-1])
        tokens = final_prompt.split(" ")
        ic(len(tokens))
        tokens.remove("")
        ic(len(tokens))

if __name__=="__main__":
    begin = time.time()
    args = get_args()
    config = load_yml(args.config)
    config['file_path'] = args.file_path
    reducing(config, args.device)
    # check_multiple_question(config, args.device)
    end = time.time()
    execute_time = end - begin
    ic(execute_time)
