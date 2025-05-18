# Format dataset by data/train_v2_ful_fol.json records
#---------------------------------
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
from utils import load_yml, load_llm, load_json, save_json, extract_predicate_from_fol
from src.dataloader import XAIDataset, load_dataloader 
from src.chat_agent import ChatAgent, Prompt
from src.module.reasoning import LOGIC_PROGRAM_EXTRACTION_PROMPTING, LOGIC_PROGRAM_EXTRACTION_PROMPTING_NEW
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

class ReasoningDatasetFullFol(XAIDataset):
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
            idx = item_value['idx']
            reasonings = item_value['explanation']
            llm_fol = item_value['LLM-FOL']
            ques_fols = item_value['question-FOL']


            # Create samples
            for q_id, (question, answer, reasoning, ques_fol) in enumerate(zip(questions, answers, reasonings, ques_fols)):
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
                        'llm_fol': llm_fol, 
                        'ques_fol': ques_fol, 
                    }
                    samples.append(sample_item)
                num_records += 1
            

                if num_samples != "all" and num_records >= num_samples:
                    return samples
        return samples


class ChatAgentExtractLogicProgram(ChatAgent):
    def __init__(self, model, config):
        super().__init__(model, config)

    def make_prompt(self, premises, list_predicates_premise, list_predicates_question):
        # PROMPT TEMPLATE
        llama2_chat_prompt_template = """
            <s>[INST] <<SYS>>
            ### Instruction:
            {instruct_prompt}

            <</SYS>>
            ### Question
            {user_question} [/INST]
        """

        # INPUT
        #----1
        pair_predicate_premise_template = """
            **Predicates {id}**: {predicates}
            **Premise {id}**: {premise} 
        """
        pair_predicate_premise_samples = [{
            "id": id + 1, 
            "predicates": ", ".join(predicates),
            "premise": premise,
        } for id, (premise, predicates) in enumerate(zip(premises, list_predicates_premise))]
        pair_predicate_premise_obj = Prompt(
            template=pair_predicate_premise_template,
            input_variables=["id", "question"]
        )
        pair_predicate_premise_obj.create_fewshot_template(
            pair_predicate_premise_samples,
            prefix=""
        )
        pair_predicate_premise_prompt = pair_predicate_premise_obj.get_prompt({})

        #----2
        pair_predicate_question_template = """
            **Question-Predicates {id}:**: {q_predicates}
        """
        pair_predicate_question_samples = [{
            "id": id + 1, 
            "q_predicates": ", ".join(predicates)
        } for id, predicates in enumerate(list_predicates_question)]
        pair_predicate_question_obj = Prompt(
            template=pair_predicate_question_template,
            input_variables=["id", "question"]
        )
        pair_predicate_question_obj.create_fewshot_template(
            pair_predicate_question_samples,
            prefix=""
        )
        pair_predicate_question_prompt = pair_predicate_question_obj.get_prompt({})

        # INSTRUCTION PROMPT
        INSTRUCTION_PROMPT = LOGIC_PROGRAM_EXTRACTION_PROMPTING_NEW()

        # FINAL PROMPT
        final_prompt_obj = Prompt(
            template=llama2_chat_prompt_template,
            input_variables=['instruct_prompt', 'user_question']
        )
        final_prompt_obj.create_prompt_template()

        # final_prompt = final_prompt_obj.get_prompt({
        #     'instruct_prompt': INSTRUCTION_PROMPT, 
        #     'user_question': f"{pair_predicate_premise_prompt}\n{pair_predicate_question_prompt}",
        # })

        final_prompt = final_prompt_obj.get_prompt({
            'instruct_prompt': INSTRUCTION_PROMPT, 
            'user_question': pair_predicate_premise_prompt,
        })
        return final_prompt


def check_multiple_question(config, device):
    # Load dataset path
    print("Load dataset")
    reasoning_dataset = XAIDataset(config['data']['train'], num_samples='all')

    for i in range(len(reasoning_dataset)):
        ques = reasoning_dataset[i]['conclusion']
        answer = reasoning_dataset[i]['answer']
        if len(answer.split(',')) >= 2:
            ic(ques, answer)


def parse_logic_program(logic_programs: list):
    '''
        Yield: predicate, natural languages
    '''
    for logic_program in logic_programs:
        pairs = logic_program.split(':::')
        predicate = pairs[0].strip()
        nl = pairs[1].strip() if len(pairs) == 2 else None
        yield predicate, nl


def extract_lp(config, device):
    save_dir = 'save'
    save_path = os.path.join(save_dir, 'logic_program.json')
    # Load dataset path
    print("Load dataset")
    reasoning_dataset = ReasoningDatasetFullFol(config['file_path'], num_samples='all')
    
    # Load ChatAgent
    model = load_llm(
        model_id=config['model_id'],
        config=config['model_config'],
        model_type=config['model_type'],
        device=device,
    )

    print("Chain example")
    chat_agent_extract_lp = ChatAgentExtractLogicProgram(model, config)  
    for i in range(len(reasoning_dataset)):
        if i != 1:
            continue

        # Logic Programs
        premises = reasoning_dataset[-1]['premises'].split(". ") # List
        questions = reasoning_dataset[-1]['conclusion']
        llm_fol = reasoning_dataset[-1]['llm_fol']
        ques_fol = reasoning_dataset[-1]['ques_fol']
        ic(questions)
        if type(questions) != list:
            questions = [questions]
        if type(ques_fol) != list:
            ques_fol = [ques_fol]
        
        llm_fol.extend(ques_fol)
        premises.extend(questions)

        if "" in llm_fol:
            llm_fol.remove("")
        if "" in premises:
            premises.remove("")

        ic(premises)
        ic(len(llm_fol), len(premises))

        list_predicates_premise = [extract_predicate_from_fol(fol) for fol in llm_fol]
        list_predicates_question = [extract_predicate_from_fol(fol) for fol in ques_fol]
        
        # Input question
        # final_prompt = chat_agent_extract_lp.make_prompt(
        final_prompt = chat_agent_extract_lp.make_prompt(
            premises=premises,
            list_predicates_premise=list_predicates_premise,
            list_predicates_question=list_predicates_question,
        )
        ic(final_prompt)
        raise
        extract_lp_results = chat_agent_extract_lp.inference_direct(
            prompt=final_prompt,
        )
        output = extract_lp_results['text'].split("<</INST>>")[-1]
        
        ### Post processing

        
        ic(output)
        raise
        tokens = final_prompt.split(" ")
        ic(len(tokens))
        tokens.remove("")
        ic(len(tokens))

if __name__=="__main__":
    begin = time.time()
    args = get_args()
    config = load_yml(args.config)
    config['file_path'] = args.file_path
    extract_lp(config, args.device)
    # check_multiple_question(config, args.device)
    end = time.time()
    execute_time = end - begin
    ic(execute_time)
