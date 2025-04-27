import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from icecream import ic
from pprint import pprint
import time
import logging
from dotenv import load_dotenv
from tqdm import tqdm

import sys
sys.path.append("/data/npl/ViInfographicCaps/llm_fol/final_contest/XAI")
from utils import load_yml, load_llm, save_json
from src.dataloader import XAIDataset
from src.chat_agent import ChatAgent, Prompt
from src.module.reasoning import UNDERSTAND_BACKGROUND_PROMPT_v2, UNDERSTAND_BACKGROUND_PROMPT_WITHOUT_PREMISE

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %--- LangChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

load_dotenv()

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
        Sampling data based on num_samples (all or specific number)
        """
        samples = []
        num_records = 0
        for item_value in tqdm(self.annotation):
            premises = ' '.join(item_value['premises-NL'])
            fol_premises = '.'.join(item_value['premises-FOL'])
            questions = item_value['questions']
            answers = item_value['answers']
            reasonings = item_value['explanation']
            # logic_program_predicates = item_value['logic_program_predicate_LLM']
            logic_program_predicates = item_value['logic_program_predicates']
            logic_program_premises = item_value['logic_program_premises']
            logic_program_predicate_LLM = item_value['logic_program_predicate_LLM']

            logic_program_premises = item_value['LLM-FOL']

            for q_id, (question, answer, reasoning) in enumerate(zip(questions, answers, reasonings)):
                sub_questions = question.split(', and')
                for sub_question in sub_questions:
                    sample_item = {
                        'id': item_value['id'],
                        'q_id': q_id,
                        'premises': premises,
                        'fol_premises': fol_premises,
                        'conclusion': sub_question.strip(),
                        'reasoning': reasoning,
                        'answer': answer,
                        'logic_program_predicates': logic_program_predicates, 
                        'logic_program_premises': logic_program_premises,
                    }
                    samples.append(sample_item)
                    num_records += 1

            if num_samples != "all" and num_records >= num_samples:
                break
        return samples

def make_reasoning_prompt(lp_predicates_list, lp_premises_list, question, background_prompt):
    """
    Creates reasoning prompt for LLM based on predicates, premises, and question.
    """
    lp_predicates_samples = [
        {"predicate": predicate, "nl_explain": nl_explain} 
        for predicate, nl_explain in parse_logic_program(lp_predicates_list)
    ]

    lp_premises_samples = [
        {"fol": fol, "nl_explain": nl_explain} 
        for fol, nl_explain in parse_logic_program(lp_premises_list)
    ]
    
    # Creating the prompt for predicates
    lp_predicates_obj = Prompt(template=UNDERSTAND_BACKGROUND_PROMPT_v2(), input_variables=["predicate", "nl_explain"])
    lp_predicates_obj.create_fewshot_template(lp_predicates_samples)
    lp_predicates_prompt = lp_predicates_obj.get_prompt({})
    
    # Creating the prompt for premises
    lp_premises_obj = Prompt(template=UNDERSTAND_BACKGROUND_PROMPT_v2(), input_variables=["fol", "nl_explain"])
    lp_premises_obj.create_fewshot_template(lp_premises_samples)
    lp_premises_prompt = lp_premises_obj.get_prompt({})

    # Construct the final instruction prompt
    instruct_prompt_obj = Prompt(template=background_prompt, input_variables=["lp_predicates", "lp_premises"])
    instruct_prompt_obj.create_prompt_template()
    instruct_prompt = instruct_prompt_obj.get_prompt({
        "lp_predicates": lp_predicates_prompt, 
        "lp_premises": lp_premises_prompt,
    })
    
    # Final full prompt for LLM
    final_prompt_obj = Prompt(template=UNDERSTAND_BACKGROUND_PROMPT_v2(), input_variables=["instruct_prompt"])
    final_prompt_obj.create_prompt_template()
    final_prompt = final_prompt_obj.get_prompt({
        "instruct_prompt": instruct_prompt,
        "user_question": question
    })
    
    return final_prompt

class ChatAgentReasoning(ChatAgent):
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
            This **Predicate** {predicate} means: {nl_explain}
        """

        lp_premises_prompt_template = """
            This **FOL** {fol} means: {nl_explain}
        """

        # Logic Program example
        lp_predicates_samples = [{
            "predicate": predicate,
            "nl_explain": nl_explain
        } for predicate, nl_explain in parse_logic_program(lp_predicates_list)]
        
        lp_premises_samples = [{
            "fol": fol,
            "nl_explain": nl_explain
        } for fol, nl_explain in parse_logic_program(lp_premises_list)]

        
        # Input Context
        lp_predicates_samples_obj = Prompt(
            template=lp_predicates_prompt_template,
            input_variables=["predicate", "nl_explain"]
        )
        lp_predicates_samples_obj.create_fewshot_template(
            lp_predicates_samples,
            prefix="")
        lp_predicates_samples_prompt = lp_predicates_samples_obj.get_prompt({})
        
        lp_premises_samples_obj = Prompt(
            template=lp_premises_prompt_template,
            input_variables=["premise", "nl_explain"]
        )
        lp_premises_samples_obj.create_fewshot_template(
            lp_premises_samples,
            prefix="")
        lp_premises_samples_prompt = lp_premises_samples_obj.get_prompt({})
        
        # INSTRUCT PROMPT
        BACKGROUND_PROMPT = UNDERSTAND_BACKGROUND_PROMPT_v2()
        instruct_prompt_obj = Prompt(
            template=BACKGROUND_PROMPT,
            input_variables=['lp_predicates', 'lp_premises']
        )
        instruct_prompt_obj.create_prompt_template()
        instruct_prompt = instruct_prompt_obj.get_prompt({
            'lp_predicates': lp_predicates_samples_prompt, 
            'lp_premises': lp_premises_samples_prompt,
        })

        # FINAL PROMPT
        final_prompt_obj = Prompt(
            template=llama2_chat_prompt_template,
            input_variables=['instruct_prompt']
        )
        final_prompt_obj.create_prompt_template()
        final_prompt = final_prompt_obj.get_prompt({
            'instruct_prompt': instruct_prompt,
            'user_question': question,
        })
        return final_prompt


class ChatAgentReasoningWithoutPremise(ChatAgent):
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
            This **Predicate** {predicate} means: {nl_explain}
        """

        lp_premises_prompt_template = """
            Understand this **FOL**: {fol}
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
            prefix="")
        lp_predicates_samples_prompt = lp_predicates_samples_obj.get_prompt({})
        
        lp_premises_samples_obj = Prompt(
            template=lp_premises_prompt_template,
            input_variables=["fol"]
        )
        lp_premises_samples_obj.create_fewshot_template(
            lp_premises_samples,
            prefix="")
        lp_premises_samples_prompt = lp_premises_samples_obj.get_prompt({})
        
        # INSTRUCT PROMPT
        BACKGROUND_PROMPT = UNDERSTAND_BACKGROUND_PROMPT_WITHOUT_PREMISE()
        # BACKGROUND_PROMPT = UNDERSTAND_BACKGROUND_PROMPT_v2()
        instruct_prompt_obj = Prompt(
            template=BACKGROUND_PROMPT,
            input_variables=['lp_predicates', 'lp_premises']
        )
        instruct_prompt_obj.create_prompt_template()
        instruct_prompt = instruct_prompt_obj.get_prompt({
            'lp_predicates': lp_predicates_samples_prompt, 
            'lp_premises': lp_premises_samples_prompt,
        })

        # FINAL PROMPT
        final_prompt_obj = Prompt(
            template=llama2_chat_prompt_template,
            input_variables=['instruct_prompt']
        )
        final_prompt_obj.create_prompt_template()
        final_prompt = final_prompt_obj.get_prompt({
            'instruct_prompt': instruct_prompt,
            'user_question': question,
        })
        return final_prompt

def reasoning(config, device):
    # Load dataset and model
    logger.info("Loading dataset and model...")
    reasoning_dataset = ReasoningDataset(config['file_path'], num_samples='all')
    model = load_llm(config['model_id'], config['model_config'], config['model_type'], device)

    logger.info("Starting reasoning...")
    chat_agent_reasoning = ChatAgentReasoning(model, config)
    chat_agent_reasoning_without_premises = ChatAgentReasoningWithoutPremise(model, config)

    for i, sample in tqdm(enumerate(reasoning_dataset)):
        if i > 1:  # Limit to the first two samples for testing
            break
        
        logic_program_predicates = sample['logic_program_predicate_LLM']
        logic_program_premises = sample['llm_fol']
        question = sample['conclusion']

        final_prompt = make_reasoning_prompt(
            lp_predicates_list=logic_program_predicates, 
            lp_premises_list=logic_program_premises, 
            question=question,
            background_prompt=UNDERSTAND_BACKGROUND_PROMPT_WITHOUT_PREMISE() 
        )

        reasoning_results = chat_agent_reasoning.inference_direct(prompt=final_prompt)
        logger.info(f"Reasoning result: {reasoning_results['text'].split('### Answer Response:')[-1]}")

def parse_logic_program(logic_programs: list):
    """
    Yield: predicate and corresponding natural language explanations.
    """
    for logic_program in logic_programs:
        pairs = logic_program.split(':::')
        predicate = pairs[0].strip()
        nl = pairs[1].strip() if len(pairs) == 2 else None
        yield predicate, nl

if __name__ == "__main__":
    start_time = time.time()
    args = get_args()
    config = load_yml(args.config)
    config['file_path'] = args.file_path
    reasoning(config, args.device)
    end_time = time.time()
    logger.info(f"Execution time: {end_time - start_time:.2f} seconds")
