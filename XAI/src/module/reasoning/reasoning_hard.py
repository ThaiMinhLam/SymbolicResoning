import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from icecream import ic
from pprint import pprint
import time
import re
import random
from typing import Dict, List

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
from src.chat_agent import ChatAgent, Prompt
from src.module.reasoning.template import MULTIPLE_CHOICE_PROMPT_EN, OPEN_QUESTION_PROMPT_EN, YES_NO_PROMPT_EN
load_dotenv()
'''
    Matching predicate from context to question, and then extract predicate of question and trackback context 
       
'''
def parse_logic_program(logic_programs: list):
    '''
        Yield: predicate, natural languages
    '''
    for logic_program in logic_programs:
        pairs = logic_program.split(':::')
        predicate = pairs[0].strip()
        nl = pairs[1].strip() if len(pairs) == 2 else None
        yield predicate, nl

def check_multiple_choice_ques(question: str):
    if re.findall(r"\n[A-D][\.\)]? (.*?)(?=\n[A-D][\.\)]? |\Z)", question):
        return True
    return False

yes_no_patterns = r"^(Do|Does|Did|Is|Are|Was|Were|Have|Has|Had|Can|Could|Shall|Should|Will|Would|May|Might|Must)\b.*\?$"

def check_yes_no_question(sentence: str) -> bool:
    sentence = sentence.strip()
    if "following statement true" in sentence.lower():
        return True
    if re.match(yes_no_patterns, sentence, re.IGNORECASE):
        return True
    return False


class ChatAgentReasoning(ChatAgent):
    def __init__(self, model, config):
        super().__init__(model, config)

    def make_prompt(self,
                    lp_predicates_list: list[str],
                    lp_premises_list: list[str],
                    question: str) -> str:

        # 1) Khung chat dành cho Llama2
        llama2_chat_prompt_template = """
            <s>[INST] <<SYS>>
            {system_and_instruction}
            <</SYS>> [/INST]
        """

        # 2) Template cho từng predicate
        lp_predicates_prompt_template = "- **Predicate** {predicate} means: {nl_explain}"

        # 3) Template cho từng FOL premise
        lp_premises_prompt_template = "- Understand this **FOL**: {fol}"

        # 4) Tạo few-shot examples
        predicate_examples = [
            {"predicate": predicate, "nl_explain": nl_explain}
            for predicate, nl_explain in parse_logic_program(lp_predicates_list)
        ]
        premise_examples   = [
            {"fol": f"P{idx}. {fol}", "nl_explain": nl_explain}
            for idx, (fol, nl_explain) in enumerate(parse_logic_program(lp_premises_list), start=1)
        ]

        # 5) Build prompt text cho predicates
        pred_prompt = Prompt(
            template=lp_predicates_prompt_template,
            input_variables=["predicate", "nl_explain"]
        )
        pred_prompt.create_fewshot_template(
            predicate_examples,
            prefix="Logic Program Predicates:\n"
        )
        lp_predicates_block = pred_prompt.get_prompt({})

        # 6) Build prompt text cho premises
        prem_prompt = Prompt(
            template=lp_premises_prompt_template,
            input_variables=["fol", "nl_explain"]
        )
        prem_prompt.create_fewshot_template(
            premise_examples,
            prefix="First-Order Logic Premises:\n"
        )
        lp_premises_block = prem_prompt.get_prompt({})

        # 7) Chọn template phù hợp cho loại câu hỏi
        if check_multiple_choice_ques(question):
            background = MULTIPLE_CHOICE_PROMPT_EN()
        elif check_yes_no_question(question):
            background = YES_NO_PROMPT_EN()
        else:
            background = OPEN_QUESTION_PROMPT_EN()

        # 8) Đưa vào hệ thống + hướng dẫn chi tiết
        system_and_instruction = PromptTemplate(
            input_variables=["lp_predicates", "lp_premises", "question"],
            template=background
        ).format(
            lp_predicates=lp_predicates_block,
            lp_premises=lp_premises_block,
            question=question
        )

        # Final chat prompt
        final_prompt = PromptTemplate(
            input_variables=["system_and_instruction"],
            template=llama2_chat_prompt_template
        ).format(system_and_instruction=system_and_instruction)

        return final_prompt

def extract_sections(text: str, marker) -> Dict[str, str]:
    idx = text.find(marker)
    if idx == -1:
        return ""  
    start = idx + len(marker)
    end = text.find("\n", start)
    if end == -1:
        end = len(text)
    return text[start:end].strip()

uncertain_words = ['maybe', 'uncertain', 'cannot determine', 'not sure']
regex = ['A', 'B', 'C', 'D', 'Yes', 'No', 'Uncertain']
def check_ans_invalid(ans):
    if ans not in regex:
        return True
    else: 
        return False

def post_process_ans(ans, question):
    ans = ans.replace("True", "Yes").replace("False", "No")
    if check_ans_invalid(ans):
        first_word = ans.split()[0] if ' ' in ans else ans
        first_word = re.sub(r'[^a-zA-Z0-9]', '', first_word)
        if first_word in regex:
            ans = first_word
            return ans
    if check_ans_invalid(ans):
        for word in uncertain_words:
            if(word in ans.lower()):
                return 'Uncertain'
    if check_ans_invalid(ans):
        if 'true' in ans.lower():
            return 'Yes'
        elif 'false' in ans.lower():
            return 'No'
    if ans.strip() == '':
        if check_multiple_choice_ques(question):     
            options = ['A', 'B', 'C', 'D']
            return random.choice(options)
        if check_yes_no_question(question):
            options = ['Yes', 'No', 'Uncertain']
            return random.choice(options)   
    return ans


def reasoning_hard(model, logic_program, premise_fol, question, config):
    agent = ChatAgentReasoning(model, config)
    final_prompt = agent.make_prompt(
        lp_predicates_list=logic_program,
        lp_premises_list=premise_fol,
        question=question
    )

    # ic(final_prompt)
    reasoning_results = agent.inference_direct(
        prompt=final_prompt,
    )
    response = reasoning_results['text'].split("### Answer Response:")[-1].split("<</SYS>> [/INST]")[-1]
    markers = [
        "1. Selected Premises:",
        "2. Rationale:",
        "3. Final Answer:",
        "4. Explanation:"
    ]
    selected_premises = extract_sections(response, markers[0]).replace('P','').strip()
    selected_premises_final = [int(n) for n in re.findall(r'\d+', selected_premises)]
    answer = extract_sections(response, markers[2])
    answer = post_process_ans(answer, question)

    pattern = re.compile(r"2\.\s\*\*Rationale\*\*:(.*?)(?=3\.\s\*\*Final Answer\*\*:)", re.S)
    match = pattern.search(response)
    rationale = ""
    if match:
        rationale = match.group(1).strip()
    explanation = extract_sections(response, markers[3])
    final_explaination = rationale + '\n' + explanation

    ic(reasoning_results['text'].split("### Answer Response:")[-1])
    tokens = final_prompt.split(" ")
    ic(len(tokens))
    tokens.remove("")
    ic(len(tokens))
    return selected_premises_final, answer, final_explaination