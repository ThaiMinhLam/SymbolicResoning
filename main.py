import argparse
import yaml
import time
import numpy as np
import copy
import re
from tqdm import tqdm
from icecream import ic
import ast
from sentence_transformers import SentenceTransformer, util
import sys
sys.path.append("/workspace/XAI")
# Open Packages
import spacy
from collections import Counter
nlp = spacy.load("en_core_web_sm")

# Modules
from src.module import (
    nl_to_fol,
    Extract_Logic_Progam,
    reducing,
    Prover9_K,
    FOL_Prover9_Program,
    make_conclusion,
    Extract_Hypothesis,
    convert_entity,
    Preprocessing_PremiseNL
)
from src.module.reasoning import reasoning_hard, create_template_explain, generate_explain, create_template_reasoning_easy_sample_v2,extract_llm_output
from src.dataloader_v2.dataset import XAIDataset, load_dataloader
from utils import (
    load_llm,
    load_yml,
    load_finetune_model,
    save_json,
    map_to_fol,
    handle_missing_predicates_with_same_name,
    format_sep,
    clean_nl,
    clean_fol,
    check_exist_entity_in_args,
    preprocessing_entity_map,
    map_to_fol_entity,
    map_to_lp_entity,
    lemmatize_word_fol,
    filter_duplicate_predicate,
    create_common_map_dict,
    convert_common_lp,
    convert_common_fol,
)


def get_args():
    parser = argparse.ArgumentParser(description="Load model config and run something")
    
    parser.add_argument('--file_path', type=str, required=True, help='Path to Reasoning Json File')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--device', type=int, required=True, default='cuda:0', help='Path to YAML config file')
    
    return parser.parse_args()


def clustering_lp(model, lps, threshold=0.5, cluster_by=0):
    """
        Params
        ------

        cluster_by: 
            + 0 - Predicate_name
            + 1 - Definition
    """
    # Get the definition of predicate
    # predicates = [lp.split(":::")[0].strip() for lp in lps]
    text = [lp.split(":::")[cluster_by].strip() for lp in lps]
    if cluster_by==0:
        text = [t.split("(")[0].strip() for t in text]

    # Calculate score cos_sim similarity of each record to another
    embeddings = model.encode(text, convert_to_tensor=True)
    list_cosine_scores = util.cos_sim(embeddings, embeddings)
    list_cosine_scores = [scores.detach().cpu() for scores in list_cosine_scores]

    # Filter the score that has higher score than threshold
    list_idxs = [np.where(cosine_scores > threshold)[0] for cosine_scores in list_cosine_scores]
    used_idxs = []
    clusters = []
    for id, idxs in enumerate(list_idxs):
        if id in used_idxs:
            continue
        idxs = [i for i in idxs if i not in used_idxs]
        select_lps = list(np.array(lps)[idxs])
        clusters.append(select_lps)
        used_idxs.extend(idxs)
    return clusters
    

def remove_intro_phrase_if_common(sentences: list[str]):
    def extract_intro_phrase(sentence: str):
        doc = nlp(sentence)
        comma_idx = None
        for token in doc:
            if token.text == ',':
                comma_idx = token.i
                break
        if comma_idx is None:
            return None
        for token in doc[:comma_idx]:
            if token.dep_ in ['prep', 'advmod', 'npadvmod']:
                return sentence[:doc[comma_idx + 1].idx].strip()
        return None
    
    intro_phrases = [extract_intro_phrase(sen) for sen in sentences]
    phrase_counts = Counter(intro_phrases)
    most_common_phrase, freq = phrase_counts.most_common(1)[0]
    
    if most_common_phrase and freq == len(sentences):
        return [s.replace(most_common_phrase, "").lstrip().capitalize() for s in sentences]
    else:
        return sentences


def mapping_entity_func(model, logic_program, clusters, premise_fol, question_fol, config):
        #---Map only the lp with no entity in the argument---%
    clusters_with_no_entity = [[lp for lp in cluster if check_exist_entity_in_args(lp)==False] for cluster in clusters]
    clusters_with_entity = [[lp for lp in cluster if check_exist_entity_in_args(lp)==True] for cluster in clusters]
        #---Map only the distinct lp name---%
    all_map_entity = []
    for cluster in clusters_with_no_entity:
        if len(cluster) <= 1:
            continue
        map_entity = convert_entity(model, cluster, config)
        all_map_entity.append(map_entity)
    
    # Preprocessing mapping entity
    maps_entity = []
    for cluster_dict in all_map_entity:
        list_predicate_name_cluster = [k for k in cluster_dict.keys()]
        final_map = {k: preprocessing_entity_map(k, v, list_predicate_name_cluster) for k, v in cluster_dict.items()}
        maps_entity.append(final_map)

    maps_entity = {k: v for map in maps_entity for k, v in map.items()}
    new_premise_fol = map_to_fol_entity(maps_entity, premise_fol)
    new_question_fol = map_to_fol_entity(maps_entity, question_fol)
    new_logic_program = map_to_lp_entity(maps_entity, logic_program)
    new_clusters = [map_to_lp_entity(maps_entity, lp) for lp in clusters]

    # filter 
    new_logic_program = filter_duplicate_predicate(new_logic_program)
    new_clusters = [filter_duplicate_predicate(cluster) for cluster in new_clusters]
    return new_premise_fol, new_question_fol, new_logic_program, new_clusters, maps_entity



def check_multiple_choice(question: str):
    if re.findall(r"\n[A-D][\.\)]? (.*?)(?=\n[A-D][\.\)]? |\Z)", question):
        return True
    return False

# Loại bỏ predicates dư thừa
def remove_redundant_predicates(sample: dict, fact_indices: list[int]) -> list:
    questions_fol = sample.get('new_question_fol', [])
    premises_fol = sample.get('new_premise_fol', [])
    if not questions_fol or not premises_fol:
        print('Bug!!!!')
        return None
    
    facts_fol = [premises_fol[index] for index in fact_indices]

    predicates_question = set()
    predicates_premises = set()
    
    for ques in questions_fol:
        pred_matches = re.findall(r'([a-zA-Z0-9_]+)\(([^)]+)\)', ques)
        for pred_name, args in pred_matches:
            predicate = f"{pred_name}({args})"
            predicates_question.add(predicate)
    
    for fact in facts_fol:
        pred_matches = re.findall(r'([a-zA-Z0-9_]+)\(([^)]+)\)', fact)
        for pred_name, args in pred_matches:
            predicate = f"{pred_name}({args})"
            predicates_premises.add(predicate)
            
    predicates_question = list(predicates_question)
    predicates_premises = list(predicates_premises)
    predicates_total = predicates_question + predicates_premises
            
    implication_indices = [idx for idx in range(len(premises_fol)) if idx not in fact_indices]
    implications_fol = []
    
    for index in implication_indices:
        implication_fol = premises_fol[index]
        condition, consequence = implication_fol.split('→')
        pred_matches = re.findall(r'([a-zA-Z0-9_]+)\(([^)]+)\)', condition)
        for pred_name, args in pred_matches:
            temp = True
            predicate = f"{pred_name}({args})"
            if predicate not in predicates_total:
                temp = False
                break
        if temp:
            pred_matches = re.findall(r'([a-zA-Z0-9_]+)\(([^)]+)\)', consequence)
            for pred_name, args in pred_matches:
                predicate = f"{pred_name}({args})"
                predicates_total.append(predicate) 
                implications_fol.append(implication_fol)              

    return facts_fol + implications_fol


# -------------------------------- FastAPI -> app.py ---------------------------------
class LogicModel:
    def __init__(self, config_path='/workspace/SymbolicResoning/XAI/config/config_model.yml', device='cuda:0'):
        self.device = device
        self._init_config(config_path)
        self._init_llama_models()
        self._init_mistral_models()
        self.model_embedding = SentenceTransformer(self.config['model_embedding']).to(self.device)
        self._init_modules()
    
    def _init_config(self, config_path: str):
        self.config = load_yml(config_path)
    
    def _load_finetune_model(self):
        return load_finetune_model(
            model_base = self.llama_base,
            peft_path = self.config['module_nl2fol']['peft_path'],
            device = self.device
        ) 
    
    def _init_llama_models(self):
        self.llm_llama_model, self.llama_base, self.tokenizer_llama, self.llama_pipeline = load_llm(
            model_id = self.config['model_llama_id'],
            config = self.config['model_config'],
            model_type = self.config['model_llama_type'],
            device = self.device,
        )
        self.logicllama = self._load_finetune_model()
        
    def _init_mistral_models(self):
        self.llm_mistral_model, self.mistral_base, self.tokenizer_mistral, self.mistral_pipeline = load_llm(
            model_id = self.config['model_mistral_id'],
            config = self.config['model_config'],
            model_type = self.config['model_mistral_type'],
            device = self.device,
        )
    
    def _init_modules(self):
        # Modules Create Hypothesis for Questions
        self.extract_hypothesis_another = Extract_Hypothesis(
            base_model = self.mistral_base, # Thuê instance mới thì thay = Mistral
            prompt_template_path = self.config['module_nl2fol']['prompt_template_path'],
            max_output_len = self.config['module_nl2fol']['max_output_len'],
            tokenizer = self.tokenizer_mistral, # Thuê instance mới thì thay = Mistral
            load_in_8bit = self.config['module_nl2fol']['load_in_8bit']
        )
        
        # Module Preprocessing Natural Language Premises
        self.rewrite_premises = Preprocessing_PremiseNL(
            base_model = self.mistral_base, # Thuê instance mới thì thay = Mistral
            prompt_template_path = self.config['module_nl2fol']['prompt_template_path'],
            max_output_len = self.config['module_nl2fol']['max_output_len'],
            tokenizer = self.tokenizer_mistral, # Thuê instance mới thì thay = Mistral
            load_in_8bit = self.config['module_nl2fol']['load_in_8bit']
        )
        
        # Module Convert NL to FOL
        self.nl2fol = nl_to_fol(
            base_model = self.llama_base,
            finetune_model = self.logicllama,
            prompt_template_path = self.config['module_nl2fol']['prompt_template_path'],
            tokenizer = self.tokenizer_llama,
            load_in_8bit = self.config['module_nl2fol']['load_in_8bit'],
            max_output_len = self.config['module_nl2fol']['max_output_len'],
            device = self.device,
        )
        
        # Module Extract Logic Program
        self.extract_logic_program = Extract_Logic_Progam(
            base_model = self.mistral_base, # Thuê instance mới thì thay = Mistral
            prompt_template_path = self.config['module_nl2fol']['prompt_template_path'],
            max_output_len = self.config['module_nl2fol']['max_output_len'],
            tokenizer = self.tokenizer_mistral, # Thuê instance mới thì thay = Mistral
            load_in_8bit = self.config['module_nl2fol']['load_in_8bit'],
        )

        # Module Solver
        self.prover9 = Prover9_K(solver=FOL_Prover9_Program)
        
    def response(self, premises_NL: list[str], question: str):
        # Create Hypothesis For Question
        if check_multiple_choice(question):
            new_question = make_conclusion(
                model = self.llm_llama_model,
                question = question,
                config = self.config
            )
        else:
            new_question = self.extract_hypothesis_another.generate_hypothesis(question)
        
        # Preprocessing Premises-NL -> new Premises-NL + List index of fact
        new_premises_NL, fact_indices = self.rewrite_premises.create_new_premises(premises_NL)
        
        # NL -> FOL
        new_premises_NL = [clean_nl(premise) for premise in new_premises_NL]
        new_body = {
            'premises-nl': new_premises_NL,
            'questions': [new_question],
        }
        new_body['old_questions'] = [question]
    
        new_body = self.nl2fol.generate_sample(new_body)
        ic("Start preprocessing")
        # Post-Processing
        new_body['questions'] = [clean_nl(nl, check_multiple_choice(nl)) for nl in new_body['questions']]
        new_body['question-FOL'] = [clean_fol(fol, check_multiple_choice(fol)) for fol in new_body['question-FOL']]
        new_body['LLM-FOL'] = [clean_fol(premise_FOL) for premise_FOL in new_body['LLM-FOL']]
        ic(new_body)
        
        # 2. EXTRACT LOGIC PROGRAM - Preprocessing
        new_body['LLM-FOL'] = [lemmatize_word_fol(fol) for fol in new_body['LLM-FOL']]
        new_body['question-FOL'] = [lemmatize_word_fol(fol) for fol in new_body['question-FOL']]

        # LOGIC PROGRAM - Premises
        logic_program_premise = self.extract_logic_program.generate_sample(new_body, mode='premise')
        logic_program_premise = [format_sep(lp) for lp in logic_program_premise] # Format sep ":::"
        logic_program_premise = handle_missing_predicates_with_same_name(logic_program_premise, new_body['LLM-FOL'])
        new_body['logic_program_premise'] = logic_program_premise
        ic(logic_program_premise)
        
        # LOGIC PROGRAM - Question
        logic_program_question = self.extract_logic_program.generate_sample(new_body, mode='question')
        logic_program_question = [format_sep(lp) for lp in logic_program_question] # Format sep ":::"
        logic_program_question = handle_missing_predicates_with_same_name(logic_program_question, new_body['question-FOL'])
        new_body['logic_program_question'] = logic_program_question
        ic(logic_program_question)
        
        # Combine 
        lp_predicate_premises = [lp.split(':::')[0].strip() for lp in logic_program_premise]
        logic_program = logic_program_premise + [lp for lp in logic_program_question if lp.split(':::')[0].strip() not in lp_predicate_premises]
        new_body['logic_program'] = logic_program
        ic(new_body)
        
        # 3. MAPPING - Converting entity to arguments
        clusters = clustering_lp(self.model_embedding, logic_program, 0.5)
        clusters = [[lp.item() for lp in cluster] for cluster in clusters]
        old_clusters = clusters
        # clusters = [convert_common(cluster) for cluster in clusters]
        
        for cluster in old_clusters:
            new_body["LLM-FOL"] = convert_common_fol(cluster, new_body["LLM-FOL"])
            new_body["question-FOL"] = convert_common_fol(cluster, new_body["question-FOL"])
        
        common_convert_dic = [create_common_map_dict(cluster) for cluster in clusters]
        clusters = [convert_common_lp(cluster) for cluster in clusters]
        clusters = [filter_duplicate_predicate(cluster) for cluster in clusters]
        
        new_body['common_convert_dic'] = common_convert_dic
        use_premise_fol = new_body["LLM-FOL"]
        use_question_fol = new_body["question-FOL"]
        use_logic_programs = logic_program
        use_clusters = clusters
        mapping_info = []

        for i in range(2):
            new_premise_fol, new_question_fol, new_logic_program, new_clusters, maps_entity = mapping_entity_func(
                model=self.llm_llama_model,
                logic_program=use_logic_programs,
                clusters=use_clusters,
                premise_fol=use_premise_fol,
                question_fol=use_question_fol,
                config=self.config,
            )
            use_premise_fol = new_premise_fol
            use_question_fol = new_question_fol
            use_logic_programs = new_logic_program
            use_clusters = new_clusters
            mapping_info.append(maps_entity)

        new_body['new_premise_fol'] = use_premise_fol 
        new_body['new_question_fol'] = use_question_fol 
        new_body['new_logic_program'] = use_logic_programs 
        new_body['new_clusters'] = use_clusters 
        new_body['mapping_info'] = mapping_info
        
        # Loại bỏ các Premises-FOL không sử dụng suy diễn liên quan đến Fact + Question
        final_premise_fol = remove_redundant_predicates(new_body, fact_indices)
        new_body['new_premise_fol'] = final_premise_fol
        ic(final_premise_fol)
        
        # 4. SOLVER
        new_body['solver'] = []
        list_premises_fol = new_body['new_premise_fol']
        list_questions_fol = new_body['new_question_fol']
        solver_info = {
            "answers": [],
            "idxs": [],
            "explanations": [],
        }
        for i, (question_fol, _question) in enumerate(zip(list_questions_fol, new_body['old_questions'])):
            fol = [question_fol]
            result_solver = self.prover9.solving_questions(list_premises_fol, fol)
            answer = result_solver['final_ans']
            used_idx = result_solver['idx_final_ans']
            explanation = ""

            num_numbers_threshold =  0 
            num_premises_threshold = float(len(list_premises_fol)/2)
            # if reasoning_hard(list_premises_fol, num_numbers_threshold, num_premises_threshold):
            # 4. REASONING TRUNG
            idx, answer, explanation = reasoning_hard(
                model=self.llm_mistral_model, 
                logic_program=new_logic_program, 
                premise_fol=list_premises_fol, 
                question=_question, 
                config=self.config
            )
            # else:
            #     # 4. REASONING TOAN
            #     if not used_idx:
            #         ic("Create prompt")
            #         prompt = create_template_reasoning_easy_sample_v2(
            #             new_body['premises-nl'], # premises dạng NL
            #             new_body['new_premise_fol'],  # premises dạng FOL 
            #             _question,  # câu hỏi gốc, chuẩn format có \nA \nB
            #             self.model_embedding, # mô hình embedding
            #         )

            #         ic("Generate Output")
            #         output_mistral = generate_explain(mistral_pipeline, prompt)
            #         answer, idx, explanation = extract_llm_output(output_mistral)
            #         ic('idx: ', idx)
            #         ic('answer: ', answer)
            #         ic('question: ', question)
            #         ic('explanation: ', explanation)
            #     else:
            #         prompt = create_template_explain(
            #             new_body['logic_program'], 
            #             new_body['premises-nl'],
            #             list_premises_fol,
            #             used_idx,
            #             question,
            #             answer,
            #             False
            #         )
            #         explanation = generate_explain(mistral_pipeline, prompt)
                    
            solver_info['answers'].append(answer)
            solver_info['idxs'].append(idx)
            solver_info['explanations'].append(explanation)
        
        new_body['solver_info'] = solver_info
        
        answers = solver_info['answers'][0]
        idx = solver_info['idxs'][0]
        explanation = solver_info['explanations'][0]
        ic(answers)
        ic(idx)
        ic(explanation)
        
        return answers, idx, explanation

# -------------------------------- FastAPI -> app.py ---------------------------------
def main():
    model = LogicModel()
    premises = [
      "Students who have completed the core curriculum and passed the science assessment are qualified for advanced courses.",
      "Students who are qualified for advanced courses and have completed research methodology are eligible for the international program.",
      "Students who have passed the language proficiency exam are eligible for the international program.",
      "Students who are eligible for the international program and have completed a capstone project are awarded an honors diploma.",
      "Students who have been awarded an honors diploma and have completed community service qualify for the university scholarship.",
      "Students who have been awarded an honors diploma and have received a faculty recommendation qualify for the university scholarship.",
      "Sophia has completed the core curriculum.",
      "Sophia has passed the science assessment.",
      "Sophia has completed the research methodology course.",
      "Sophia has completed her capstone project.",
      "Sophia has completed the required community service hours."
    ]
    question = "Based on the above premises, which is the strongest conclusion?\nA. Sophia qualifies for the university scholarship\nB. Sophia needs a faculty recommendation to qualify for the scholarship\nC. Sophia is eligible for the international program\nD. Sophia needs to pass the language proficiency exam to get an honors diploma"
    response = model.response(premises, question)

if __name__ == "__main__":
    begin = time.time()
    # main()
    end = time.time()
    time_execute = end - begin
    ic(time_execute)
