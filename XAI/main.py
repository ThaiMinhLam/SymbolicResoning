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
    Preprocessing_PremiseNL,
    nl_to_fol_new,
    predicate_nl_extractor,
    start_corenlp_server,
    stop_corenlp_server
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

yes_no_patterns = r"^(Do|Does|Did|Is|Are|Was|Were|Have|Has|Had|Can|Could|Shall|Should|Will|Would|May|Might|Must)\b.*\?$"

def check_yes_no_question(question: str) -> bool:
    question = question.strip()
    if "following statement true" in question.lower():
        return True
    if re.match(yes_no_patterns, question, re.IGNORECASE):
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
        self.predicate_extractor = predicate_nl_extractor(self.mistral_pipeline, self.model_embedding, threshold=0.5)
        self.nl_to_fol_converter = nl_to_fol_new(self.mistral_pipeline)
        
        
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
        new_body = {
            'premises-nl': None,
            'questions': None,
            'old_questions': [question],
        }
        
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
        new_body['premises-nl'] = new_premises_NL
        new_body['questions'] = [new_question]

        # NL -> FOL
        start_corenlp_server(8000)
        premise_pred_dic, preds, subject_pred_dic, subjects = self.predicate_extractor.extract(new_body['premises-nl'], new_body['questions'])
        fol_pred_dic = self.nl_to_fol_converter.convert(preds)
        premise_nl_list = list(premise_pred_dic.keys())
        fol_formula = self.nl_to_fol_converter.convert_premise_to_fol(premise_nl_list, premise_pred_dic, fol_pred_dic, subject_pred_dic)
        print(fol_formula)
        raise
        
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
            # 4. REASONING TRUNG
            idx, answer, explanation = reasoning_hard(
                model=self.llm_mistral_model, 
                logic_program=new_logic_program, 
                premise_fol=list_premises_fol, 
                question=_question,
                config=self.config
            )
                    
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
        "Ebbinghaus' forgetting curve formula: R = e^(-t/S), where R is retention rate, t is elapsed time, and S is review interval.",
        "A learning algorithm based on spaced repetition can adjust review intervals based on individual proficiency.",
        "Adequate sleep enhances memory consolidation after each review session.",
        "Creating flashcards with concise questions improves retention compared to passive reading.",
        "Reviewing just before forgetting significantly boosts memory efficiency.",
        "Neuroscience studies show self-testing activates the hippocampus, enhancing information recall.",
        "Encountering knowledge in various contexts improves retention compared to monotonous repetition.",
        "Too short review intervals reduce retention due to lack of time for consolidation.",
        "Too long review intervals risk forgetting most of the material before review.",
        "AI can personalize study schedules, optimizing memory retention for each student based on their progress."
    ]
    question = "Which of these conclusions is supported by the forgetting curve research?\nA. Review timing has no impact on retention\nB. Optimal intervals prevent both premature review and excessive forgetting\nC. Sleep has no effect on memory consolidation\nD. All learning methods yield identical results"

    response = model.response(premises, question)

if __name__ == "__main__":
    begin = time.time()
    main()
    end = time.time()
    time_execute = end - begin
    ic(time_execute)
