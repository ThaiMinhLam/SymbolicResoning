import argparse
import yaml
import time
import numpy as np
import copy
import re
from tqdm import tqdm
from icecream import ic
from sentence_transformers import SentenceTransformer, util
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
    is_nearly_similar,
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
    parser.add_argument('--device', type=int, required=True, default=0, help='Path to YAML config file')
    
    return parser.parse_args()


def clustering_lp_similar_name(lps, threshold=0.5, cluster_by=0):
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
    list_idxs = [[is_nearly_similar(i, j, threshold=threshold, take_exactly_the_same=True)[1] for i in text] for j in text] # Get boolean variables
    list_scores = [[is_nearly_similar(i, j, threshold=threshold, take_exactly_the_same=True)[0] for i in text] for j in text] # Get boolean variables
    list_idxs = [[id for id, is_similar in enumerate(cluster_idxs) if is_similar] for cluster_idxs in list_idxs]
    print(list_idxs)
    all_used_idxs = []
    clusters = []
    idInCluster = {}
    for id, idxs in enumerate(list_idxs):
        unused_idxs = [i for i in idxs if i not in all_used_idxs]
        used_idxs = [i for i in idxs if i not in unused_idxs]
        select_lps = list(np.array(lps)[unused_idxs])
        all_used_idxs.extend(unused_idxs)
        if len(unused_idxs) == 1 and len(idxs) != 1: # Originally, idxs only have sigular value (no value higher than threshold except itself)
            max_cluster_id = np.argsort(list_scores[id])[-2]
            if max_cluster_id in used_idxs:
                cluster_id_of_max_score = idInCluster[max_cluster_id]
                clusters[cluster_id_of_max_score].append(np.array(lps)[id])
                continue

        # select_lps = list(np.array(lps)[idxs])
        for idx in unused_idxs:
            idInCluster[idx] = len(clusters)
        clusters.append(select_lps)
    clusters = [cluster for cluster in clusters if cluster != []]
    return clusters

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
        # if id in used_idxs:
        #     continue
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
    ic(clusters)
    clusters_with_no_entity = [[lp for lp in cluster if check_exist_entity_in_args(lp)==False] for cluster in clusters]
    clusters_with_entity = [[lp for lp in cluster if check_exist_entity_in_args(lp)==True] for cluster in clusters]
        #---Map only the distinct lp name---%
    all_map_entity = []
    ic(clusters_with_no_entity)
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


# Extract Predicate
def extract_predicates(sample: dict, fact_indices: list[int]) -> tuple[list, list]:        
    questions_fol = sample.get('new_premise_fol', [])
    premises_fol = sample.get('new_premise_fol', [])
    if not questions_fol or not premises_fol:
        print('Bug!!!!')
        return None
    
    premises_fol = [premises_fol[index] for index in fact_indices]

    predicates_question = set()
    predicates_premises = set()
    
    for ques in questions_fol:
        pred_matches = re.findall(r'([a-zA-Z0-9_]+)\(([^)]+)\)', ques)
        for pred_name, args in pred_matches:
            predicate = f"{pred_name}({args})"
            predicates_question.add(predicate)
    
    for premise in premises_fol:
        pred_matches = re.findall(r'([a-zA-Z0-9_]+)\(([^)]+)\)', premise)
        for pred_name, args in pred_matches:
            predicate = f"{pred_name}({args})"
            predicates_premises.add(predicate)

    return list(predicates_question), list(predicates_premises)

# Trích xuất các predicate in A in (A -> B)
def extract_predicate_implication(premise_fol: str) -> list:
    return None
    # predicates_implication = set()

def count_numbers_in_premise(premise: str):
    return len(re.findall(r'\d+', premise))

def reasoning_gate(premises: list,  num_numbers_threshold: int, num_premises_threshold: int):
    '''
    premises: list of premise
    num_numbers_threshold: the threshold of the number of number existing in each premise
    num_premises_threshold: the threshold of the number of premises that satisfy the condition
    '''
    total_premises = len(premises)
    if total_premises == 0:
        return False
    premises_count = 0
    for premise in premises:
        num_numbers = count_numbers_in_premise(premise)
        if num_numbers > num_numbers_threshold:
            premises_count += 1
    if premises_count > num_premises_threshold:
        return True # Trung reasoning
    else:
        return False # Toan reasoning


def main():
    ic("Load config")
    args = get_args()
    config = load_yml(args.config)
    # config['file_path'] = args.file_path

    ic(config.keys())

    # Load LLM and Required Model
    ic("Load llama base model")
    print("Load llama base model")
    llm_llama_model, llama_base, tokenizer_llama, llama_pipeline = load_llm(
        model_id=config['model_llama_id'],
        config=config['model_config'],
        model_type='llama',
        device=args.device,
    )

    ic("Load mistral base model")
    print("Load mistral base model")
    llm_mistral_model, mistral_base, tokenizer_mistral, mistral_pipeline = load_llm(
        model_id=config['model_mistral_id'],
        config=config['model_config'],
        model_type='mistral',
        device=args.device,
    )

    
    ic("logic llama model")
    print("logic llama model")
    logicllama = load_finetune_model(
        model_base=llama_base,
        peft_path=config['module_nl2fol']['peft_path'],
        device=args.device
    ) 

    print("Load embedding model")
    model_embedding = SentenceTransformer(config['model_embedding']).cuda(args.device)

    # Load dataset
    dataset = XAIDataset(config['data']['train'], config['data']['num_samples'])
    dataloader = load_dataloader(dataset, batch_size=config['data']['batch_size'], shuffle=False)

    # -----------------------------------------MODULES-------------------------------------
    # 0. Create Hypothesis
    ic("extract_hypothesis_another")
    extract_hypothesis_another = Extract_Hypothesis(
        base_model=logicllama,
        prompt_template_path=config['module_nl2fol']['prompt_template_path'],
        max_output_len=config['module_nl2fol']['max_output_len'],
        tokenizer=tokenizer_llama,
        load_in_8bit=config['module_nl2fol']['load_in_8bit']
    )
    
    # Module Preprocessing Natural Language Premises
    rewrite_premises = Preprocessing_PremiseNL(
        base_model = logicllama, # Thuê instance mới thì thay = Mistral
        prompt_template_path = config['module_nl2fol']['prompt_template_path'],
        max_output_len = config['module_nl2fol']['max_output_len'],
        tokenizer = tokenizer_llama, # Thuê instance mới thì thay = Mistral
        load_in_8bit = config['module_nl2fol']['load_in_8bit']
    )
    
    # 1. NL2FOL
    module_1 = nl_to_fol(
        base_model=llama_base,
        finetune_model = logicllama,
        prompt_template_path=config['module_nl2fol']['prompt_template_path'],
        tokenizer=tokenizer_llama,
        load_in_8bit = config['module_nl2fol']['load_in_8bit'],
        max_output_len = config['module_nl2fol']['max_output_len'],
        device=args.device,
    )

    # 2. FOL2FOL
    extract_logic_program = Extract_Logic_Progam(
        base_model=mistral_base,
        prompt_template_path=config['module_nl2fol']['prompt_template_path'],
        max_output_len=config['module_nl2fol']['max_output_len'],
        tokenizer=tokenizer_mistral,
    )

    # 3. SOLVER
    prover9 = Prover9_K(solver = FOL_Prover9_Program)
    

    for step, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        if step != 0:
            continue
        # if step != 3:
        #     continue
        premises = batch['premises-nl']
        fol_premises = batch['fol_premises']
        questions = batch['questions']
        reasonings = batch['reasonings']
        answers = batch['answers']
        idxs = batch['idxs']

        # Solve tung sample 1 
        for i in range(len(premises)):
            data = {
                'idx': idxs[i],
                'premises-nl': premises[i],
                'fol_premises': fol_premises[i],
                'questions': questions[i],
                'reasonings': reasonings[i],
                'ground_truth_answer': answers[i],
                'LLM-FOL': None,
                'question-FOL': None,
                'logic_program': None,
                'new_fol': None, 
                'maps': None,
                'q_ids': None # Each sub question belongs to which original question
            }

# -----------------------------------------EXECUTE---------------------------------------------
            def check_multiple_choice(question: str):
                if re.findall(r"\n[A-D][\.\)]? (.*?)(?=\n[A-D][\.\)]? |\Z)", question):
                    return True
                return False

            new_questions = []

            for question in data["questions"]:
                if check_multiple_choice(question):
                    new_question = make_conclusion(
                        model=llm_llama_model,
                        question=question,
                        config=config
                    )
                    new_questions.append(new_question)
                else: # Cho các câu hỏi loại khác
                    new_question = extract_hypothesis_another.generate_hypothesis(question)
                    new_questions.append(new_question)
            data['old_questions'] = data['questions']
            data['questions'] = new_questions
            
            # Preprocessing premises-nl
            # data['premises-nl'] = remove_intro_phrase_if_common(data['premises-nl'])
            data['premises-nl'], fact_indices = rewrite_premises.create_new_premises(data['premises-nl'])
            
            # 1. NL2FOL
                #---Preprocessing nl--#
            data['premises-nl'] = [clean_nl(nl) for nl in data['premises-nl']]
            res_module_1 = module_1.generate_sample(data)
            # save_json(
            #     {"res":res_module_1}, 
            #     f"/data/npl/ViInfographicCaps/Contest/final_contest/final_code/save/test/res_{step}.json"
            # )
            res_module_1['questions'] = [clean_nl(nl, check_multiple_choice(nl)) for nl in res_module_1['questions']]
                #---Preprocessing fol--#
            res_module_1['question-FOL'] = [clean_fol(fol, check_multiple_choice(fol)) for fol in res_module_1['question-FOL']]
            
            # 2. EXTRACT LOGIC PROGRAM - Preprocessing
            res_module_1['LLM-FOL'] = [lemmatize_word_fol(fol) for fol in res_module_1['LLM-FOL']]
            res_module_1['question-FOL'] = [lemmatize_word_fol(fol) for fol in res_module_1['question-FOL']]

            print(res_module_1['LLM-FOL'])
            print("----")
            print(res_module_1['question-FOL'])

            # raise
            # LOGIC PROGRAM - Premises
            logic_program_premise = extract_logic_program.generate_sample(res_module_1, mode='premise')
            logic_program_premise = [format_sep(lp) for lp in logic_program_premise] # Format sep ":::"
            # logic_program_premise = handle_missing_predicates_with_same_name(logic_program_premise, res_module_1['LLM-FOL'])
            res_module_1['logic_program_premise'] = logic_program_premise
            ic(logic_program_premise)
            
            # LOGIC PROGRAM - Question
            logic_program_question = extract_logic_program.generate_sample(res_module_1, mode='question')
            logic_program_question = [format_sep(lp) for lp in logic_program_question] # Format sep ":::"
            # logic_program_question = handle_missing_predicates_with_same_name(logic_program_question, res_module_1['question-FOL'])
            res_module_1['logic_program_question'] = logic_program_question
            ic(logic_program_question)
            #--- Combine
            lp_predicate_premises = [lp.split(':::')[0].strip() for lp in logic_program_premise]
            logic_program = logic_program_premise + [lp for lp in logic_program_question if lp.split(':::')[0].strip() not in lp_predicate_premises]
            logic_program = filter_duplicate_predicate(logic_program)
            res_module_1['logic_program'] = logic_program
            ic(logic_program)



            # 3. MAPPING - With is_nearly_similar() function
            clusters_name_similarity = clustering_lp_similar_name(logic_program, 0.7)
            # clusters_name_similarity = [[lp.item() for lp in cluster] for cluster in clusters_name_similarity]
            old_clusters = clusters_name_similarity
            for cluster in old_clusters:
                res_module_1["LLM-FOL"] = convert_common_fol(cluster, res_module_1["LLM-FOL"])
                res_module_1["question-FOL"] = convert_common_fol(cluster, res_module_1["question-FOL"])
            common_convert_dic = [create_common_map_dict(cluster) for cluster in clusters_name_similarity]
            clusters_name_similarity = [convert_common_lp(cluster) for cluster in clusters_name_similarity]
            clusters_name_similarity = [filter_duplicate_predicate(cluster) for cluster in clusters_name_similarity]
            logic_program_name_similarity = [lp for cluster in clusters_name_similarity for lp in cluster]
            # ic(clusters_name_similarity)
            # ic(logic_program_name_similarity)
            res_module_1['clusters_name_similarity_before'] = clusters_name_similarity
            res_module_1["LLM-FOL"], res_module_1["question-FOL"], logic_program_name_similarity, clusters_name_similarity, _ = mapping_entity_func(
                model=llm_mistral_model,
                logic_program=logic_program_name_similarity,
                clusters=clusters_name_similarity,
                premise_fol=res_module_1["LLM-FOL"],
                question_fol=res_module_1["question-FOL"],
                config=config,
            )
            res_module_1['old_clusters'] = old_clusters
            res_module_1['clusters_name_similarity'] = clusters_name_similarity
            res_module_1['logic_program_name_similarity'] = logic_program_name_similarity
            

            # 3. MAPPING - With model embedding
            clusters = clustering_lp(model_embedding, logic_program_name_similarity, 0.5)
            # ic(clusters)

            # 3. MAPPING - Converting entity to arguments
            res_module_1['common_convert_dic'] = common_convert_dic
            use_premise_fol = res_module_1["LLM-FOL"]
            use_question_fol = res_module_1["question-FOL"]
            use_logic_program = logic_program_name_similarity
            use_clusters = clusters_name_similarity
            mapping_info = []

            # save_json(
            #     {
            #         "old_clusters": old_clusters, 
            #         "clusters_name_similarity": clusters_name_similarity, 
            #         "clusters": clusters, 
            #         "use_premise_fol": use_premise_fol, 
            #         "use_question_fol": use_question_fol, 
            #         "use_logic_program": use_logic_program, 
            #         "use_cluster": use_clusters,
            #         "mapping_info": mapping_info,
            #     }, 
            #     f"/data/npl/ViInfographicCaps/Contest/final_contest/final_code/save/check/info_{step}.json"
            # )

            # raise 
            for i in range(2):
                new_premise_fol, new_question_fol, new_logic_program, new_clusters, maps_entity = mapping_entity_func(
                    model=llm_mistral_model,
                    logic_program=use_logic_program,
                    clusters=use_clusters,
                    premise_fol=use_premise_fol,
                    question_fol=use_question_fol,
                    config=config,
                )
                use_premise_fol = new_premise_fol
                use_question_fol = new_question_fol
                use_logic_program = new_logic_program
                use_clusters = new_clusters
                mapping_info.append(maps_entity)

            res_module_1['new_premise_fol'] = use_premise_fol 
            res_module_1['new_question_fol'] = use_question_fol 
            res_module_1['new_logic_program'] = use_logic_program 
            res_module_1['new_clusters'] = use_clusters 
            res_module_1['old_clusters'] = old_clusters 
            res_module_1['mapping_info'] = mapping_info 

            # save_json(
            #     {
            #         "old_clusters": old_clusters, 
            #         "clusters": clusters, 
            #         "new_premise_fol": new_premise_fol, 
            #         "new_question_fol": new_question_fol, 
            #         "new_logic_program": new_logic_program, 
            #         "new_cluster": new_clusters,
            #         "mapping_info": mapping_info,
            #     }, 
            #     f"/data/npl/ViInfographicCaps/Contest/final_contest/final_code/save/modify_save/info_{step}.json"
            # )
            
            # Loại bỏ predicates dư thừa
            predicates_ques, predicates_fact = extract_predicates(res_module_1, fact_indices)
            ic(predicates_ques)
            ic(predicates_fact)


            # 4. SOLVER
            res_module_1['solver'] = []
            list_premises_fol = res_module_1['new_premise_fol']
            list_questions_fol = res_module_1['new_question_fol']
            solver_info = {
                "answers": [],
                "idxs": [],
                "explanations": [],
            }
            # ic(data['questions'])
            for i, (question_fol, question) in enumerate(zip(list_questions_fol, data['old_questions'])):
                fol = [question_fol]
                result_solver = prover9.solving_questions(list_premises_fol, fol)
                answer = result_solver['final_ans']
                used_idx = result_solver['idx_final_ans']
                explanation = ""

                idx, answer, explanation = reasoning_hard(
                        model=llm_mistral_model, 
                        logic_program=new_logic_program, 
                        premise_fol=list_premises_fol, 
                        question=question, 
                        config=config
                    )

                # num_numbers_threshold =  0 
                # num_premises_threshold = float(len(list_premises_fol)/2)
                # if reasoning_gate(list_premises_fol, num_numbers_threshold, num_premises_threshold):
                #     # 4. REASONING TRUNG
                #     idx, answer, explanation = reasoning_hard(
                #         model=llm_mistral_model, 
                #         logic_program=new_logic_program, 
                #         premise_fol=list_premises_fol, 
                #         question=question, 
                #         config=config
                #     )
                # else:
                #     # 4. REASONING TOAN
                #     if used_idx==[]:
                #         ic("Create prompt")
                #         prompt=create_template_reasoning_easy_sample_v2(
                #             res_module_1['premises-nl'], # premises dạng NL
                #             res_module_1['new_premise_fol'],  # premises dạng FOL 
                #             question,  # câu hỏi gốc, chuẩn format có \nA \nB
                #             model_embedding, # mô hình embedding
                #         )

                #         ic("Generate Output")
                #         output_mistral = generate_explain(mistral_pipeline, prompt)
                #         answer, idx, explanation = extract_llm_output(output_mistral)
                #         ic('idx: ',idx)
                #         ic('answer: ',answer)
                #         ic('question: ', question)
                #         ic('explanation: ', explanation)
                #     else:
                #         prompt = create_template_explain(
                #             res_module_1['logic_program'], 
                #             res_module_1['premises-nl'],
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

            res_module_1['solver_info'] = solver_info
            path = f"/workspace/SymbolicResoning/XAI/save/all_files/log_{step}.json"
            save_json(res_module_1, path)
        


if __name__ == "__main__":
    begin = time.time()
    main()
    end = time.time()
    time_execute = end - begin
    ic(time_execute)
