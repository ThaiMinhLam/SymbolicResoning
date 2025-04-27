import argparse
import yaml
from tqdm import tqdm
from icecream import ic

# Modules
from src.module import nl_to_fol, Extract_Logic_Progam
from src.dataloader_v2.dataset import XAIDataset, load_dataloader
from utils.utils import load_llm, load_yml, load_finetune_model

def get_args():
    parser = argparse.ArgumentParser(description="Load model config and run something")
    
    parser.add_argument('--file_path', type=str, required=True, help='Path to Reasoning Json File')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--device', type=int, required=True, default='cuda:0', help='Path to YAML config file')
    
    return parser.parse_args()


def main():
    args = get_args()
    config = load_yml(args.config)
    config['file_path'] = args.file_path
    ic(config.keys())

    # Load LLM
    llm_model, llama_base, tokenizer = load_llm(
        model_id=config['model_id'],
        config=config['model_config'],
        model_type=config['model_type'],
        device=args.device,
    )

    logicllama = load_finetune_model(
        model_base=llama_base,
        peft_path=config['module_nl2fol']['peft_path'],
        device=args.device
    ) 

    # Load dataset
    dataset = XAIDataset(config['data']['train'], config['data']['num_samples'])
    dataloader = load_dataloader(dataset, batch_size=config['data']['batch_size'], shuffle=True)

    # nl2fol
    module_1 = nl_to_fol(
        base_model=llama_base,
        finetune_model = logicllama,
        prompt_template_path=config['module_nl2fol']['prompt_template_path'],
        tokenizer=tokenizer,
        load_in_8bit = config['module_nl2fol']['load_in_8bit'],
        max_output_len = config['module_nl2fol']['max_output_len'],
    )


    # fol2fol

    extract_logic_program = Extract_Logic_Progam(
        base_model=logicllama,
        prompt_template_path=config['module_nl2fol']['prompt_template_path'],
        max_output_len=config['module_nl2fol']['max_output_len'],
        tokenizer=tokenizer,
        load_in_8bit=config['module_nl2fol']['load_in_8bit'],
    )



    for batch in tqdm(dataloader, desc="Processing batches"):
        premises = batch['premises-nl']
        fol_premises = batch['fol_premises']
        questions = batch['questions']
        reasonings = batch['reasonings']
        answers = batch['answers']

        # Solve tung sample 1 
        for i in range(len(premises)):
            data = {
                'premises-nl': premises[i],
                'fol_premises': fol_premises[i],
                'questions': questions[i],
                'reasonings': reasonings[i],
                'answers': answers[i],
                'LLM-FOL': None,
                'question-FOL': None

            }

            # Output using logicllama
            res_module_1 = module_1.generate_sample(data)

            # print(res_module_1['LLM-FOL'])
            # print(res_module_1['question-FOL'])
            
            res_modele_2 = extract_logic_program.generate_sample(res_module_1)
        
            print(res_modele_2)
            
            break
        







if __name__ == "__main__":
    main()
