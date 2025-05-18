/home/npl/.conda/envs/diffsinger/bin/python /data/npl/ViInfographicCaps/Contest/final_contest/final_code/main.py --config /data/npl/ViInfographicCaps/Contest/final_contest/final_code/config/config_llama2.yml \
    --file_path hihi \
    --device 0

python /workspace/XAI/main_dataloader.py --config /workspace/XAI/config/config_llama2.yml \
    --file_path hihi \
    --device 0

python /workspace/SymbolicResoning/XAI/main_modify.py --config //workspace/SymbolicResoning/XAI/config/config_model.yml \
    --file_path hihi \
    --device 0

python /workspace/SymbolicResoning/XAI/main_new_version.py --config //workspace/SymbolicResoning/XAI/config/config_model.yml \
    --file_path hihi \
    --device 0


python /workspace/SymbolicResoning/XAI/main_test_full_pipeline.py --config //workspace/SymbolicResoning/XAI/config/config_model.yml \
    --file_path hihi \
    --device 0