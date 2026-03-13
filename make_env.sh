conda install -c nvidia cuda-toolkit=12.4 cuda-runtime=12.4 -y
pip install -r requirements.txt
pip install vllm==0.16.0
pip install llmcompressor==0.10.0
pip install compressed_tensors==0.14.0 # Override vllm dependency
pip uninstall -y tensorflow
