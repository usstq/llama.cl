# Chat with Llama on x86 PC (w/o discrete GPU)

To support infinite chat exprience, model & pipeline was re-implemented using pytorch, with neccessary optimizations done in python + torch extension.

build and install chatllama

```bash
python -m pip install . -v
# or in editable/develop mode
python -m pip install -e .
```

Download models:

 - [Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2#%E5%AE%8C%E6%95%B4%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD)


Chat with the model
```bash
# export quantized 4bit pickle to saved_model.pkl (default pickle )
python -m chatllama -hf /path/to/hugging_face_model/ -q q4a --save
# run model (in Q&A mode)
python -m chatllama "What's oxygen?"
# run model (in chat mode with kv-cache of 512 tokens)
python -m chatllama --kv-len 512
# check internal kv-cache status
python -m chatllama -v
```
