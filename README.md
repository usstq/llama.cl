# Chat with Llama on x86 PC (w/o discrete GPU)

To support infinite chat exprience, model & pipeline was re-implemented using pytorch, with neccessary optimizations done in python + torch extension.

install requirements:

```bash
pip install -r requirements.txt
```

## build using Intel Compiler

> ref: https://www.intel.com/content/www/us/en/developer/articles/guide/porting-guide-for-icc-users-to-dpcpp-or-icx.html

```bat
:: linux
source /opt/intel/oneapi/setvars.sh
CC=icx CXX=icpx cmake ..

:: windows: using cmd.exe instead of PS, (Ninja is required for Intel Compiler to work correctly)
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 vs2022
cmake -G Ninja -DPYTHON_EXECUTABLE="C:/Users/tingqian/AppData/Local/Programs/Python/Python38/python38.exe" ..

:: clean
cmake.exe --build . --config Release --target clean
:: build
cmake.exe --build . --config Release --verbose

:: check DLL dependency on windows
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x86\dumpbin.exe" /DEPENDENTS ./llmops.cp38-win_amd64.pyd
:: Intel compiler usually introduces additional dependencies on :
::    - libmmd.dll/libiomp5md.dll   oneAPI/compiler/latest/windows/redist/intel64_win/compiler
::    - sycl6.dll                   oneAPI/compiler/2023.2.1/windows/bin

:: analyze performance with viztracer (view result.json inside chrome://tracing/)
numactl -C0,2,4,6,8,10,12,14 python -m viztracer ./chatllama/main.py Hello
```

## Download models:

 - [Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2#%E5%AE%8C%E6%95%B4%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD)


Chat with the model
```bash
# export quantized 4bit pickle to saved_model.pkl (default pickle )
python ./chatllama/main.py -hf /path/to/hugging_face_model/ --save --kv-len 2048
```

## PPL test

`python ./chatllama/main.py --ppl ./wikitext-2/wiki.test.raw`

| Model    | Measure          | F32   |  Q4_1(32)  |  Q4_1(128)  | 
| -------- | -------          |-------|------------|-------------|
| Llama-7B | fc weight size   | 26G   |  4.0G      |  3.4G       | 
|          | ms/tok @ 8 Pcore | 383   |   77       |   67        | 
|          |  perplexity      | 7.49  |  7.92      |  8.20       | 
