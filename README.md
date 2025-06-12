<!-- for local mlflow debugging run: mlflow ui --backend-store-uri ./bert_finetuning/mlruns --port 5000 -->

This software was developed with the support of the Faculty of Information Technology, Czech Technical University in Prague, fit.cvut.cz

![FIT CTU Logo](fit_ctu_logo.jpg)  <!-- Adjust path and alt text as needed -->

# Abstract
This thesis addresses malicious URL detection using only the URL string, aiming to develop a model faster than the BERT-Small baseline while maintaining comparable predictive performance. Experiments were conducted on both publicly available datasets and a private dataset collected from a real computer network. A thorough analysis of the datasets and a detailed description of methods for malicious URL detection are provided prior to proposing the final solution. Two complementary approaches were combined to achieve the best results. The first one involves training smaller models, optimizing their hyper-parameters and proposing a new augmentation method -- domain masking, which prevents model from memorizing specific second level domain names and forces it to focus on general string features. To further improve inference speed, model compression techniques, such as static quantization, computation in Float16, were applied. The resulting BERT-Mini model with Float16 and domain masking surpassed the BERT-Small baseline in recall and achieved a 9.5x throughput improvement.

# Setup
- Run this in the root folder based on what machine you are running the code on.
- CPU machine: 
    - conda env create -f environment-cpu.yml
- GPU machine
    - conda env create -f environment-gpu.yml
- GPU machine for static quantization and quantized-aware trainning
    - TODO
- copy .env.example to .env
- create a data folder and copy datasets there to match following structure:
- data/processed/grambeddings/train.csv
- data/processed/grambeddings/test.csv
- if running non non-mac devices locally, change the the utils/environment_specific.py is_local_development function to return True

# Available scripts
- baseline_run_urlnet.ipynb - used for running URLNet for benchmarking performance
- data_exploration.ipynb - runs data exploration on the preprocessed datasets
- data_preprocessing.ipynb - does preprocessing on the raw datasets
- domain_masked_train.ipynb - trains BERT model using domain masking on whole train set
- feature_models.ipynb - runs feature based models on the datasets
- find_hypetperams_transformer.ipynb - this is used for hyperparameter tuning on validation set
- q_float16_and_dynamic.ipynb - does dynamic quantization, float16 quantization, optimization using onnx runtime
- q_static_qat_tensorrt_inference.ipynb - does static quantization, quantized aware training and runs them using tensorrt.
- run_model_from_file.ipynb - This script can be used to load model from file (useful for running attached models)
- train_transformer - Script for running transformer training on whole train set and evaluating on test set

# Other files
- baseline_code - code for URLNet baseline
- libraries - contains the adjusted transformers-deploy library part as mentioned in the thesis
