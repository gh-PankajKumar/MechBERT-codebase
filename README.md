# Evaluation Scripts
The basic evaluation scripts that include:
1. `process_data.py` - functions to convert csv question answer datasets into formats suitable for SQuAD v1 and v2 HuggingFace pipelines.
2. `evaluate_QA.py` - uses the evaluate library from HuggingFace to generate scores of question answering models on SQuAD-like datasets. 

These scripts are used to process a domain-specific question answering dataset which is used to test various fine-tuned models on this task.

See the [associated paper](https://doi.org/10.1021/acs.jcim.4c00857) and models on [Hugging Face](https://huggingface.co/collections/CambridgeMolecularEngineering/mechbert-models-finetuned-on-squad-6798c159d114188bfe65fb96) for more details.


## `process_data.py` Example Usage:

```python
import json

input_csv = "domain_specific_qa.csv"
document_title = "StressEngineeringQA"

# Convert to SQuAD 1.1 format
squad1_data = csv_to_squad1_json(input_csv, document_title)
# Save to json file:
with open("Eval_QA_SQuAD1.json", "w") as squad1_file:
    json.dump(squad1_data, squad1_file, indent=4)

# Convert to SQuAD 2.0 format
squad2_data = csv_to_squad2_json(input_csv, document_title)
# Save to json file:
with open("Eval_QA_SQuAD2.json", "w") as squad2_file:
    json.dump(squad2_data, squad2_file, indent=4)


```

## `evaluate_QA.py` Example Usage:
Evaluating models fine-tuned on SQuAD v1-like data:
```python
MODELS_TO_EVALUATE = ["PATH_TO_MODELS/MechBERT-cased-squad", "PATH_TO_MODELS/MechBERT-uncased-squad"]
DATASET_PATH = "PATH_TO_DATASET/Eval_QA_SQuAD1.json"
RESULTS_DIR = "SAVE_DIR/evaluation_results"
for model_path in MODELS_TO_EVALUATE:
    # SQuAD v1 Eval
    QA_eval = QAModelEvaluator(model_name=model_path, data_path=DATASET_PATH, squad_v2=False)
    results = QA_eval.evaluate_QA(output_dir=RESULTS_DIR)
```

Evaluating models fine-tuned on SQuAD v2-like data:
```python
MODELS_TO_EVALUATE = ["PATH_TO_MODELS/MechBERT-cased-squad2", "PATH_TO_MODELS/MechBERT-uncased-squad2"]
DATASET_PATH = "PATH_TO_DATASET/Eval_QA_SQuAD2.json"
RESULTS_DIR = "SAVE_DIR/evaluation_results"
for model_path in MODELS_TO_EVALUATE:
    # SQuAD v2 Eval
    QA_eval = QAModelEvaluator(model_name=model_path, data_path=DATASET_PATH, squad_v2=True)
    results = QA_eval.evaluate_QA(output_dir=RESULTS_DIR)
```

## Download:

### Clone repo:

```BASH
git clone https://github.com/gh-PankajKumar/MechBERT-eval-scripts
cd MechBERT-eval-scripts
```

### Use uv to install requirements:
```BASH
# Install uv if not already installed
curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh | sh

uv sync
```

## Citation:
```
@article{mechbert-kumar2025,
  title={MechBERT: Language Models for Extracting Chemical and Property Relationships about Mechanical Stress and Strain},
  author={Pankaj Kumar, Saurabh Kabra, Jacqueline M. Cole},
  journal={Journal of Chemical Information and Modelling},
  doi={10.1021/acs.jcim.4c00857},
  year={2025}
}