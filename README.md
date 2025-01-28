# MechBERT: Language Models for Extracting Chemical and Property Relationships about Mechanical Stress and Strain

This repo contains the codebase used to train domain-specific language models for information extraction. 

- `eval-scripts` - scripts to process a domain-specific question answering dataset and evaluate fine-tuned models on this task
- `pretraining-scripts` - BASH and Python scripts to perform tokenization and pretraining using HuggingFace on a HPC cluster with the PBS scheduler
- `finetuning-scripts` - Python and BASH scripts to fine-tune the models for question answering including a parameter sweep to find the optimal configuration

See the [associated paper](https://doi.org/10.1021/acs.jcim.4c00857) and models on [Hugging Face](https://huggingface.co/collections/CambridgeMolecularEngineering/mechbert-models-finetuned-on-squad-6798c159d114188bfe65fb96) for more details.


## Citation:
Please use the following the citation if you use any of this codebase in your work.
```
@article{mechbert-kumar2025,
  title={MechBERT: Language Models for Extracting Chemical and Property Relationships about Mechanical Stress and Strain},
  author={Pankaj Kumar, Saurabh Kabra, Jacqueline M. Cole},
  journal={Journal of Chemical Information and Modelling},
  doi={10.1021/acs.jcim.4c00857},
  year={2025}
}