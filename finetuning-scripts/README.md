# Finetuning Scripts
Python and BASH scripts used for fine-tuning the models for question answering. The sweep makes use of the Weights&Biases framework to perform a search for optimal hyperparameters. The BASH scripts distribute the fine-tuning onto individual nodes within a cluster and are designed specifically for the ALCF Polaris computing cluster which makes use of the PBS scheduler.

See the [associated paper](https://doi.org/10.1021/acs.jcim.4c00857) and models on [Hugging Face](https://huggingface.co/collections/CambridgeMolecularEngineering/mechbert-models-finetuned-on-squad-6798c159d114188bfe65fb96) for more details.


## Citation:
```
@article{mechbert-kumar2025,
  title={MechBERT: Language Models for Extracting Chemical and Property Relationships about Mechanical Stress and Strain},
  author={Pankaj Kumar, Saurabh Kabra, Jacqueline M. Cole},
  journal={Journal of Chemical Information and Modelling},
  doi={10.1021/acs.jcim.4c00857},
  year={2025}
}