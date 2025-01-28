# Pretraining Scripts
Collection of the pretraining scripts that were used in the creation of MechBERT models. `run_mlm.py` is a slightly modified version of the training script provided by HuggingFace. The BASH scripts were used to train these models on the Polaris Supercomputing cluster at ALCF.

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