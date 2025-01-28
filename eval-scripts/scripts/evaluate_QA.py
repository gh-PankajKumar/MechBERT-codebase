from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import json
import logging
import torch

from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from datasets import load_dataset
from evaluate import evaluator


class QAModelEvaluator:
    """Question Answering model evaluator for SQuAD-formatted datasets.

    Attributes:
        model_name: Name or path of the pretrained model
        dataset: Loaded evaluation dataset
        pipeline: QA inference pipeline
        results: Dictionary to store evaluation results
    """

    def __init__(self, model_name: str, data_path: str, squad_v2: bool = True):
        """Initialize QA evaluator with model and dataset.

        Args:
            model_name: Pretrained model name or path
            data_path: Path to evaluation dataset (JSON)
            squad_v2: Whether to use SQuAD v2 evaluation format

        Raises:
            FileNotFoundError: If model or data file not found
        """
        self.model_name = model_name
        self.squad_v2 = squad_v2
        self.results = {}

        # Validate paths
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        try:
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {str(e)}")

        self._load_dataset(data_path)
        self._create_pipeline()

    def _load_dataset(self, data_path: str) -> None:
        self.dataset = load_dataset(
            "json", data_files=data_path, split="train", field="data"
        )

    def _create_pipeline(self) -> None:
        self.pipeline = pipeline(
            task="question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )

    def evaluate_QA(self, output_dir: str = "./results") -> Dict[str, Any]:
        """Run evaluation and save results.

        Args:
            output_dir: Directory to save evaluation results

        Returns:
            Dictionary containing evaluation metrics
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        eval_task = evaluator("question-answering")

        self.results = eval_task.compute(
            model_or_pipeline=self.pipeline,
            data=self.dataset,
            metric="squad_v2" if self.squad_v2 else "squad",
            squad_v2_format=self.squad_v2,
        )

        self._save_results(output_dir)
        return self.results

    def _save_results(self, output_dir: str) -> None:
        """Save evaluation results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{output_dir}/{self.model_name.split('/')[-1]}_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(
                {
                    "model": self.model_name,
                    "timestamp": timestamp,
                    "metrics": self.results,
                },
                f,
                indent=2,
            )

        logging.info(f"Results saved to {filename}")
