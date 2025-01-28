import csv
import uuid
import pandas as pd
from typing import Dict


def csv_to_squad1_json(csv_file: str, title: str) -> Dict:
    """
    Convert CSV data to JSON format compatible with SQuAD v1 HuggingFace datasets.

    Args:
        csv_file: Path to input CSV file
        title: Document title for all entries

    Returns:
        Dictionary structured according to SQuAD v1 format for use with HuggingFace datasets

    Raises:
        ValueError: If required columns are missing
    """

    required_columns = {"context", "question", "answer", "answer_start"}
    squad_json_data = {"title": title, "data": []}

    with open(csv_file, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        missing_columns = required_columns - set(reader.fieldnames)

        # Validate Columns
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        for row in reader:
            if not row["answer"]:
                continue

            entry = {
                "title": title,
                "context": str(row["context"]),
                "question": str(row["question"]),
                "id": str(uuid.uuid4().hex),
                "answers": [
                    {
                        "answer_start": int(row["answer_start"]),
                        "text": str(row["answer"]),
                    }
                ],
            }
            squad_json_data["data"].append(entry)
        return squad_json_data


def csv_to_squad2_json(csv_file: str, title: str) -> Dict:
    """
    Convert CSV data to JSON format compatible with SQuAD v2 HuggingFace datasets.

    Args:
        csv_file: Path to input CSV file
        title: Document title for all entries

    Returns:
        Dictionary structured according to SQuAD v2 format for use with HuggingFace datasets
    """
    required_columns = {"context", "question", "answer", "answer_start"}
    qa_data = pd.read_csv(csv_file, na_values=[""])
    qa_data.fillna("", inplace=True)

    # Validate columns
    missing_columns = required_columns - set(qa_data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    squad_json_data = {"title": title, "data": []}

    for _, row in qa_data.iterrows():
        entry = {
            "context": row["context"],
            "question": row["question"],
            "id": uuid.uuid4().hex,
            "answers": [{"text": row["answer"], "answer_start": row["answer_start"]}]
            if row["answer"]
            else [],
            "is_impossible": str(not bool(row["answer"])).lower(),
        }
        squad_json_data["data"].append(entry)

    return squad_json_data
