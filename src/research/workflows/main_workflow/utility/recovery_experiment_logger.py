import os
import csv
import json
import time
import uuid
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

LOG_DIR = os.path.join(os.getcwd(), "experiments")
CSV_PATH = os.path.join(LOG_DIR, "recovery_experiments.csv")
JSONL_PATH = os.path.join(LOG_DIR, "recovery_experiments.jsonl")

CSV_HEADERS = [
    "Experiment ID",
    "Error / Failure Code",
    "Error Category",
    "Repair Level",
    "Recovery Method Attempted",
    "Success / Failure",
    "Recovery Time",
    "Recovery Cost",
    "Tokens Used"
]


class RecoveryExperimentLogger:
    """
    Handles persisting recovery attempt metrics into CSV and JSONL files
    and displaying visual terminal tables.
    """

    @staticmethod
    def ensure_log_files():
        os.makedirs(LOG_DIR, exist_ok=True)
        if not os.path.exists(CSV_PATH):
            with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)

    @staticmethod
    def calculate_cost(prompt_tokens: int, completion_tokens: int, model_name: str = "") -> float:
        """
        Estimates USD cost based on standard model token pricing.
        Defaults to approx ~$0.14 / 1M prompt, ~$0.28 / 1M completion.
        """
        prompt_rate = 0.14 / 1_000_000
        completion_rate = 0.28 / 1_000_000
        return round((prompt_tokens * prompt_rate) + (completion_tokens * completion_rate), 6)

    @classmethod
    def log_recovery_attempt(
        cls,
        experiment_id: str,
        error_code: str,
        error_category: str,
        repair_level: str,
        recovery_method: str,
        success: bool,
        recovery_time_seconds: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model_name: str = ""
    ) -> Dict[str, Any]:
        """
        Appends an experiment result to CSV and JSONL files and prints a Rich table summary.
        """
        cls.ensure_log_files()

        total_tokens = prompt_tokens + completion_tokens
        cost = cls.calculate_cost(prompt_tokens, completion_tokens, model_name)
        status_str = "Success" if success else "Failure"
        time_str = f"{recovery_time_seconds:.3f}s"
        cost_str = f"${cost:.6f}"
        tokens_dict = {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens
        }
        tokens_str = json.dumps(tokens_dict)

        record = {
            "Experiment ID": experiment_id,
            "Error / Failure Code": error_code,
            "Error Category": error_category,
            "Repair Level": repair_level,
            "Recovery Method Attempted": recovery_method,
            "Success / Failure": status_str,
            "Recovery Time": time_str,
            "Recovery Cost": cost_str,
            "Tokens Used": tokens_str
        }

        # 1. Append to CSV
        try:
            with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    record["Experiment ID"],
                    record["Error / Failure Code"],
                    record["Error Category"],
                    record["Repair Level"],
                    record["Recovery Method Attempted"],
                    record["Success / Failure"],
                    record["Recovery Time"],
                    record["Recovery Cost"],
                    record["Tokens Used"]
                ])
        except Exception as e:
            console.print(f"[red]Failed writing to CSV log: {e}[/red]")

        # 2. Append to JSONL
        try:
            with open(JSONL_PATH, mode="a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            console.print(f"[red]Failed writing to JSONL log: {e}[/red]")

        # 3. Print Rich Table in Console
        status_style = "bold green" if success else "bold red"
        table = Table(
            title=f"📊 [bold cyan]Error Recovery Experiment Log[/bold cyan] ({experiment_id})",
            box=box.ROUNDED,
            border_style="cyan"
        )
        table.add_column("Error Code", style="bold yellow")
        table.add_column("Category", style="magenta")
        table.add_column("Level", style="blue")
        table.add_column("Recovery Method", style="white")
        table.add_column("Status", style=status_style)
        table.add_column("Time", style="cyan")
        table.add_column("Cost", style="green")
        table.add_column("Tokens", style="dim")

        table.add_row(
            error_code,
            error_category,
            repair_level,
            recovery_method,
            status_str,
            time_str,
            cost_str,
            f"Tot: {total_tokens} (P:{prompt_tokens}/C:{completion_tokens})"
        )
        console.print(table)

        return record

