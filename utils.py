import os
import uuid
import json
import csv
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List

# --------------------------
# Configuration structures
# --------------------------
@dataclass
class ExperimentManager:
    """
    Holds all experiment arguments/config and handles
    setup + saving of logs, configs, metrics, etc.
    """
    # ----------------
    # Required fields
    # ----------------
    model_name_or_path: str = "roberta-large"
    task_name: str = "boolq"
    approach: str = "head"  # or 'prompt'
    subset_sizes: List[int] = field(default_factory=lambda: [50,100,250,500,1000,1500])
    seeds: List[int] = field(default_factory=lambda: [0,1,2])
    
    # ----------------
    # Training params
    # ----------------
    learning_rate: float = 1e-5
    train_batch_size: int = 4
    eval_batch_size: int = 4
    num_train_epochs: int = 50
    min_train_steps: int = 250
    
    # ----------------
    # Infrastructure
    # ----------------
    output_dir: str = "./outputs"
    max_train_samples: int = None
    max_eval_samples: int = None

    # ----------------
    # Internal state (populated later)
    # ----------------
    master_log_path: str = field(init=False)
    experiment_dir: str = field(init=False, default=None)
    current_seed: int = field(init=False, default=None)
    current_subset_size: int = field(init=False, default=None)

    def __post_init__(self):
        """
        Called automatically after the dataclass is initialized.
        We create the master log path here, ensuring the output_dir is set.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        self.master_log_path = os.path.join(self.output_dir, "master_log.csv")

    def initialize_experiment_run(self, seed: int, subset_size: int):
        """
        Sets up a unique experiment directory for the given seed + subset size,
        and configures logging accordingly.
        """
        self.current_seed = seed
        self.current_subset_size = subset_size
        
        # Create a unique directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]
        dir_name = f"{timestamp}_{unique_id}_task-{self.task_name}_approach-{self.approach}_seed-{seed}_subset-{subset_size}"
        self.experiment_dir = os.path.join(self.output_dir, dir_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Setup logging
        self._setup_logging()
        logging.info(f"Created experiment directory: {self.experiment_dir}")
        
        # Save the config (including updated seed/subset)
        self.save_config()

    def _setup_logging(self):
        """
        Sets up logging to both console and a file in the experiment directory.
        """
        log_file = os.path.join(self.experiment_dir, "experiment.log")
        
        # You can reset or configure the logger here.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def save_config(self):
        """
        Dumps the current dataclass config (as a dict) into `config.json` 
        in the experiment_dir. The seed + subset_size will reflect the 
        current run's values.
        """
        config_path = os.path.join(self.experiment_dir, "config.json")
        
        # Convert to a dict. The fields current_seed/current_subset_size 
        # have been set just now by `initialize_experiment_run`.
        config_dict = asdict(self)
        
        # Some fields (like `master_log_path`, `experiment_dir`) may not be
        # relevant to store, but we'll include them for completeness.
        
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        
        logging.info(f"Configuration saved to {config_path}")

    def save_metrics(self, metrics: dict):
        """
        Saves the experiment metrics to a JSON file in the current 
        experiment directory.
        """
        metrics_path = os.path.join(self.experiment_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {metrics_path}")

    def append_to_master_log(self, metrics: dict):
        """
        Appends the experiment configuration + metrics to a master CSV log
        located in `self.master_log_path`.
        """
        file_exists = os.path.isfile(self.master_log_path)
        
        # Combine config + metrics into one dictionary for the CSV row
        row_dict = asdict(self)  # all fields of the dataclass
        # Merge in the metrics. If there's a name collision, metrics will override.
        row_dict.update(metrics)

        # Write to CSV
        with open(self.master_log_path, "a", newline='') as csvfile:
            fieldnames = list(row_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)
        
        logging.info(f"Experiment appended to master log at {self.master_log_path}")