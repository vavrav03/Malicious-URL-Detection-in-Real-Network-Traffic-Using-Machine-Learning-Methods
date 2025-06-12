"""Everything that has to do with starting / loading from mlflow experiments, runs, model storages (except for datasets)"""

import datetime
import os
from types import SimpleNamespace

import mlflow
from torch.utils.data import DataLoader

from utils.base_models import BaseModelManager, log_dict_recursively_mlflow
from utils.dataset import URLDatasetPart
from utils.transformers import TransformerModelManager
from utils.output import print_dict_level1_inline


def get_run_name(prefix, model_type, dataset_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H-%M")
    return f"{prefix}-{model_type}-{dataset_name}-{timestamp}"


def setup_mlflow_client(debugging=False):
    client = mlflow.MlflowClient()
    experiments_folder_path = os.getenv("EXPERIMENTS_PATH")
    experiment_name = os.getenv("EXPERIMENT_NAME") if not debugging else "debugging"

    assert experiments_folder_path is not None and experiment_name is not None
    experiment_path = os.path.join(experiments_folder_path, experiment_name)
    mlflow.set_experiment(experiment_path)
    return client, experiment_path, experiments_folder_path, experiment_name


def make_json_serializable(obj):
    """
    Recursively convert obj into JSON-serializable form:
    - dicts and lists are walked
    - ints, floats, bools, None, str stay as-is
    - everything else → str(obj)
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)


def log_dict_mlflow(dict, path):
    mlflow.log_dict(make_json_serializable(dict), artifact_file=path)


def get_file_size_mb(file_path):
    """Returns the size of a file in megabytes (MB)."""
    if os.path.exists(file_path):
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        return file_size_mb
    else:
        return


def safe_cast(v):
    """Convert string values into appropriate Python types if possible,
    including simple flat lists like '[a, b, c]'."""
    if v == "None":
        return None

    if isinstance(v, str):
        if v == "True":
            return True
        if v == "False":
            return False

    if isinstance(v, str) and v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        parts = [item.strip() for item in inner.split(",")]
        return [safe_cast(item) for item in parts]
    try:
        return int(v)
    except (ValueError, TypeError):
        pass

    try:
        return float(v)
    except (ValueError, TypeError):
        pass

    return v


def load_model_from_run(run_id, device=None):
    model_uri = f"runs:/{run_id}/classification_model"
    components = mlflow.transformers.load_model(model_uri=model_uri, return_type="components")
    model = components["model"]
    tokenizer = components["tokenizer"]
    run = mlflow.get_run(run_id)
    args = SimpleNamespace(**{k: safe_cast(v) for k, v in run.data.params.items()})
    return TransformerModelManager(args, model, tokenizer, device)



def do_evaluation_run(model: BaseModelManager, dataset_part: URLDatasetPart, args, run_name, store_predictions):
    with mlflow.start_run(run_name=run_name) as run:
        print(f"[runner] Run name: {run_name}")
        print(f"[runner] Run id {run.info.run_id}")
        print(f"[runner] Parent run args: {model.args}")
        print(f"[runner] Args: {args}")
        print()
        mlflow.log_params(vars(args))
        # we should use validation, because we are choosing which model is better
        dataset_part.log_to_mlflow(should_save=False)
        eval_data_loader = DataLoader(
            dataset_part,
            batch_size=model.args.batch_size,
            shuffle=False,
            collate_fn=model.prepare_batch,
        )

        eval_metrics_parent, eval_output_parent, alternative_threshold_metrics_parent = model.evaluate(eval_data_loader)
        log_dict_recursively_mlflow(eval_metrics_parent, prefix="eval", step=0)
        print("[runner] PARENT eval metrics:")
        print_dict_level1_inline(eval_metrics_parent)
        model.log_persistent_performance(
            metrics=eval_metrics_parent,
            output=eval_output_parent,
            alternative_threshold_metrics=alternative_threshold_metrics_parent,
            store_predictions=store_predictions,
        )
        perf_result = model.measure_performance(eval_data_loader)
        print("[runner] Evaluating performance")
        print_dict_level1_inline(perf_result)