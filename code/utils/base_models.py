"""
- Unified training and evaluation interface for PyTorch-based and Onnx-based models (they share evaluation mechanism)
- Prevents inconsistency in evaluation logic across experiments or forks
- Centralized code for logging metrics, curves, and confusion matrices to MLflow
"""

import argparse
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
import mlflow
import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import onnxruntime as ort

from utils.focal_loss import BinaryFocalLoss
from utils.dataset import URLDatasetPart
from utils.output import print_dict_level1_inline


def get_loss_function(args, label_count):
    if args.loss == "focal":
        if label_count != 2:
            raise ValueError("Focal loss is only supported for binary classification (label_count == 2)")
        return BinaryFocalLoss(alpha=args.focal_loss_alpha, gamma=args.focal_loss_gamma)
    elif args.loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid loss function provided: {args.loss}")


def should_stop_early(val_f1_history, patience):
    if len(val_f1_history) > patience and max(val_f1_history[-patience:]) < max(val_f1_history):
        return True
    return False


def get_roc_curve_figure(true_labels, probabilities):
    fpr, tpr, _ = roc_curve(true_labels, probabilities)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC Curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    plt.close()
    return fig


def get_precision_recall_curve_figure(true_labels, probabilities):
    precision, recall, _ = precision_recall_curve(true_labels, probabilities)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label="Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    plt.close()
    return fig


def get_confusion_matrix_figure(true_labels, predictions):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(true_labels, predictions, ax=ax)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.close()
    return fig


def find_decision_threshold_maximizing_f1(probs, labels, resolution=100):
    """
    Finds the decision threshold for label prediction by maximizing macro f1 score.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    thresholds = np.linspace(0.0, 1.0, resolution)
    best_threshold = 0.5
    best_f1 = 0.0

    for t in thresholds:
        predictions = (probs >= t).astype(int)
        f1 = f1_score(labels, predictions, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold


def log_dict_recursively_mlflow(d: dict, prefix, step):
    """
    Logs a dictionary recursively to MLflow.
    Logs scalar values as metrics; everything else as params (JSON-encoded).
    """
    for key, value in d.items():
        full_key = f"{prefix}_{key}"
        if isinstance(value, dict):
            log_dict_recursively_mlflow(value, full_key, step)
        else:
            try:
                mlflow.log_metric(full_key, float(value), step=step, synchronous=False)
            except Exception as e:
                print(f"[mlflow] Skipping key '{full_key}' (unserializable): {e}")


def calculate_metrics_common(true_labels, class_probabilities, predictions):
    if not (len(true_labels) == len(class_probabilities) == len(predictions)):
        raise ValueError("true_labels, probabilities and predictions must have equal length")
    metrics = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_labels, predictions)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def calculate_metrics_binary(
    true_labels,
    class_probabilities,
    predictions,
):
    metrics = calculate_metrics_common(true_labels, class_probabilities, predictions)
    cm = metrics["confusion_matrix"]
    metrics["tp"] = cm[1][1]
    metrics["tn"] = cm[0][0]
    metrics["fp"] = cm[0][1]
    metrics["fn"] = cm[1][0]
    metrics["fpr"] = metrics["fp"] / (metrics["fp"] + metrics["tn"])
    metrics["roc_auc_score"] = roc_auc_score(true_labels, class_probabilities[:, 1])

    return metrics


def log_persistent_performance(
    metrics: dict,
    best_threshold_metrics: dict,
    true_labels,
    class_probabilities: np.ndarray,
    predictions,
    prefix: str = "",
    store_predictions: bool = False,
) -> None:
    mlflow.log_dict(metrics, artifact_file=f"{prefix}best_model_metrics.json")
    mlflow.log_dict(
        best_threshold_metrics,
        artifact_file=f"{prefix}best_model_metrics_with_best_decision_threshold.json",
    )

    if store_predictions:
        df_out = pd.DataFrame(
            {
                "true_label": true_labels.tolist(),
                "class_probabilities": class_probabilities.tolist(),
                "prediction": predictions.tolist(),
            }
        )
        mlflow.log_text(df_out.to_csv(index=False), f"{prefix}best_model_output.csv")

    mlflow.log_figure(
        get_confusion_matrix_figure(true_labels, predictions),
        f"{prefix}confusion_matrix.png",
    )
    if class_probabilities.shape[1] != 2:
        return
    class_1_probabilities = class_probabilities[:, 1]
    mlflow.log_figure(
        get_roc_curve_figure(true_labels, class_1_probabilities),
        f"{prefix}roc_curve.png",
    )
    mlflow.log_figure(
        get_precision_recall_curve_figure(true_labels, class_1_probabilities),
        f"{prefix}precision_recall_curve.png",
    )


def perf_time_synchronized(device: torch.device):
    """if you do model(batch) and then measure time using time.perf_counter right after that, synchronization error can occur
    model will still be calculated on GPU, but this will be called on CPU. This method ensures that CPU waits when command model(batch) is finished
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


class BaseModelManager(ABC):
    """
    Base class for all models (pytorch based, onnx based), which offers unified evaluation framework
    """

    args: argparse.Namespace
    # onnx needs pytorch device to calculate stuff with loss
    device: torch.device
    loss_fn: Callable

    def __init__(self, args, device=None):
        self.args = args
        if hasattr(self.args, 'label_count'):
            self.label_count = self.args.label_count
        else:
            self.label_count = 2
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = get_loss_function(args, self.label_count)

    def evaluate(self, data_loader, decision_threshold=None):
        if decision_threshold is None:
            decision_threshold = self.args.decision_threshold
        n_samples = len(data_loader.dataset)
        true_labels = np.empty(n_samples, dtype=np.int64)
        class_probs = np.empty((n_samples, self.label_count), dtype=np.float32)
        write_ptr = 0
        batch_losses = []

        self.run_before_eval()
        pbar = tqdm.tqdm(data_loader, bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}")
        with torch.no_grad():
            for batch_number, batch in enumerate(pbar):
                inputs, labels = batch[0], batch[1]

                inputs = self.get_moved_batch_inputs_to_device(inputs)
                logits = self.forward(inputs)
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu()

                batch_size = labels.size(0)
                idx_end = write_ptr + batch_size

                loss = self.loss_fn(logits, labels.to(self.device)).cpu().item()

                class_probs[write_ptr:idx_end, :] = probs.numpy()
                true_labels[write_ptr:idx_end] = labels.cpu().numpy()
                batch_losses.append(loss)

                write_ptr = idx_end

        if self.label_count == 2:
            class1_p = class_probs[:, 1]
            predictions = (class1_p >= decision_threshold).astype(int)
            metrics = calculate_metrics_binary(true_labels, class_probs, predictions)
            best_thr = find_decision_threshold_maximizing_f1(class1_p, true_labels)
            metrics["best_decision_threshold"] = best_thr
            alt_metrics = calculate_metrics_binary(true_labels, class_probs, (class1_p >= best_thr).astype(int))
        else:
            predictions = class_probs.argmax(axis=1)
            metrics = calculate_metrics_common(true_labels, class_probs, predictions)
            metrics["best_decision_threshold"] = None
            alt_metrics = metrics
        metrics["average_loss"] = float(np.mean(batch_losses))

        output = {
            "true_labels": true_labels,
            "class_probabilities": class_probs,
            "predictions": predictions,
        }
        return metrics, output, alt_metrics

    def measure_performance(
        self, data_loader: DataLoader, log_to_mlflow=True, enough_samples_to_measure=5000, prefix=""
    ):
        self.run_before_eval()
        device = self.device
        # --- Warmup Phase ---
        warmup_batches = 2
        print("[profiler] Warmup initiated")
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= warmup_batches:
                    break
                inputs = batch[0]
                inputs = self.get_moved_batch_inputs_to_device(inputs)
                logits = self.forward(inputs)
                _ = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
                # Synchronize to ensure warmup execution is complete.
                perf_time_synchronized(device)

        # --- Measurement Phase ---
        time_without_tokenizer_with_softmax = 0.0
        time_without_tokenizer_without_softmax = 0.0
        total_samples = 0
        n_samples_to_use = (
            min(enough_samples_to_measure, len(data_loader.dataset))
            if hasattr(data_loader, "dataset")
            else enough_samples_to_measure
        )
        print(f"[profiler] n_samples_to_use {n_samples_to_use}")
        progress = tqdm.tqdm(
            total=n_samples_to_use,
            desc="Samples processed",
            unit="sample",
            dynamic_ncols=False,  # Fixed column width
            leave=False,  # Remove progress bar after completion
            bar_format="{l_bar}{bar:20}{r_bar}",  # Customize format as desired
        )
        with_tokenizer_start = perf_time_synchronized(device)
        time_total_without_tokenizer = 0.0
        time_without_tokenizer_without_softmax_with_moving = 0.0
        with torch.no_grad():
            for batch in data_loader:
                start_all = perf_time_synchronized(device)
                batch_size = batch[1].size(0)
                total_samples += batch_size
                progress.update(batch_size)

                inputs = batch[0]
                start_forward_with_moving_to_device = perf_time_synchronized(device)
                inputs = self.get_moved_batch_inputs_to_device(inputs)
                start_forward = perf_time_synchronized(device)
                logits = self.forward(inputs)
                end_model = perf_time_synchronized(device)

                _ = torch.nn.functional.softmax(logits, dim=-1)

                end_all = perf_time_synchronized(device)
                time_without_tokenizer_without_softmax_with_moving += end_model - start_forward_with_moving_to_device
                time_without_tokenizer_without_softmax += end_model - start_forward
                time_without_tokenizer_with_softmax += end_all - start_forward

                time_total_without_tokenizer += perf_time_synchronized(device) - start_all
                if total_samples >= enough_samples_to_measure:
                    break
        progress.close()
        time_with_tokenizer = perf_time_synchronized(device) - with_tokenizer_start

        metrics = {
            "time_n_per_s_total_with_tokenizer": total_samples / time_with_tokenizer,
            "time_n_per_s_total_without_tokenizer": total_samples / time_total_without_tokenizer,
            "time_n_per_s_forward_with_softmax": total_samples / time_without_tokenizer_with_softmax,
            "time_n_per_s_forward_without_softmax_with_moving": total_samples / time_without_tokenizer_without_softmax_with_moving,
            "time_n_per_s_forward_without_softmax": total_samples / time_without_tokenizer_without_softmax,
            "time_s_per_1_total_with_tokenizer": time_with_tokenizer / total_samples,
            "time_s_per_1_total_without_tokenizer": time_total_without_tokenizer / total_samples,
            "time_s_per_1_forward_with_softmax": time_without_tokenizer_with_softmax / total_samples,
            "time_s_per_1_forward_without_softmax": time_without_tokenizer_without_softmax / total_samples,
            "time_s_per_1_forward_without_softmax_with_moving": time_without_tokenizer_without_softmax_with_moving / total_samples,
        }

        if log_to_mlflow:
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_dict(json.loads(json.dumps(metrics, default=float)), artifact_file=f"{prefix}_performance.json")

        return metrics

    def run_before_eval(self):
        pass

    @abstractmethod
    def forward(self, batch_input):
        """
        - move everything that you use from batch_input into correct device
        - return logits
        - do not use torch.without_grad here
        """
        pass

    @abstractmethod
    def save_to_mlflow(self):
        pass

    def get_moved_batch_inputs_to_device(self, batch_input):
        return batch_input


class TorchModelManager(BaseModelManager):
    """
    Pytorch based model which offers unified training function for all descendents
    """

    args: argparse.Namespace
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: Callable

    def __init__(self, args, model, device=None, move_to_device=True):
        super().__init__(args, device)
        # device created in parent if None
        self.model = model
        if move_to_device:
            self.model.to(self.device)
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
        self.optimizer = None

    def train_epoch(self, epoch, train_data_loader):
        print(f"[training] Starting epoch {epoch}")
        self.update_optimizer(epoch)
        pbar = tqdm.tqdm(train_data_loader, bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}")
        pbar.set_description("Training batches")
        self.model.train()
        for batch_number, batch in enumerate(pbar):
            self.train_step(
                batch=batch,
                batch_number=batch_number,
            )
        

    def train(self, train_data_loader, eval_data_loader):
        eval_f1_history = []
        for epoch in range(self.args.epochs_max):
            self.train_epoch(epoch, train_data_loader)

            eval_metrics, eval_output, alternative_threshold_metrics = self.evaluate(eval_data_loader)
            if self.label_count == 2:
                f1 = eval_metrics["1"]["f1-score"]
            else:
                f1 = eval_metrics["macro"]["f1-score"]
            log_dict_recursively_mlflow(eval_metrics, prefix="eval", step=epoch)
            eval_f1_history.append(f1)
            print(f"[training] Evaluation result")
            print_dict_level1_inline(eval_metrics)
            if self.args.patience is not None:
                if should_stop_early(eval_f1_history, patience=self.args.patience):
                    print("[training] Early stopping triggered!")
                    break
            if f1 >= max(eval_f1_history[:-1], default=0):
                print(f"[training] New best class f1-score {max(eval_f1_history)} achieved at epoch {epoch}.")
                log_persistent_performance(
                    metrics=eval_metrics,
                    best_threshold_metrics=alternative_threshold_metrics,
                    true_labels=eval_output["true_labels"],
                    class_probabilities=eval_output["class_probabilities"],
                    predictions=eval_output["predictions"],
                    # prefix="",
                    # store_predictions=True,
                )
                self.save_to_mlflow()

    def train_step(self, batch, batch_number):
        inputs = batch[0]
        labels = batch[1]
        inputs = self.get_moved_batch_inputs_to_device(batch_input=inputs)
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, labels.to(self.device))
        # loss = self.loss_fn(logits=logits, batch=batch)
        if self.args.label_count == 2:
            class_1_probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
            predictions = (class_1_probs >= self.args.decision_threshold).int().detach().cpu()
        else:
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predictions = probabilities.argmax(dim=-1).detach().cpu()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        mlflow.log_metrics(
            {
                "train_batch_average_loss": loss.item(),
                "train_batch_accuracy": float(torch.sum(torch.eq(predictions, labels).long()).item() / len(labels)),
            },
            step=batch_number,
            synchronous=False,
        )

    def do_training_run(
        self,
        args,
        run_name,
        train_dataset: URLDatasetPart,
        eval_dataset: URLDatasetPart,
        num_workers_train=0,
        num_workers_eval=0,
        persistent_workers=False,
        prefetch_factor=None,
        should_save_datasets=False,
    ):
        """Do training run and evaluate on test set designated for evaluation

        Use test set from earlier from the script - everything has the same set
        """
        print(f"[runner] Loading dataset...")
        print(f"[runner] Run name: {run_name}")
        print(f"[runner] Args: {args}")
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=self.prepare_batch,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            num_workers=num_workers_train,
        )
        eval_data_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=self.prepare_batch,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            num_workers=num_workers_eval,
        )
        print()
        with mlflow.start_run(run_name=run_name) as run:
            print(f"[runner]: Run id {run.info.run_id}")
            mlflow.log_params(vars(args))
            # log dataset info without actually storing the datasets
            train_dataset.log_to_mlflow("train", args.dataset_name, should_save=should_save_datasets)
            eval_dataset.log_to_mlflow("eval", args.dataset_name, should_save=should_save_datasets)
            self.train(train_data_loader=train_data_loader, eval_data_loader=eval_data_loader)
            result = self.measure_performance(data_loader=eval_data_loader)
            print("[runner] Evaluating performance")
            print_dict_level1_inline(result)

    def _get_model(self):
        """This is necessary if model is wrapped in DataParallel and we need the model class"""
        return self.model.module if hasattr(self.model, "module") else self.model

    def run_before_eval(self):
        self.model.eval()

    @abstractmethod
    def update_optimizer(self, epoch: int):
        """
        Update self.optimizer whenever is necessary
        """
        pass

    @abstractmethod
    def prepare_batch(self, batch):
        """collate_fn for dataloader. Output format must be (anything model needs), labels"""
        pass

    def create_onnx_file(self, target_file):
        self.model.cpu()
        self.model.eval()
        pass


class OnnxModelManager(BaseModelManager):

    model_session: ort.InferenceSession

    def __init__(self, model_session, args):
        super().__init__(args)
        self.model_session = model_session

    @staticmethod
    def get_gpu_session(model_path):
        opt_so = ort.SessionOptions()
        opt_so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        opt_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(model_path, opt_so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    
    @staticmethod
    def get_cpu_session(model_path):
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])