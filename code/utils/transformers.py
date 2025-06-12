import copy

import mlflow
from transformers import BertForSequenceClassification, AutoTokenizer
import torch

from utils.base_models import OnnxModelManager, TorchModelManager, perf_time_synchronized


def get_huggingface_model_checkpoint(model_type):
    checkpoints = {
        "bert_tiny": "google/bert_uncased_L-2_H-128_A-2",
        "bert_mini": "google/bert_uncased_L-4_H-256_A-4",
        "bert_small": "google/bert_uncased_L-4_H-512_A-8",
        "bert_medium": "google/bert_uncased_L-8_H-512_A-8",
        "bert_base": "bert-base-uncased",
        # "charbert": "imvladikon/charbert-bert-wiki",
    }
    if model_type not in checkpoints:
        raise ValueError(f"Invalid model_type '{model_type}'. Choose from {list(checkpoints.keys())}.")
    return checkpoints[model_type]


class TransformerModelManager(TorchModelManager):

    tokenizer: AutoTokenizer

    def __init__(self, args, model, tokenizer, device, move_to_device=True):
        super().__init__(args, model, device=device, move_to_device=move_to_device)
        self.tokenizer = tokenizer

    @classmethod
    def from_args(cls, args, device=None):
        model_checkpoint = get_huggingface_model_checkpoint(args.model_type)
        args.model_checkpoint = model_checkpoint
        if args.dropout is None:
            # let default values be set
            model = BertForSequenceClassification.from_pretrained(
                model_checkpoint, num_labels=args.label_count
            )
        else:
            model = BertForSequenceClassification.from_pretrained(
                model_checkpoint, num_labels=args.label_count, hidden_dropout_prob=args.dropout
            )
            
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        return TransformerModelManager(args, model, tokenizer, device)

    def forward(self, batch_input):
        output = self.model(input_ids=batch_input["input_ids"], attention_mask=batch_input["attention_mask"])
        return output.logits
    

    def get_moved_batch_inputs_to_device(self, batch_input):
        return {
            "input_ids": batch_input["input_ids"].to(self.device),
            "attention_mask": batch_input["attention_mask"].to(self.device),
        }

    def update_optimizer(self, epoch: int):
        """
        Only 2 times new optimizer should be initialized - at the beginning and at freeze_epochs epoch
        """
        freeze_epochs = self.args.freeze_epochs or 0
        if freeze_epochs <= 0:
            # Single‐init, unfrozen
            if self.optimizer is None:
                freeze_transformer = False
            else:
                return
        else:
            # Two‐phase init
            if self.optimizer is None:
                # Phase 1: at epoch 0
                freeze_transformer = True
            elif epoch == freeze_epochs:
                # Phase 2: at the freeze_epochs boundary
                freeze_transformer = False
            else:
                return

        print(f"[training] Initializing new optimizer with "
            f"{'un' if not freeze_transformer else ''}freezed transformer layers")

        model = self._get_model()
        
        for param in model.bert.parameters():
            param.requires_grad = not freeze_transformer

        self.optimizer = torch.optim.AdamW(
            [
                {"params": model.bert.parameters(), "lr": self.args.bert_learning_rate},
                {"params": model.classifier.parameters(), "lr": self.args.classifier_learning_rate},
            ],
            weight_decay=self.args.weight_decay,
        )

    def save_to_mlflow(self):
        model = copy.deepcopy(self._get_model())
        model.to("cpu")
        mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "tokenizer": self.tokenizer,
            },
            artifact_path="classification_model",
            input_example="duckduckgo.com/?t=h_&q=bacon&ia=web",
            task="text-classification",
        )

    def prepare_batch(self, batch):
        urls, labels = zip(*batch)
        encoded_batch = self.tokenizer(
            urls, padding="max_length", truncation=True, max_length=self.args.max_sequence_length, return_tensors="pt"
        )
        input_dict = {
            "urls": urls,
            "input_ids": encoded_batch["input_ids"],
            "attention_mask": encoded_batch["attention_mask"],
        }
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return input_dict, labels_tensor
    
    def prepare_batch_measured(self, batch):
        point_start = perf_time_synchronized(self.device)
        urls, labels = zip(*batch)
        point_zipping = perf_time_synchronized(self.device)
        encoded_batch = self.tokenizer(
            urls, padding=True, truncation=True, max_length=self.args.max_sequence_length, return_tensors="pt"
        )
        point_tokenizer = perf_time_synchronized(self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        point_labels_tensor = perf_time_synchronized(self.device)
        input_dict = {
            "urls": urls,
            "input_ids": encoded_batch["input_ids"],
            "attention_mask": encoded_batch["attention_mask"],
            "measurements": {
                "time_zipping": point_zipping - point_start,
                "time_tokenizer": point_tokenizer - point_zipping,
                "time_labels_tensor": point_labels_tensor - point_tokenizer,
                "time_total": point_labels_tensor - point_start,
            },
        }
        return input_dict, labels_tensor

    def create_onnx_file(self, target_file):
        super().create_onnx_file(target_file=target_file)
        sample_item = "https://seznam.cz"
        dummy_input = self.tokenizer(sample_item, return_tensors="pt")

        torch.onnx.export(
            self.model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            target_file,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            opset_version=13,
            # optimize constant expressions at export time.
            do_constant_folding=True,
            # Tells ONNX that some dimensions (like batch size and sequence length) are variable,
            # so the model can handle inputs of different sizes at runtime
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
        )


class OnnxTransformerModelManager(OnnxModelManager):

    def __init__(self, model_session, tokenizer, args):
        super().__init__(model_session=model_session, args=args)
        self.tokenizer = tokenizer

    @classmethod
    def from_file(cls, model_path, tokenizer, args):
        model_session = OnnxModelManager.get_inference_session(model_path)
        return OnnxTransformerModelManager(model_session, tokenizer, args)

    def forward(self, batch_input):
        output = self.model_session.run(
            ["logits"], {"input_ids": batch_input["input_ids"], "attention_mask": batch_input["attention_mask"]}
        )
        # output is a list with number of outcomes defined in
        logits = output[0]
        return torch.tensor(logits, dtype=torch.float32).to(self.device)

    def prepare_batch(self, batch):
        urls, labels = zip(*batch)
        encoded_batch = self.tokenizer(
            urls,
            padding=True,
            truncation=True,
            max_length=self.args.max_sequence_length,
            return_tensors="pt",
        )
        input_dict = {
            "urls": urls,
            "input_ids": encoded_batch["input_ids"].numpy(),
            "attention_mask": encoded_batch["attention_mask"].numpy(),
        }
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return input_dict, labels_tensor

    def save_to_mlflow(self):
        # TODO
        pass
