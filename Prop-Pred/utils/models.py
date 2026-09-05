import os
import torch
import torch.nn as nn
from transformers import AutoModel


class AttnPool(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.scorer(x).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


def _mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(x.dtype)
    summed = (x * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def _find_transformer_blocks(model: nn.Module):
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    return None


def _freeze_all_but_last_n(model: nn.Module, unfreeze_last_n: int):
    for p in model.parameters():
        p.requires_grad = False

    blocks = _find_transformer_blocks(model)
    if blocks is None:
        print("WARNING: Could not locate transformer blocks to partially unfreeze; leaving base model fully frozen.")
        return

    if unfreeze_last_n <= 0:
        return

    for block in list(blocks)[-unfreeze_last_n:]:
        for p in block.parameters():
            p.requires_grad = True

    if hasattr(model, "norm"):
        for p in model.norm.parameters():
            p.requires_grad = True
    elif hasattr(model, "model") and hasattr(model.model, "norm"):
        for p in model.model.norm.parameters():
            p.requires_grad = True


class ChemlacticaRegressor(nn.Module):
    """
    Single-task regressor head on top of a ChemLlama backbone.
    """
    def __init__(
        self,
        model_name: str,
        pooling: str = "last_token",
        mlp_hidden_size: int = 512,
        mlp_layers: int = 2,
        dropout: float = 0.1,
        unfreeze_last_n: int = 0,
        tokenizer_len: int = 128258,
        revision: str = None,
        use_gradient_checkpointing: bool = None
    ):
        super().__init__()
        self.pooling = pooling
        token = os.environ.get("HF_TOKEN")
        self.backbone = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            token=token
        )
        self.backbone.config.use_cache = False
        self.backbone.resize_token_embeddings(tokenizer_len)
        _freeze_all_but_last_n(self.backbone, unfreeze_last_n=unfreeze_last_n)

        # Enable gradient checkpointing if explicitly requested or for 3B models
        is_3b = "3B" in model_name or "3b" in model_name
        should_checkpoint = use_gradient_checkpointing if use_gradient_checkpointing is not None else is_3b
        if should_checkpoint:
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()
            if hasattr(self.backbone, "enable_input_require_grads"):
                self.backbone.enable_input_require_grads()

        base_hidden = int(self.backbone.config.hidden_size)
        if pooling == "attn":
            self.pool = AttnPool(base_hidden, dropout=dropout)
        elif pooling in ("mean", "last_token"):
            self.pool = None
        else:
            raise ValueError(f"Unknown pooling='{pooling}'")

        mlp = []
        in_dim = base_hidden
        for _ in range(int(mlp_layers)):
            mlp.append(nn.Linear(in_dim, mlp_hidden_size))
            mlp.append(nn.GELU())
            mlp.append(nn.Dropout(dropout))
            in_dim = mlp_hidden_size
        self.shared_mlp = nn.Sequential(*mlp) if len(mlp) > 0 else nn.Identity()
        self.head = nn.Linear(in_dim, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state

        if self.pooling == "attn":
            pooled = self.pool(x, attention_mask=attention_mask)
        elif self.pooling == "last_token":
            batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
            last_token_idx = attention_mask.sum(1).long() - 1
            pooled = x[batch_idx, last_token_idx, :]
        else:
            pooled = _mean_pool(x, attention_mask=attention_mask)

        z = self.shared_mlp(pooled)
        return self.head(z).squeeze(-1)


class ChemlacticaMultiTaskRegressor(nn.Module):
    """
    Multi-task regressor head on top of a ChemLlama backbone (e.g. for Polaris ADME).
    """
    def __init__(
        self,
        model_name: str,
        tasks: list[str],
        pooling: str = "last_token",
        mlp_hidden_size: int = 512,
        mlp_layers: int = 2,
        dropout: float = 0.1,
        unfreeze_last_n: int = 0,
        tokenizer_len: int = 128258,
        revision: str = None,
        use_gradient_checkpointing: bool = None
    ):
        super().__init__()
        self.tasks = tasks
        self.pooling = pooling
        token = os.environ.get("HF_TOKEN")
        self.backbone = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            token=token
        )
        self.backbone.config.use_cache = False
        self.backbone.resize_token_embeddings(tokenizer_len)
        _freeze_all_but_last_n(self.backbone, unfreeze_last_n=unfreeze_last_n)

        # Enable gradient checkpointing if explicitly requested or for 3B models
        is_3b = "3B" in model_name or "3b" in model_name
        should_checkpoint = use_gradient_checkpointing if use_gradient_checkpointing is not None else is_3b
        if should_checkpoint:
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()
            if hasattr(self.backbone, "enable_input_require_grads"):
                self.backbone.enable_input_require_grads()

        base_hidden = int(self.backbone.config.hidden_size)
        if pooling == "attn":
            self.pool = AttnPool(base_hidden, dropout=dropout)
        elif pooling in ("mean", "last_token"):
            self.pool = None
        else:
            raise ValueError(f"Unknown pooling='{pooling}'")

        mlp = []
        in_dim = base_hidden
        for _ in range(int(mlp_layers)):
            mlp.append(nn.Linear(in_dim, mlp_hidden_size))
            mlp.append(nn.GELU())
            mlp.append(nn.Dropout(dropout))
            in_dim = mlp_hidden_size
        self.shared_mlp = nn.Sequential(*mlp) if len(mlp) > 0 else nn.Identity()

        self.heads = nn.ModuleDict({t: nn.Linear(in_dim, 1) for t in tasks})

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state

        if self.pooling == "attn":
            pooled = self.pool(x, attention_mask=attention_mask)
        elif self.pooling == "last_token":
            batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
            last_token_idx = attention_mask.sum(1).long() - 1
            pooled = x[batch_idx, last_token_idx, :]
        else:
            pooled = _mean_pool(x, attention_mask=attention_mask)

        z = self.shared_mlp(pooled)
        return {t: self.heads[t](z).squeeze(-1) for t in self.tasks}
