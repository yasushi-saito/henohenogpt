#!./venv/bin/python3

from dataclasses import dataclass
from importlib.metadata import version
import logging
from typing import Optional
from jaxtyping import Float
import tiktoken
import torch
from torch import Tensor, nn

logging.basicConfig(
    format="%(levelname)s:%(filename)s:%(lineno)d:%(message)s",
    level=logging.DEBUG,
)

torch.manual_seed(123)
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
logging.info(f"Using {DEVICE} for pytorch")

from torch.utils.data import Dataset, DataLoader

TextBlock = Float[Tensor, "max_length"]

tokenizer = tiktoken.get_encoding("gpt2")


@dataclass
class Settings:
    pass


@dataclass
class Config:
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias = False


class GELU(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Float[Tensor, "shape"]) -> Float[Tensor, "shape"]:
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(
        self, x: Float[Tensor, "batch_size context_length emb_dim"]
    ) -> Float[Tensor, "batch_size context_length emb_dim"]:
        assert (
            x.shape[2] == self.cfg.emb_dim
        ), f"x.shape={x.shape}, want={self.cfg.emb_dim}"
        return self.layers(x)


class LayerNorm(nn.Module):

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: Float[Tensor, "shape"]) -> Float[Tensor, "shape"]:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class MultiHeadAttention(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg
        self.d_out = cfg.emb_dim
        d_in = cfg.emb_dim
        d_out = cfg.emb_dim
        self.d_out = d_out
        assert (
            self.d_out % cfg.n_heads == 0
        ), "d_out must be divisible by n_heads"

        self.head_dim = (
            d_out // cfg.n_heads
        )  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=cfg.qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=cfg.qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(
            d_out, d_out
        )  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(cfg.drop_rate)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(cfg.context_length, cfg.context_length), diagonal=1
            ),
        )

    def forward(
        self, x: Float[Tensor, "batch_size context_length emb_dim"]
    ) -> Float[Tensor, "batch_size context_length emb_dim"]:
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `n_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, n_heads, head_dim)
        keys = keys.view(b, num_tokens, self.cfg.n_heads, self.head_dim)
        values = values.view(b, num_tokens, self.cfg.n_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.cfg.n_heads, self.head_dim)

        # Transpose: (b, num_tokens, n_heads, head_dim) -> (b, n_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(
            2, 3
        )  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.n_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        # print("FORWARD: ", x.shape, "=>", context_vec.shape)
        return context_vec


class TransformerBlock(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.att = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        # A
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        shortcut = x  # B
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # C
        return x


class GPTModel(nn.Module):

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )  # A
        self.final_norm = LayerNorm(cfg.emb_dim)  # B
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(
        self, in_idx: Float[Tensor, "batch_size context_length"]
    ) -> Float[Tensor, "batch_size context_length vocab_size"]:
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class TextDataset(Dataset):

    def __init__(self, txt: str, max_length: int, stride: int) -> None:
        self.input_ids: list[TextBlock] = []
        self.target_ids: list[TextBlock] = []

        token_ids = tokenizer.encode(txt)  # A

        for i in range(0, len(token_ids) - max_length, stride):  # B
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):  # C
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[TextBlock, TextBlock]:
        # Returns tuple (context, target)
        return self.input_ids[idx], self.target_ids[idx]


class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  # A
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )  # B

    def forward(self, x):
        batch_size, num_tokens, embedding_size = x.shape  # C
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)  # C
        attn_scores.masked_fill_(  # D
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


def text_to_token_ids(text):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):  # A
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # B
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]  # C
        probas = torch.softmax(logits, dim=-1)  # D
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # E
        idx = torch.cat((idx, idx_next), dim=1)  # F

    return idx


def calc_loss_batch(
    input_batch: Float[Tensor, "batch_size context_length"],
    target_batch: Float[Tensor, "batch_size context_length"],
    model: GPTModel,
) -> float:
    input_batch, target_batch = input_batch.to(DEVICE), target_batch.to(DEVICE)

    logits = model(input_batch)  # (batch_size, context_length, vocab_size)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    logging.info(f"Calcloss logits={logits.shape} inputogits={input_batch.shape}")
    logging.info(f"Calcloss logits(flattened)={logits.flatten(0, 1).shape} target={target_batch.flatten().shape}")
    logging.info(
        f"Calcloss batch: logits={logits.shape} input={input_batch.shape} =>"
        f" {loss}"
    )
    return loss


def calc_loss_loader(
    data_loader: DataLoader, model: GPTModel, num_batches: Optional[int] = None
):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(
    model: GPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    eval_iter: int,
) -> tuple[float, float]:
    model.eval()  # A
    with torch.no_grad():  # B
        train_loss = calc_loss_loader(
            train_loader, model, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, num_batches=eval_iter)
    model.train()
    logging.info(f"EvalateModel: trainloss={train_loss} valueloss={val_loss}")
    return train_loss, val_loss


def train_model_simple(
    model: GPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
):
    train_losses, val_losses, track_tokens_seen = [], [], []  # A
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):  # B
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # C
            loss = calc_loss_batch(input_batch, target_batch, model)
            loss.backward()  # D
            logging.info(f"Loss: {loss}")
            optimizer.step()  # E
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:  # F
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                logging.info(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(model, start_context)  # G
    return train_losses, val_losses, track_tokens_seen


def create_dataloader_v1(
    txt: str,
    batch_size,
    max_length,
    stride,
    shuffle=True,
    drop_last=True,
    num_workers=0,
) -> DataLoader:
    # Create dataset
    dataset = TextDataset(txt, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


def generate_and_print_sample(model, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context).to(DEVICE)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )
        decoded_text = token_ids_to_text(token_ids)
        logging.info(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def main_3():
    cfg = Config(context_length=256)
    torch.manual_seed(123)
    model = GPTModel(cfg)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0004, weight_decay=0.1
    )  # A
    num_epochs = 10

    path = "LLMs-from-scratch/appendix-D/01_main-chapter-code/the-verdict.txt"
    with open(path) as fd:
        text_data = fd.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    batch_size = 2
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=batch_size,
        max_length=cfg.context_length,
        stride=cfg.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=batch_size,
        max_length=cfg.context_length,
        stride=cfg.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=1,
        start_context="Every effort moves you",
    )
    return
    # txt2 = "Every day holds a"

    # batch = []
    # batch.append(torch.tensor(tokenizer.encode(txt1)))
    # batch.append(torch.tensor(tokenizer.encode(txt2)))
    # batch = torch.stack(batch, dim=0)
    # print("batch shape=", batch.shape)

    model.eval()
    out = generate_text_simple(
        model, text_to_token_ids(start_context), 10, cfg.context_length
    )
    # out = model(batch)
    # print("Input batch:\n", batch)
    logging.info(f"\nOutput shape: {out.shape}")
    # logging.info("Output: ", out)
    logging.info("Output: {tokenizer.decode(out.squeeze(0).tolist())}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total number of parameters: {total_params:,}")


main_3()

# Local Variables: ***
# pyformat-args: "-i --indent_size=4" ***
# python-indent-offset: 4 ***
# End: ***
