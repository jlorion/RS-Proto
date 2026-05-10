import os
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import requests
import psutil

from time import time
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

load_dotenv()

API_BASE_URL = os.getenv("CLOUDFLARE_API_BASE_URL")
HEADERS = {"Authorization": f"Bearer {os.getenv('CLOUDFLARE_API_TOKEN')}"}
MODEL_NAME = os.getenv("CLOUDFLARE_MODEL_NAME")

def create_prompt(text):
    return f"""
You are a content moderation assistant. Identify the list of [rationales] words or phrases from the text that make it hateful,
list of [derogatory language], and [list of cuss words] and [hate_classification] such as "hateful" or "non-hateful".
If there are none, respond exactly with [non-hateful] only.
Output should be in JSON format only. Text: {text}.
"""

def run_mistral_model(model, inputs):
    payload = {"messages": inputs}
    response = requests.post(f"{API_BASE_URL}{model}", headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()

def flatten_json_string(json_string):
    try:
        obj = json.loads(json_string)
        return json.dumps(obj, separators=(",", ":"))
    except:
        return json_string

def get_rationale_from_mistral(text, retries=10):
    for attempt in range(retries):
        try:
            inputs = [{"role": "user", "content": create_prompt(text)}]
            output = run_mistral_model(MODEL_NAME, inputs)
            
            result = output.get("result", {})
            response_text = result.get("response", "").strip()
            
            if not response_text or response_text.startswith("I cannot"):
                print(f"⚠️ Model returned 'I cannot...' — retrying ({attempt+1}/{retries})")
                continue  # retry
            cleaned_rationale = flatten_json_string(response_text).replace("\n", " ").strip()
            return cleaned_rationale
        
        except requests.exceptions.HTTPError as e:
            print(f"⚠️ HTTP Error on attempt {attempt+1}: {e}")
            if "RESOURCE_EXHAUSTED" in str(e) or e.response.status_code == 429:
                raise
    
    return "non-hateful"

def preprocess_rationale_mistral(raw_rationale):
    try:
        x = str(raw_rationale).strip()

        if x.startswith("```"):
            x = x.replace("```json", "").replace("```", "").strip()

        x = x.replace('""', '"')

        # Extract JSON object
        start = x.find("{")
        end = x.rfind("}") + 1
        if start == -1 or end == -1:
            return x.lower()  

        j = json.loads(x[start:end])

        keys = ["rationales", "derogatory_language", "cuss_words"]

        if all(k in j and isinstance(j[k], list) and len(j[k]) == 0 for k in keys):
            return "non-hateful"

        cleaned = {k: j.get(k, []) for k in keys}
        return json.dumps(cleaned).lower()

    except Exception:
        return str(raw_rationale).lower()
    
class TemporalCNN(nn.Module):
    def __init__(self, input_dim=768, num_filters=32, kernel_sizes=(3,4,5), dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence_embeddings, attention_mask=None):
        x = sequence_embeddings.transpose(1, 2).contiguous() 
    
        pooled_outputs = []
        for conv in self.convs:
            conv_out = conv(x)              
            conv_out = F.relu(conv_out)
            L_out = conv_out.size(2)
    
            if attention_mask is not None:
                mask = attention_mask.float()
                if mask.size(1) != L_out:
                    mask = F.interpolate(mask.unsqueeze(1), size=L_out, mode='nearest').squeeze(1)
                mask = mask.unsqueeze(1).to(conv_out.device)  # (B,1,L_out)
    
                neg_inf = torch.finfo(conv_out.dtype).min / 2
                max_masked = torch.where(mask.bool(), conv_out, neg_inf*torch.ones_like(conv_out))
                max_pooled = torch.max(max_masked, dim=2)[0]  # (B, num_filters)
    
                sum_masked = (conv_out * mask).sum(dim=2)    # (B, num_filters)
                denom = mask.sum(dim=2).clamp_min(1e-6)     # (B,1)
                mean_pooled = sum_masked / denom            # (B, num_filters)
            else:
                max_pooled = torch.max(conv_out, dim=2)[0]
                mean_pooled = conv_out.mean(dim=2)
    
            pooled_outputs.append(max_pooled)
            pooled_outputs.append(mean_pooled)
    
        out = torch.cat(pooled_outputs, dim=1)  
        out = self.dropout(out)
        return out


class MultiScaleAttentionCNN(nn.Module):
    def __init__(self, hidden_size=768, num_filters=32, kernel_sizes=(3,4,5), dropout=0.3):
        super().__init__()

        self.hidden_size = hidden_size
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList()
        self.pads  = nn.ModuleList()

        for k in self.kernel_sizes:
            pad_left  = (k - 1) // 2
            pad_right = k - 1 - pad_left

            self.pads.append(nn.ConstantPad1d((pad_left, pad_right), 0.0))

            self.convs.append(
                nn.Conv1d(hidden_size, num_filters, kernel_size=k, padding=0)
            )

        self.attn = nn.ModuleList([nn.Linear(num_filters, 1) for _ in self.kernel_sizes])
        self.output_size = num_filters * len(self.kernel_sizes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, mask):
        x = hidden_states.transpose(1, 2) 
        attn_mask = mask.unsqueeze(1).float()

        conv_outs = []

        for pad, conv, att in zip(self.pads, self.convs, self.attn):
            padded = pad(x)     
            c = conv(padded)     
            c = F.relu(c)
            c = c * attn_mask

            c_t = c.transpose(1, 2)   
            w = att(c_t)            
            w = w.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            w = F.softmax(w, dim=1)

            pooled = (c_t * w).sum(dim=1)  
            conv_outs.append(pooled)

        out = torch.cat(conv_outs, dim=1)   
        return self.dropout(out)
    
class ConcatModelWithRationale(nn.Module):
    def __init__(self,
                 hatebert_model,
                 additional_model,
                 hidden_size=768,
                 gumbel_temp=0.5,
                 freeze_additional_model=True,
                 cnn_num_filters=128,
                 cnn_kernel_sizes=(3, 4, 5),
                 cnn_dropout=0.3,
                 num_classes=2):
        super().__init__()
        self.hatebert_model = hatebert_model
        self.additional_model = additional_model
        self.gumbel_temp = gumbel_temp
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Freeze HateBERT embeddings and lower encoder layers
        for param in self.hatebert_model.embeddings.parameters():
            param.requires_grad = False

        for layer in self.hatebert_model.encoder.layer[:8]:
            for param in layer.parameters():
                param.requires_grad = False

        # Freeze additional model if requested
        if freeze_additional_model:
            for param in self.additional_model.parameters():
                param.requires_grad = False

        # Selector head for rationale extraction
        self.selector = nn.Linear(hidden_size, 1)

        # Temporal CNN over HateBERT embeddings
        self.temporal_cnn = TemporalCNN(
            input_dim=hidden_size,
            num_filters=cnn_num_filters,
            kernel_sizes=cnn_kernel_sizes,
            dropout=cnn_dropout
        )
        self.temporal_out_dim = cnn_num_filters * len(cnn_kernel_sizes) * 2

        # MultiScaleAttentionCNN over rationale embeddings
        self.msa_cnn = MultiScaleAttentionCNN(
            hidden_size=hidden_size,
            num_filters=cnn_num_filters,
            kernel_sizes=cnn_kernel_sizes,
            dropout=cnn_dropout
        )
        self.msa_out_dim = self.msa_cnn.output_size

        # === 4 branch-specific classifiers ===
        self.cls_head = nn.Linear(hidden_size, num_classes)
        self.rationale_head = nn.Linear(hidden_size, num_classes)
        self.temporal_head = nn.Linear(self.temporal_out_dim, num_classes)
        self.msa_head = nn.Linear(self.msa_out_dim, num_classes)

        # Learnable branch weights for weighted averaging
        # Initialized equally; softmax will normalize them
        self.branch_weights = nn.Parameter(torch.ones(4))

    def gumbel_sigmoid_sample(self, logits):
        noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
        y = logits + noise
        return torch.sigmoid(y / self.gumbel_temp)

    def forward(self,
                input_ids,
                attention_mask,
                additional_input_ids,
                additional_attention_mask,
                return_attentions=False):

        # =========================
        # Main text through HateBERT
        # =========================
        hatebert_out = self.hatebert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attentions,
            return_dict=True
        )
        hatebert_emb = hatebert_out.last_hidden_state         # (B, L, H)
        cls_emb = hatebert_emb[:, 0, :]                       # (B, H)

        # =====================================
        # Rationale text through additional model
        # =====================================
        if any(param.requires_grad for param in self.additional_model.parameters()):
            add_out = self.additional_model(
                input_ids=additional_input_ids,
                attention_mask=additional_attention_mask,
                return_dict=True
            )
        else:
            with torch.no_grad():
                add_out = self.additional_model(
                    input_ids=additional_input_ids,
                    attention_mask=additional_attention_mask,
                    return_dict=True
                )

        rationale_emb = add_out.last_hidden_state            # (B, L, H)

        # =========================
        # Selector / rationale pooling
        # =========================
        selector_logits = self.selector(hatebert_emb).squeeze(-1)   # (B, L)
        rationale_probs = self.gumbel_sigmoid_sample(selector_logits)
        rationale_probs = rationale_probs * attention_mask.float().to(rationale_probs.device)

        masked_hidden = hatebert_emb * rationale_probs.unsqueeze(-1)
        denom = rationale_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled_rationale = masked_hidden.sum(dim=1) / denom          # (B, H)

        # =========================
        # CNN feature branches
        # =========================
        temporal_features = self.temporal_cnn(hatebert_emb, attention_mask)               # (B, temporal_out_dim)
        rationale_features = self.msa_cnn(rationale_emb, additional_attention_mask)        # (B, msa_out_dim)

        # =========================
        # Branch-specific logits
        # =========================
        logits_cls = self.cls_head(cls_emb)
        logits_rationale = self.rationale_head(pooled_rationale)
        logits_temporal = self.temporal_head(temporal_features)
        logits_msa = self.msa_head(rationale_features)

        # =========================
        # Weighted probability averaging
        # =========================
        probs_cls = F.softmax(logits_cls, dim=1)
        probs_rationale = F.softmax(logits_rationale, dim=1)
        probs_temporal = F.softmax(logits_temporal, dim=1)
        probs_msa = F.softmax(logits_msa, dim=1)

        weights = F.softmax(self.branch_weights, dim=0)  # shape: (4,)

        final_probs = (
            weights[0] * probs_cls +
            weights[1] * probs_rationale +
            weights[2] * probs_temporal +
            weights[3] * probs_msa
        )

        logits = torch.log(final_probs.clamp_min(1e-9))

        attns = hatebert_out.attentions if (return_attentions and hasattr(hatebert_out, "attentions")) else None

        return logits, rationale_probs, selector_logits, attns


class ProjectionMLPBase(nn.Module):
    def __init__(self, input_size, output_size):
        super(ProjectionMLPBase, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, 2)
        )

    def forward(self, x):
        return self.layers(x)
    
class BaseShield(nn.Module):
    def __init__(self, hatebert_model, additional_model, projection_mlp, device='cpu', freeze_additional_model=True):
        super().__init__()
        self.hatebert_model = hatebert_model
        self.additional_model = additional_model
        self.projection_mlp = projection_mlp
        self.device = device

        if freeze_additional_model:
            for param in self.additional_model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, additional_input_ids, additional_attention_mask):
        hatebert_outputs = self.hatebert_model(input_ids=input_ids, attention_mask=attention_mask)
        hatebert_embeddings = hatebert_outputs.last_hidden_state[:, 0, :]

        additional_outputs = self.additional_model(input_ids=additional_input_ids, attention_mask=additional_attention_mask)
        additional_embeddings = additional_outputs.last_hidden_state[:, 0, :]

        concatenated_embeddings = torch.cat((hatebert_embeddings, additional_embeddings), dim=1)
        logits = self.projection_mlp(concatenated_embeddings)
        return logits
    

    
def load_model_from_hf(model_type="altered"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if model_type.lower() == "altered":
        model_path = os.path.join(base_dir, "models", "modified", "ModifiedModel.pth")
    elif model_type.lower() == "base":
        model_path = os.path.join(base_dir, "models", "base", "BaseShield.pth")
    else:
        raise ValueError("model_type must be 'base' or 'altered'")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    hatebert_name = "GroNLP/hateBERT"
    rationale_name = "bert-base-uncased"

    hatebert_model = AutoModel.from_pretrained(hatebert_name)
    rationale_model = AutoModel.from_pretrained(rationale_name)

    tokenizer_hatebert = AutoTokenizer.from_pretrained(hatebert_name)
    tokenizer_rationale = AutoTokenizer.from_pretrained(rationale_name)

    temporal_keys = [k for k in state_dict if k.startswith("temporal_cnn.convs")]
    is_altered = len(temporal_keys) > 0

    if not is_altered or model_type.lower() == "base":

        # ✅ dynamic input size (safer)
        input_size = 768 * 2

        projection_mlp = ProjectionMLPBase(
            input_size=input_size,
            output_size=512
        )

        model = BaseShield(
            hatebert_model=hatebert_model,
            additional_model=rationale_model,
            projection_mlp=projection_mlp,
            freeze_additional_model=True,
            device=device
        )

    else:
        conv_weights = [
            v for k, v in state_dict.items()
            if k.startswith("temporal_cnn.convs") and k.endswith("weight")
        ]

        cnn_num_filters = conv_weights[0].shape[0]
        cnn_kernel_sizes = tuple(w.shape[2] for w in conv_weights)
        cnn_dropout = 0.3

        # ✅ FIXED: added missing params
        model = ConcatModelWithRationale(
            hatebert_model=hatebert_model,
            additional_model=rationale_model,
            hidden_size=768,
            gumbel_temp=0.5,
            freeze_additional_model=True,
            cnn_num_filters=cnn_num_filters,
            cnn_kernel_sizes=cnn_kernel_sizes,
            cnn_dropout=cnn_dropout,
            num_classes=2
        )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)   # ✅ FIXED
    model.eval()

    config = {"max_length": 128}

    return model, tokenizer_hatebert, tokenizer_rationale, config, device

def predict_text(
    text,
    rationale,
    model,
    tokenizer_hatebert,
    tokenizer_rationale,
    device="cpu",
    max_length=128,
    model_type="altered"
):

    model.eval()

    # Convert to string and handle None/NaN values
    text = str(text) if text is not None else ""
    rationale = str(rationale) if rationale is not None else ""

    main_inputs = tokenizer_hatebert(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    rationale_inputs = tokenizer_rationale(
        rationale if rationale else text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = main_inputs["input_ids"].to(device)
    attention_mask = main_inputs["attention_mask"].to(device)

    add_input_ids = rationale_inputs["input_ids"].to(device)
    add_attention_mask = rationale_inputs["attention_mask"].to(device)

    tokens = tokenizer_hatebert.convert_ids_to_tokens(input_ids[0])
    with torch.no_grad():

        if model_type.lower() == "base":
            logits = model(
                input_ids,
                attention_mask,
                add_input_ids,
                add_attention_mask
            )
            rationale_scores = None

        else:
            outputs = model(
                input_ids,
                attention_mask,
                add_input_ids,
                add_attention_mask
            )

            if isinstance(outputs, tuple) and len(outputs) == 4:
                logits, rationale_probs, _, _ = outputs
                rationale_scores = rationale_probs[0].cpu().numpy()
            else:
                raise ValueError(f"Unexpected number of outputs from model: {len(outputs)}")

        probs = F.softmax(logits, dim=1)

        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones_like(logits) / logits.size(1)

        prediction = logits.argmax(dim=1).item()
        confidence = probs[0, prediction].item()
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probs[0].cpu().numpy(),
        "tokens": tokens,
        "rationale_scores": rationale_scores
    }

def predict_hatespeech_from_file(
    text_list,
    rationale_list,
    true_label,
    model,
    tokenizer_hatebert,
    tokenizer_rationale,
    config,
    device,
    model_type="altered"
):

    print(f"\nStarting inference for model: {type(model).__name__}")

    predictions = []
    all_probs = []
    cpu_percent_list = []
    memory_percent_list = []

    process = psutil.Process(os.getpid())

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # warmup
    with torch.no_grad():
        _ = predict_text(
            text=text_list[0],
            rationale=rationale_list[0],
            model=model,
            tokenizer_hatebert=tokenizer_hatebert,
            tokenizer_rationale=tokenizer_rationale,
            device=device,
            max_length=config.get('max_length', 128),
            model_type=model_type
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time()

    for idx, (text, rationale) in enumerate(zip(text_list, rationale_list)):

        result = predict_text(
            text=text,
            rationale=rationale,
            model=model,
            tokenizer_hatebert=tokenizer_hatebert,
            tokenizer_rationale=tokenizer_rationale,
            device=device,
            max_length=config.get('max_length', 128),
            model_type=model_type
        )

        predictions.append(result['prediction'])
        all_probs.append(result['probabilities'])

        if idx % 10 == 0 or idx == len(text_list) - 1:
            cpu_percent_list.append(process.cpu_percent())
            memory_percent_list.append(process.memory_info().rss / 1024 / 1024)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    runtime = time() - start_time

    print(f"Inference completed for {type(model).__name__}")
    print(f"Total runtime: {runtime:.4f} seconds")

    all_probs = np.array(all_probs)

    f1 = f1_score(true_label, predictions, zero_division=0)
    accuracy = accuracy_score(true_label, predictions)
    precision = precision_score(true_label, predictions, zero_division=0)
    recall = recall_score(true_label, predictions, zero_division=0)
    cm = confusion_matrix(true_label, predictions).tolist()

    avg_cpu = sum(cpu_percent_list) / len(cpu_percent_list) if cpu_percent_list else 0
    avg_memory = sum(memory_percent_list) / len(memory_percent_list) if memory_percent_list else 0
    peak_memory = max(memory_percent_list) if memory_percent_list else 0
    peak_cpu = max(cpu_percent_list) if cpu_percent_list else 0

    return {
        'model_name': type(model).__name__,
        'f1_score': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'cpu_usage': avg_cpu,
        'memory_usage': avg_memory,
        'peak_cpu_usage': peak_cpu,
        'peak_memory_usage': peak_memory,
        'runtime': runtime,
        'all_probabilities': all_probs.tolist()
    }

def predict_hatespeech_from_file_batched(
    text_list,
    rationale_list,
    true_label,
    model,
    tokenizer_hatebert,
    tokenizer_rationale,
    config,
    device,
    model_type="altered",
    batch_size=16
):

    print(f"\nStarting batched inference for model: {type(model).__name__}")

    predictions = []
    all_probs = []
    cpu_percent_list = []
    memory_percent_list = []

    process = psutil.Process(os.getpid())
    max_length = config.get('max_length', 128)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # warmup
    with torch.no_grad():
        _ = predict_text(
            text=text_list[0],
            rationale=rationale_list[0],
            model=model,
            tokenizer_hatebert=tokenizer_hatebert,
            tokenizer_rationale=tokenizer_rationale,
            device=device,
            max_length=max_length,
            model_type=model_type
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time()

    # Process in batches
    for batch_start in range(0, len(text_list), batch_size):
        batch_end = min(batch_start + batch_size, len(text_list))
        batch_texts = text_list[batch_start:batch_end]
        batch_rationales = rationale_list[batch_start:batch_end]

        # Convert to strings and handle None/NaN values
        batch_texts = [str(t) if t is not None else "" for t in batch_texts]
        batch_rationales = [str(r) if r is not None else "" for r in batch_rationales]

        # Tokenize batch
        main_batch_inputs = tokenizer_hatebert(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        rationale_batch_inputs = tokenizer_rationale(
            [r if r else t for r, t in zip(batch_rationales, batch_texts)],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Move to device
        batch_input_ids = main_batch_inputs["input_ids"].to(device)
        batch_attention_mask = main_batch_inputs["attention_mask"].to(device)
        batch_add_input_ids = rationale_batch_inputs["input_ids"].to(device)
        batch_add_attention_mask = rationale_batch_inputs["attention_mask"].to(device)

        with torch.no_grad():
            if model_type.lower() == "base":
                batch_logits = model(
                    batch_input_ids,
                    batch_attention_mask,
                    batch_add_input_ids,
                    batch_add_attention_mask
                )
                batch_rationale_probs = None
            else:
                batch_outputs = model(
                    batch_input_ids,
                    batch_attention_mask,
                    batch_add_input_ids,
                    batch_add_attention_mask
                )

                if isinstance(batch_outputs, tuple) and len(batch_outputs) == 4:
                    batch_logits, batch_rationale_probs, _, _ = batch_outputs
                else:
                    raise ValueError(f"Unexpected number of outputs from model: {len(batch_outputs)}")

            batch_probs = F.softmax(batch_logits, dim=1)

            if torch.isnan(batch_probs).any() or torch.isinf(batch_probs).any():
                batch_probs = torch.ones_like(batch_logits) / batch_logits.size(1)

            batch_predictions = batch_logits.argmax(dim=1).cpu().numpy()
            batch_probabilities = batch_probs.cpu().numpy()

        # Collect batch results
        predictions.extend(batch_predictions.tolist())
        all_probs.extend(batch_probabilities.tolist())

        # Log metrics periodically
        if batch_end % max(10, batch_size) == 0 or batch_end == len(text_list):
            cpu_percent_list.append(process.cpu_percent())
            memory_percent_list.append(process.memory_info().rss / 1024 / 1024)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    runtime = time() - start_time

    print(f"Batched inference completed for {type(model).__name__}")
    print(f"Total runtime: {runtime:.4f} seconds")

    all_probs = np.array(all_probs)

    f1 = f1_score(true_label, predictions, zero_division=0)
    accuracy = accuracy_score(true_label, predictions)
    precision = precision_score(true_label, predictions, zero_division=0)
    recall = recall_score(true_label, predictions, zero_division=0)
    cm = confusion_matrix(true_label, predictions).tolist()

    avg_cpu = sum(cpu_percent_list) / len(cpu_percent_list) if cpu_percent_list else 0
    avg_memory = sum(memory_percent_list) / len(memory_percent_list) if memory_percent_list else 0
    peak_memory = max(memory_percent_list) if memory_percent_list else 0
    peak_cpu = max(cpu_percent_list) if cpu_percent_list else 0

    return {
        'model_name': type(model).__name__,
        'f1_score': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'cpu_usage': avg_cpu,
        'memory_usage': avg_memory,
        'peak_cpu_usage': peak_cpu,
        'peak_memory_usage': peak_memory,
        'runtime': runtime,
        'all_probabilities': all_probs.tolist()
    }
    
def predict_hatespeech(text, rationale, model, tokenizer_hatebert, tokenizer_rationale, config, device, model_type="altered"):

    return predict_text(
        text=text,
        rationale=rationale,
        model=model,
        tokenizer_hatebert=tokenizer_hatebert,
        tokenizer_rationale=tokenizer_rationale,
        device=device,
        max_length=config.get('max_length', 128),
        model_type=model_type
    )