from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig


class CustomBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # x: [B, L, all_head]
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)          # [B, L, H, D]
        return x.permute(0, 2, 1, 3)    # [B, H, L, D]

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask  # additive mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)  # [B,H,L,D]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [B,L,H,D]
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)  # [B,L,all_head]

        if output_attentions:
            return context_layer, attention_probs
        return (context_layer,)


class CustomBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = CustomBertSelfAttention(config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attn_output = self.output(self_outputs[0])
        attn_output = self.dropout(attn_output)
        attn_output = self.LayerNorm(attn_output + hidden_states)
        return (attn_output,) + self_outputs[1:]


class CustomBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CustomBertAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        attn_out = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = attn_out[0]
        outputs = attn_out[1:]

        intermediate_output = F.relu(self.intermediate(attention_output))
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.LayerNorm(layer_output + attention_output)

        return (layer_output,) + outputs


class CustomBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([CustomBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        all_attentions = () if output_attentions else None

        for layer_module in self.layer:
            layer_out = layer_module(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
            hidden_states = layer_out[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_out[1],)

        return {"last_hidden_state": hidden_states, "attentions": all_attentions}


class CustomBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = CustomBertEncoder(config)

    @staticmethod
    def get_extended_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        """
        attention_mask: [B, L] with 1=keep, 0=mask
        return: [B, 1, 1, L] additive mask (0 keep, -10000 mask)
        """
        extended = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=torch.float32)
        return (1.0 - extended) * -10000.0

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, output_attentions=False):
        ext = self.get_extended_attention_mask(attention_mask)
        enc = self.encoder(hidden_states=inputs_embeds, attention_mask=ext, output_attentions=output_attentions)
        return enc


class MetaboliteBERTModel(nn.Module):
    
    def __init__(
        self,
        num_metabolites: int = 249,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        config = BertConfig(
            vocab_size=num_metabolites,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=num_metabolites + 1,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            is_decoder=False,
        )
        self.bert = CustomBertModel(config)

        self.expr_weight = nn.Parameter(torch.ones(1, num_metabolites, hidden_size))
        self.expr_bias = nn.Parameter(torch.zeros(1, num_metabolites, hidden_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, expressions_z: torch.Tensor, observed_mask: torch.Tensor, output_attentions: bool = False):
        """
        expressions_z: [B, M]
        observed_mask: [B, M] 1=observed, 0=missing
        """
        B, M = expressions_z.shape

        expr_embeds = expressions_z.unsqueeze(-1) * self.expr_weight + self.expr_bias  # [B,M,H]
        cls_tokens = self.cls_token.expand(B, 1, -1)  # [B,1,H]
        embeddings = torch.cat((cls_tokens, expr_embeds), dim=1)  # [B,M+1,H]

        attn_mask = torch.cat(
            (torch.ones(B, 1, device=observed_mask.device, dtype=observed_mask.dtype), observed_mask),
            dim=1,
        )  # [B,M+1]

        outputs = self.bert(inputs_embeds=embeddings, attention_mask=attn_mask, output_attentions=output_attentions)
        last_hidden = outputs["last_hidden_state"]  # [B,M+1,H]

        z_pred = self.output_layer(last_hidden[:, 1:, :]).squeeze(-1)  # [B,M]
        return z_pred, outputs.get("attentions", None)
