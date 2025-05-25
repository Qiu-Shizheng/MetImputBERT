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
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        past_key_value = (key_layer, value_layer) if self.is_decoder else None
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class CustomBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = CustomBertSelfAttention(config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions=output_attentions)
        attention_output = self.output(self_outputs[0])
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)
        return (attention_output,) + self_outputs[1:]

class CustomBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CustomBertAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = CustomBertAttention(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions=output_attentions)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = F.relu(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.LayerNorm(layer_output + attention_output)
        return (layer_output,) + outputs

class CustomBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([CustomBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i] if head_mask is not None else None, encoder_hidden_states, encoder_attention_mask, past_key_values[i] if past_key_values is not None else None, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_attentions] if v is not None)
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': next_decoder_cache,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
        }

class CustomBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = CustomBertEncoder(config)
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.config = config
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        return (1.0 - extended_attention_mask) * -10000.0

    def get_head_mask(self, head_mask, num_hidden_layers):
        if head_mask is not None:
            head_mask = torch.tensor(head_mask, dtype=torch.float32)
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).expand(num_hidden_layers, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1)
            return head_mask
        return [None] * num_hidden_layers

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=True):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot pass both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        batch_size, seq_length = input_shape[0], input_shape[1]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device if inputs_embeds is not None else next(self.parameters()).device)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device if inputs_embeds is not None else next(self.parameters()).device)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, next(self.parameters()).device)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = inputs_embeds
        encoder_outputs = self.encoder(hidden_states=embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True)
        sequence_output = encoder_outputs['last_hidden_state']
        pooled_output = self.pooler(sequence_output.transpose(1, 2)).squeeze(-1) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + tuple(v for v in [encoder_outputs['past_key_values'], encoder_outputs['hidden_states'], encoder_outputs['attentions']] if v is not None)
        return {
            'last_hidden_state': sequence_output,
            'pooler_output': pooled_output,
            'past_key_values': encoder_outputs['past_key_values'],
            'hidden_states': encoder_outputs['hidden_states'],
            'attentions': encoder_outputs['attentions'],
        }

class MetaboliteBERTModel(nn.Module):
    def __init__(self, hidden_size, num_layers, num_metabolites):
        super().__init__()
        config = BertConfig(
            vocab_size=num_metabolites,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=8,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=num_metabolites + 1,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            is_decoder=False,
        )
        self.bert = CustomBertModel(config)
        self.expr_weight = nn.Parameter(torch.ones(1, num_metabolites, hidden_size))
        self.expr_bias = nn.Parameter(torch.zeros(1, num_metabolites, hidden_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, expressions, attention_mask):
        batch_size, num_metabolites = expressions.size()
        expr_embeds = expressions.unsqueeze(-1) * self.expr_weight + self.expr_bias
        cls_tokens = self.cls_token.expand(batch_size, 1, -1)
        embeddings = torch.cat((cls_tokens, expr_embeds), dim=1)
        new_attention_mask = torch.cat((torch.ones(batch_size, 1, device=attention_mask.device), attention_mask), dim=1)
        outputs = self.bert(inputs_embeds=embeddings, attention_mask=new_attention_mask, output_attentions=False)
        return self.output_layer(outputs['last_hidden_state'][:, 1:, :]).squeeze(-1)