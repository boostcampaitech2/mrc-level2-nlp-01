import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers import RobertaForQuestionAnswering, RobertaModel, RobertaPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaLMHead


class SpanMaskingModel(RobertaPreTrainedModel):
    def __init__(self, model_config):
        super().__init__(model_config)
        
        self.roberta = RobertaModel(config=model_config)
        self.lm_head = RobertaLMHead(model_config)

        self.lstm = nn.LSTM(input_size = model_config.hidden_size,
            hidden_size=model_config.hidden_size,
            num_layers=3,
            dropout=0.3,
            bidirectional=True,
            batch_first=True,
        )
        self.pooler = nn.Sequential(
            nn.Linear(model_config.hidden_size * 2, 200),
            nn.Linear(200, 2)
        )

        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None, 
        position_ids=None, head_mask=None, inputs_embeds=None, start_positions=None, 
        end_positions=None, output_attentions=None, output_hidden_states=None, 
        return_dict=None, labels=None):
       
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sentence_output = outputs[0]
        
        logit = self.lstm(sentence_output)
        logits = self.pooler(logit[0]) # batch, seq, 2
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if labels is not None:
                lm_head_outputs = self.lm_head(sentence_output) # batch, seq, vocab
                mlm_loss = self.loss_func(lm_head_outputs.view(-1, lm_head_outputs.size(-1)), labels.view(-1))
                total_loss += mlm_loss

        # if not return_dict:
        #     output = (start_logits, end_logits) + outputs[self.pooling_pos :]
        #     return ((total_loss,) + output) if total_loss is not None else output
        
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )