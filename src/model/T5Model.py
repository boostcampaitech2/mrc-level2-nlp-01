import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer, MT5EncoderModel, T5EncoderModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
# from transformer.modeling_utils import PoolerAnswerClass
from torch.nn import CrossEntropyLoss

class MT5EncoderQuestionAnsweringModel(T5EncoderModel):
    def __init__(self, model_name, model_config):
        super().__init__(model_config)
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = 2
        self.encoder = MT5EncoderModel.from_pretrained(model_name, config=self.config)
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)
        self.encoder._init_weights(self.qa_outputs)


    def forward(self,
                input_ids=None,
                attention_mask=None,
                head_mask=None,
                cross_attn_head_mask=None,
                encoder_outputs=None,
                start_positions=None,
                end_positions=None,
                inputs_embeds=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = self.qa_outputs(outputs.last_hidden_state)
        # print(outputs)
        # print(outputs.last_hidden_state.shape)
        # print(logits)
        # bs, sequence_length, 2
        start_logits, end_logits = logits.split(1, dim=-1)
        # print(start_logits, end_logits)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1) # ??
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index) #

            start_logits = start_logits.unsqueeze(-1)
            start_positions = start_positions.unsqueeze(-1)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (
                start_logits,
                end_logits,
            ) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
