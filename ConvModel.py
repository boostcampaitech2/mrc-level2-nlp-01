import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers import  BertForQuestionAnswering, RobertaForQuestionAnswering, RobertaModel, BertModel

class ConvModel(RobertaForQuestionAnswering):
    def __init__(self, model_config):
        super().__init__() 
        
        self.roberta = RobertaModel(config=model_config)
        self.conv1d_layer1 = nn.Conv1d(model_config.hidden_size, 1024, kernel_size=1)
        self.conv1d_layer3 = nn.Conv1d(model_config.hidden_size, 1024, kernel_size=3, padding=1)
        self.conv1d_layer5 = nn.Conv1d(model_config.hidden_size, 1024, kernel_size=5, padding=2)
        self.drop_out = nn.Dropout(0.3)
        self.classify_layer = nn.Linear(1024*3, 2, bias=True)
        self.init_weights()
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None, 
                start_positions=None, end_positions=None, output_attentions=None, 
                output_hidden_states=None, return_dict=None):
        
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
        
        sequence_output = outputs[0] # Convolution 연산을 위해 Transpose (B * hidden_size * max_seq_legth)
        conv_input = sequence_output.transpose(1, 2) # Conv 연산을 위한 Transpose (B * hidden_size * max_seq_length)
        conv_output1 = F.relu(self.conv1d_layer1(conv_input)) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output3 = F.relu(self.conv1d_layer3(conv_input)) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output5 = F.relu(self.conv1d_layer5(conv_input)) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        concat_output = torch.cat((conv_output1, conv_output3, conv_output5), dim=1) # Concatenation (B * num_conv_filter x 3 * max_seq_legth)
        
        concat_output = concat_output.transpose(1, 2)
        concat_output = self.drop_out(concat_output) # dropout 통과
        
        logits = self.classify_layer(concat_output) # Classifier Layer를 통해 최종 Logit을 얻음. (B * max_seq_legth * 2)
        
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
            
        return QuestionAnsweringModelOutput(
                loss=total_loss,
                start_logits=start_logits,
                end_logits=end_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )