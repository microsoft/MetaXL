import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertForTokenClassification, BertPreTrainedModel, XLMRobertaTokenizer, XLMRobertaForTokenClassification
from transformers.modeling_bert import BertLayer, BertModel, BertEmbeddings, BertEncoder, BertPooler
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from torch.nn import CrossEntropyLoss, MSELoss


IGNORED_INDEX = -100

class BERTSequenceTokenizer():
    def __init__(self, bert_name, max_len=512, cache_dir='cache', tokenizer_dir=None):
        #from pytorch_transformers import BertTokenizer
        self.CLS = '[CLS]'
        self.SEP = '[SEP]'
        self.max_len = max_len

        tok = XLMRobertaTokenizer if bert_name.startswith("xlm") else BertTokenizer
        if tokenizer_dir is None or tokenizer_dir == "None":
            self.tokenizer = tok.from_pretrained(bert_name, cache_dir=cache_dir)
        else:
            self.tokenizer = tok.from_pretrained(os.path.join(tokenizer_dir, 'vocab-vocab.txt'))
            # self.tokenizer = ByteLevelBPETokenizer(
            #         vocab_file=os.path.join(tokenizer_dir, "vocab.json"),
            #         merges_file=os.path.join(tokenizer_dir, "merges.txt"))
            # self.cls_id = self.tokenizer.token_to_id(self.CLS)
            # self.sep_id = self.tokenizer.token_to_id(self.SEP)

        self.cls_id = self.tokenizer.convert_tokens_to_ids(self.CLS)
        self.sep_id = self.tokenizer.convert_tokens_to_ids(self.SEP)

    def encode(self, token_list, label_list=None):
        if type(label_list) == list:
            assert len(token_list) == len(label_list), 'Mismatch text and label length!'
            n_tokens = len(token_list)

            ids = [self.cls_id]
            labels = [IGNORED_INDEX]

            for i, token in enumerate(token_list):
                subword_ids = self.tokenizer.encode(token, add_special_tokens=False)  # add_special_tokens has to be FALSE here
                if len(subword_ids) == 0: # some instance in wikiann is empty but has ner tags
                    subword_ids = [self.tokenizer.convert_tokens_to_ids('[OOV]')]
                    print('Emtpy subwords for |%s|, Token tag: %s, replaced with [OOV]' % (token, label_list[i]))
                ids = ids + subword_ids
                labels.append(label_list[i])

                # for further subwords append IGNORED_INDEX
                labels = labels + [IGNORED_INDEX] * (len(subword_ids) -1)

            ids.append(self.sep_id)
            labels.append(IGNORED_INDEX)
        else:
            ids = [self.cls_id] + self.tokenizer.encode(token_list, add_special_tokens=False) + [self.sep_id]
            labels = label_list

        '''
        print ('========================================== ')
        print ('TOK:', token_list, len(token_list))
        print ('LAB:', label_list, len(label_list))
        print ('SUB:', self.tokenizer.convert_ids_to_tokens(ids), len(ids))
        print ('IDS:', ids, len(ids))
        print ('TAG:', labels, len(labels))
        print ()
        '''

        if type(label_list) == list:
            assert len(ids) == len(labels), 'Wrong subword tokenization!'

        x_len = len(ids)
        if x_len > self.max_len:
            ids = ids[:self.max_len]
            mask = [1] * self.max_len
            if type(label_list) == list:
                labels = labels[:self.max_len]
            print ('Excessively long sequence, trimmed down!')
        else:
            ids = ids + [0] * (self.max_len - x_len)
            mask = [1] * x_len + [0] * (self.max_len - x_len)
            if type(label_list) == list:
                labels = labels + [IGNORED_INDEX] * (self.max_len - x_len)

        return ids, mask, labels

    def encode2(self, text, label_list):
        tokens = self.tokenizer.tokenize(text)
        tokens =  [self.CLS] + tokens + [self.SEP] # add special tokens
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        x_len = len(ids)

        print (x_len)
        print (text)
        print (tokens)
        print (label_list)

        ids = ids + [0] * (self.max_len - x_len)
        mask = [1] * x_len + [0] * (self.max_len - x_len)
        
        labels = [IGNORED_INDEX] * self.max_len # ignored for all and then set labels for actual tokens
        idx = 0

        for i, token in enumerate(tokens):
            if token not in [self.CLS, self.SEP] and not token.startswith('##'): # actual token (or prefix)
                labels[i] = label_list[idx]
                idx += 1

        return ids, mask, labels


class BERTSequenceTagger(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = SplitBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()


    def forward_embedding_head(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,):
        input_embeds, extended_attention_mask, head_mask, encoder_hidden_states, encoder_extended_attention_mask \
            = self.bert.forwardbertembeddings(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds)
        return input_embeds, extended_attention_mask, head_mask, encoder_hidden_states, encoder_extended_attention_mask

    def forward_embedding_tail(self, input_embeds,
                     extended_attention_mask,
                     head_mask,
                     encoder_hidden_states,
                     encoder_extended_attention_mask,
                     input_ids=None,
                     attention_mask=None,
                     token_type_ids=None,
                     position_ids=None,
                     labels=None,):
        outputs = self.bert.forwardberttail(input_embeds, extended_attention_mask, head_mask, encoder_hidden_states, encoder_extended_attention_mask, input_ids, position_ids, token_type_ids)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        for_classification = False
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.forward_classifier(sequence_output, for_classification)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

    def get_ext_mask(self, attention_mask):
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    # note attention_mask here should be the extended attention mask constructed from forward_head
    def forward_tail(self, k, x, attention_mask=None):
        assert k>0 and k<= 1+self.config.num_hidden_layers, 'Wrong layer index!'
        
        hidden_states = x
        for i, layer_module in enumerate(self.bert.encoder.layer[k-1:]):
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]

        sequence_output = hidden_states
        return sequence_output

    def forward_pooler(self, sequence_output):
        return self.bert.pooler(sequence_output)

    def forward_classifier(self, sequence_output, for_classification=False):
        pooler_output = self.dropout(self.forward_pooler(sequence_output))
        sequence_output = self.dropout(sequence_output)
        if for_classification:
            logits = self.classifier(pooler_output)
        else:
            logits = self.classifier(sequence_output)
        return logits


class XLMRSequenceTagger(XLMRobertaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = SplitRoberta(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward_embedding_head(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None, ):
        input_embeds, extended_attention_mask, head_mask, encoder_hidden_states, encoder_extended_attention_mask \
            = self.roberta.forwardbertembeddings(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              token_type_ids=token_type_ids,
                                              position_ids=position_ids,
                                              head_mask=head_mask,
                                              inputs_embeds=inputs_embeds)
        return input_embeds, extended_attention_mask, head_mask, encoder_hidden_states, encoder_extended_attention_mask

    def forward_embedding_tail(self, input_embeds,
                               extended_attention_mask,
                               head_mask,
                               encoder_hidden_states,
                               encoder_extended_attention_mask,
                               input_ids=None,
                               attention_mask=None,
                               token_type_ids=None,
                               position_ids=None,
                               labels=None, ):
        outputs = self.roberta.forwardberttail(input_embeds, extended_attention_mask, head_mask, encoder_hidden_states,
                                            encoder_extended_attention_mask, input_ids, position_ids, token_type_ids)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            for_classification=False
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.forward_classifier(sequence_output, for_classification)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

    def get_ext_mask(self, attention_mask):
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    # note attention_mask here should be the extended attention mask constructed from forward_head
    def forward_tail(self, k, x, attention_mask=None):
        assert k > 0 and k <= 1 + self.config.num_hidden_layers, 'Wrong layer index!'

        hidden_states = x
        for i, layer_module in enumerate(self.roberta.encoder.layer[k - 1:]):
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]

        sequence_output = hidden_states
        return sequence_output

    def forward_pooler(self, sequence_output):
        return self.roberta.pooler(sequence_output)

    def forward_classifier(self, sequence_output, for_classification=False):
        pooler_output = self.dropout(self.forward_pooler(sequence_output))
        sequence_output = self.dropout(sequence_output)
        if for_classification:
            logits = self.classifier(pooler_output)
        else:
            logits = self.classifier(sequence_output)
        return logits


class Raptors(nn.Module):
    def __init__(self, config, num_layers=1, num_langs=1, struct="transformer", add_weights=False, tied=True, bottle_size=768):
        super().__init__()

        self.nets = []
        self.num_layers = num_layers
        self.num_langs = num_langs
        self.struct = struct
        self.add_weights = add_weights
        self.tied = tied
        for i in range(num_langs):
            for j in range(num_layers):
                if struct == "transformer":
                    self.nets.append(BertLayer(config))
                elif struct == "perceptron":
                    hidden_size = config.hidden_size
                    if add_weights:
                        if tied:
                            self.nets.append(nn.Sequential(
                                             nn.Linear(hidden_size, bottle_size),
                                             nn.ReLU(),
                                             nn.Linear(bottle_size, hidden_size + 1)))
                        else:
                            self.nets.append(nn.Sequential(
                                nn.Linear(hidden_size, bottle_size),
                                nn.ReLU(),
                                nn.Linear(bottle_size, hidden_size)))
                            self.weight_net = nn.Sequential(
                                nn.Linear(hidden_size, bottle_size),
                                nn.ReLU(),
                                nn.Linear(bottle_size, 1)
                            )
                    else:
                        self.nets.append(nn.Sequential(
                            nn.Linear(hidden_size, hidden_size // 4),
                            nn.ReLU(),
                            nn.Linear(hidden_size // 4, hidden_size)))
                else:
                    print("The specified structure is not implemented.")
                    sys.exit(0)

        self.nets = nn.ModuleList(self.nets)
        self.alpha = nn.Parameter(torch.zeros(num_langs, num_layers))

        if struct == "perceptron":
            self.init_weights()

    def init_weights(self):
        for i in range(len(self.nets)):
            nn.init.xavier_normal_(self.nets[i][0].weight)
            self.nets[i][0].bias.data.zero_()
            nn.init.xavier_normal_(self.nets[i][2].weight)
            self.nets[i][2].bias.data.zero_()
            # nn.init.xavier_normal_(self.nets[i][4].weight)
            # self.nets[i][4].bias.data.zero_()
    # i: lang id j: layer id
    def forward(self, i, j, x):
        ind = i * self.num_layers + j
        if self.struct == "transformer":
            return self.nets[ind](x)[0]
        elif self.struct == "perceptron":
            out = self.nets[ind](x)
            if self.add_weights:
                if self.tied:
                    rep = out[:, :, :-1]
                    weight = F.sigmoid(out[:, :, -1]).unsqueeze(-1)
                else:
                    rep = out
                    weight = F.sigmoid(self.weight_net(x))
                    print(weight)
                out = weight * rep
            return out

    def get_alpha(self, i):
        return F.softmax(self.alpha[i], -1)


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    featurized_sentences = []
    for idx, example in enumerate(examples):
        features = {}
        features['bert_ids'], features['bert_mask'], features['ber_token_starts'] = tokenizer.subword_tokenize_to_ids(example.text_a)
        features['label'] = label_list
        featurized_sentences.append(features)


def trim_input(bert_ids, bert_mask, bert_labels=None, train_max=None):
    max_length = (bert_mask !=0).max(0)[0].nonzero().numel()
    if train_max is not None:
        max_length = min(max_length, train_max)
    
    if max_length < bert_ids.shape[1]:
        bert_ids = bert_ids[:, :max_length]
        bert_mask = bert_mask[:, :max_length]
        if bert_labels is not None and bert_labels.ndim == 2:
            bert_labels = bert_labels[:, :max_length]

    if bert_labels is not None:
        return bert_ids, bert_mask, bert_labels
    else:
        return bert_ids, bert_mask


def masked_cross_entropy(logit, labels, K):
    loss_sum = F.cross_entropy(logit.view(-1, K),
                               labels.view(-1),
                               ignore_index=IGNORED_INDEX,
                               reduction='sum')
    loss = loss_sum / (labels!=IGNORED_INDEX).sum()
    return loss


class WNets(nn.Module):
    def __init__(self, h_dim, n_lang):
        super().__init__()
        nets = []
        for _ in range(n_lang):
            nets.append(nn.Sequential(
                nn.Linear(1, h_dim),
                nn.ReLU(inplace=True),
                nn.Linear(h_dim, 1)
            ))

        self.nets = nn.ModuleList(nets)
        self.init_weights()

    def init_weights(self):
        for i in range(len(self.nets)):
            nn.init.xavier_normal_(self.nets[i][0].weight)
            self.nets[i][0].bias.data.zero_()
            nn.init.xavier_normal_(self.nets[i][2].weight)
            self.nets[i][2].bias.data.zero_()

    def forward(self, i, x):
        return torch.sigmoid(self.nets[i](x))
        

class VNet(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(VNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            #nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            #nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(h_dim, out_dim)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))

class SplitBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = SplitBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forwardbertembeddings(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        inputs_embeds = self.embeddings.forward_head(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        embedding_output = self.embeddings.forward_tail(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)

        return embedding_output, extended_attention_mask, head_mask, encoder_hidden_states, encoder_extended_attention_mask

    def forwardberttail(self, embedding_output, extended_attention_mask, head_mask, encoder_hidden_states, encoder_extended_attention_mask,
                        input_ids, position_ids, token_type_ids):
        # embedding_output = self.embeddings.forward_tail(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class SplitRoberta(SplitBertModel):
    def __init__(self, config):
        super().__init__(config)

class SplitBertEmbeddings(BertEmbeddings):
    def forward_head(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        return inputs_embeds

    def forward_tail(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


