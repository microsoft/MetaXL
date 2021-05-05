import argparse
import json
import random
import conllu
from glob import glob

from models import *
from mlt import *
from utils import *
from data_utils import DataIterator
from transformers import ( BertConfig,
                            XLMRobertaConfig,
                          get_linear_schedule_with_warmup)
from torch import nn
from torch.utils.data import (DataLoader, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import accuracy_score as seq_accuracy_score
from seqeval.metrics import precision_score as seq_precision_score
from seqeval.metrics import recall_score as seq_recall_score

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_ids = label_ids

def readfile(filename, lang=None):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        line = line.strip()
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        #splits = line.split(' ')
        splits = line.strip().split('\t')
        token = splits[0]
        if lang is not None and token.startswith('%s:' % lang):
            token = token.split('%s:' % lang)[-1]
        
        sentence.append(token)
        label.append(splits[-1])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, lang=None):
        """Reads a tab separated value file."""
        return readfile(input_file, lang=lang)


class NerProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]#, "[CLS]", "[SEP]"]
        self.label_map = dict(zip(self.labels, range(len(self.labels))))
        
    
    """Processor for the CoNLL-2003 data set."""
    def get_train_examples(self, data_dir, lang):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "%s.train" % lang)), "train")

    def get_dev_examples(self, data_dir, lang):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "%s.dev" % lang)), "dev")

    def get_test_examples(self, data_dir, lang):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "%s.test" % lang)), "test")

    def get_labels(self):
        return self.labels

    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = None
            label = [self.label_map[l] for l in label]
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples


class POSProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["DET", "VERB", "SYM", "SCONJ", "CCONJ", "PUNCT", "NUM", "ADP", "NOUN", "_", "PRON",
                       "ADJ", "PART", "ADV", "PROPN", "INTJ", "X", "AUX"]
        self.label_map = dict(zip(self.labels, range(len(self.labels))))

    """Processor for the POS data set."""

    def read_conllu_file(self, file):
        data = conllu.parse(open(file, "r").read())
        sents = []
        for sentence in data:
            sent = []
            label = []
            for token in sentence:
                sent.append(token["form"])
                label.append(token["upostag"])
            sents.append((sent, label))
        return sents

    def get_train_examples(self, data_dir, lang):
        """See base class."""
        file = os.path.join(data_dir, f"UD_{lang}", "*train.conllu")
        file = glob(file)[0]
        conllu_data = self.read_conllu_file(file)
        return self._create_examples(conllu_data,  "train")

    def get_dev_examples(self, data_dir, lang):
        """See base class."""
        file = os.path.join(data_dir, f"UD_{lang}", "*dev.conllu")
        file = glob(file)[0]
        conllu_data = self.read_conllu_file(file)
        return self._create_examples(conllu_data, "dev")

    def get_test_examples(self, data_dir, lang):
        """See base class."""
        file = os.path.join(data_dir, f"UD_{lang}", "*test.conllu")
        file = glob(file)[0]
        conllu_data = self.read_conllu_file(file)
        return self._create_examples(conllu_data, "test")

    def get_labels(self):
        return self.labels

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = None
            label = [self.label_map[l] for l in label]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class SentClassProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["negative", "positive"]
        self.label_map = dict(zip(self.labels, range(len(self.labels))))

    """Processor for the POS data set."""

    def read_json_file(self, file):
        data = json.load(open(file, "r"))
        sents = []
        for ex in data:
            sent = ex["review_body"]
            if "label" in ex:
                label = ex["label"]
            else:
                stars = int(ex["stars"])
                if stars == 3:
                   continue
                elif stars > 3:
                    label = "positive"
                else:
                    label = "negative"
            sents.append((sent, label))
        return sents

    def get_train_examples(self, data_dir, lang):
        """See base class."""
        file = os.path.join(data_dir, f"{lang}", "*.train.json")
        file = glob(file)[0]
        sents = self.read_json_file(file)
        return self._create_examples(sents,  "train")

    def get_dev_examples(self, data_dir, lang):
        """See base class."""
        file = os.path.join(data_dir, f"{lang}", "*.dev.json")
        file = glob(file)[0]
        sents = self.read_json_file(file)
        return self._create_examples(sents, "dev")

    def get_test_examples(self, data_dir, lang):
        """See base class."""
        file = os.path.join(data_dir, f"{lang}", "*.test.json")
        file = glob(file)[0]
        sents = self.read_json_file(file)
        return self._create_examples(sents, "test")

    def get_labels(self):
        return self.labels

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = None
            label = self.label_map[label]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class PANXNerProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]#, "[CLS]", "[SEP]"]
        self.label_map = dict(zip(self.labels, range(len(self.labels))))
        
    
    """Processor for the PANX data set."""
    def get_train_examples(self, data_dir, lang):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, lang, "train"), lang=lang), "train")

    def get_dev_examples(self, data_dir, lang):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, lang, "dev"), lang=lang), "dev")

    def get_test_examples(self, data_dir, lang):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, lang, "test"), lang=lang), "test")

    def get_labels(self):
        return self.labels

    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = None
            label = [self.label_map[l] for l in label]
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    features = []
    for idx, example in enumerate(examples):
        input_ids, input_mask, label_ids = tokenizer.encode(example.text_a, example.label)
        features.append(
            InputFeatures(input_ids=input_ids[:max_seq_length],
                          input_mask=input_mask[:max_seq_length],
                          label_ids=label_ids[:max_seq_length] if type(label_ids) is list else label_ids))
    return features

# def eval(model, test_dataloader, processor):
#     all_y_true = []
#     all_y_pred = []
#     for idx, batch_test in enumerate(test_dataloader):
#         batch_test = tuple(t.cuda() for t in batch_test)
#         test_ids, test_mask, test_labels = batch_test
#         test_ids, test_mask, test_labels = trim_input(test_ids, test_mask, test_labels)
#
#         with torch.no_grad():
#             test_logit = model(test_ids, attention_mask=test_mask)[0]
#
#             pred_labels = test_logit.max(-1)[1]
#
#             y_true = [y[y!=IGNORED_INDEX].cpu().numpy().tolist() for y in test_labels]
#             y_tags_true = [[processor.labels[y] for y in y_group] for y_group in y_true]
#
#             y_pred = [pred[y!=IGNORED_INDEX].cpu().numpy().tolist() for (pred, y) in zip(pred_labels, test_labels)]
#             y_tags_pred = [[processor.labels[y] for y in y_group] for y_group in y_pred]
#             all_y_true.extend(y_tags_true)
#             all_y_pred.extend(y_tags_pred)
#
#     f1 = f1_score(all_y_true, all_y_pred)
#     acc = accuracy_score(all_y_true, all_y_pred)
#     precision = precision_score(all_y_true, all_y_pred)
#     recall = recall_score(all_y_true, all_y_pred)
#
#     return f1, acc, precision, recall

def eval(model, test_dataloader, processor, for_classification=False):
    all_y_true = []
    all_y_pred = []

    for idx, batch_test in enumerate(test_dataloader):
        batch_test = tuple(t.cuda() for t in batch_test)
        test_ids, test_mask, test_labels = batch_test
        test_ids, test_mask, test_labels = trim_input(test_ids, test_mask, test_labels)

        with torch.no_grad():
            test_logit = model(test_ids, attention_mask=test_mask, for_classification=for_classification)[0] # batch * sequence lens * labels

            pred_labels = test_logit.max(-1)[1]

            if for_classification:
                all_y_true.extend(list(torch.unsqueeze(test_labels, 1).cpu().numpy()))
                all_y_pred.extend(list(torch.unsqueeze(pred_labels, 1).cpu().numpy()))

            else:
                y_true = [y[y != IGNORED_INDEX].cpu().numpy().tolist() for y in test_labels]
                y_tags_true = [[processor.labels[y] for y in y_group] for y_group in y_true]

                y_pred = [pred[y != IGNORED_INDEX].cpu().numpy().tolist() for (pred, y) in zip(pred_labels, test_labels)]
                y_tags_pred = [[processor.labels[y] for y in y_group] for y_group in y_pred]
                all_y_true.extend(y_tags_true)
                all_y_pred.extend(y_tags_pred)

    if for_classification:
        f1 = f1_score(all_y_true, all_y_pred)
        acc = accuracy_score(all_y_true, all_y_pred)
        precision = precision_score(all_y_true, all_y_pred)
        recall = recall_score(all_y_true, all_y_pred)
    else:
        f1 = seq_f1_score(all_y_true, all_y_pred)
        acc = seq_accuracy_score(all_y_true, all_y_pred)
        precision = seq_precision_score(all_y_true, all_y_pred)
        recall = seq_recall_score(all_y_true, all_y_pred)

    return f1, acc, precision, recall

def read_data(data_dir, processor, tokenizer, lang, split, max_seq_length, model_name, bert_model_type="ori", train_size=-1, seed=42):
    pt_name = '%s/%s/%s_%s_%d' % (data_dir, lang, split, model_name, max_seq_length)
    if bert_model_type != "ori":
        pt_name += f"_{bert_model_type}"
    pt_name += ".pt"

    if os.path.isfile(pt_name):
        with open(pt_name, 'rb') as f:
            data = torch.load(f)

        logger.info("***** Loading CACHED data for %s *****" % lang)
    else:
        label_list = processor.get_labels()
        if split == 'train':
            examples = processor.get_train_examples(data_dir, lang)
        elif split == 'dev':
            examples = processor.get_dev_examples(data_dir, lang)
        elif split == 'test':
            examples = processor.get_test_examples(data_dir, lang)
        else:
            raise Exception('Wrong split %s!' % split)

        features = convert_examples_to_features(
            examples, label_list, max_seq_length, tokenizer)

        logger.info("***** Loading data for %s *****" % lang)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, label_ids)

        if not os.path.exists(os.path.dirname(pt_name)):
            os.makedirs(os.path.dirname(pt_name))

        with open(pt_name, 'wb') as f:
            torch.save(data, f)

    # subsample if
    if train_size > 0: # subsample
        N = len(data)
        # reseed again to guaranttee reproducibility
        np.random.seed(seed)
        if train_size < N:
            sampled_indices = np.random.choice(np.arange(0, N), train_size, replace=False)
        else:
            sampled_indices = np.arange(0, N)
        data_subset = TensorDataset(data.tensors[0][sampled_indices],
                             data.tensors[1][sampled_indices],
                             data.tensors[2][sampled_indices])
        data = data_subset

    logger.info("  Num %s examples = %d", split, len(data))

    return data

# create one merged dataset from multiple languages
def merge_data(data_dir, processor, tokenizer, langs, split, max_seq_length, bert_model, bert_model_type, train_size=-1, seed=1, rest_all=False, tgt_lang=None):
    if rest_all:
        assert tgt_lang is not None, 'Need to specify tgt_lang when rest_all is True!'
    data_list = []
    for lang in langs:
        if rest_all:
            if lang == tgt_lang:
                data = read_data(data_dir, processor, tokenizer, lang, split, max_seq_length, bert_model, bert_model_type, train_size, seed)
            else:
                data = read_data(data_dir, processor, tokenizer, lang, split, max_seq_length, bert_model, bert_model_type,  -1, seed) # take all for src_langs
        else:
            data = read_data(data_dir, processor, tokenizer, lang, split, max_seq_length, bert_model, bert_model_type, train_size, seed)
        data_list.append(data)

    merged_data = TensorDataset(torch.cat([x.tensors[0] for x in data_list], dim=0), # input_ids
                                torch.cat([x.tensors[1] for x in data_list], dim=0), # input_mask
                                torch.cat([x.tensors[2] for x in data_list], dim=0)) # label_ids

    return merged_data

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='data/panx_dataset',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='bert-base-multilingual-cased',
                        type=str,
                        #required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='panx',
                        type=str,
                        #required=True,
                        help="The name of the task to train.")
    parser.add_argument('--tgt_lang',
                        default='en',
                        type=str,
                        required=True,
                        help='Target language (default: en)')
    parser.add_argument("--output_dir",
                        default='out',
                        type=str,
                        #required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    # set a max seq length for training to save GPU-ram in training (testing not affected)
    parser.add_argument("--train_max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")
    parser.add_argument("--do_finetune",
                        action='store_true',
                        help="Whether to run eval or not.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--train_size',
                        default=-1,
                        type=int,
                        help='Training instances used for training. (-1 for use all)')
    parser.add_argument('--target_train_size',
                        default=-1,
                        type=int,
                        help="Training instances of the target language for training. (-1 for use all)")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--augcopy",
                        default=0,
                        type=int,
                        help='Number of permuted augmented copies for training')
    parser.add_argument("--method",
                        default='mlt_multi',
                        choices=['mlt', 'mlt_mix', 'gold_only', 'gold_all', 'gold_mix', 'mlt_multi', 'mlt_multi_mix', 'metaw', 'metawt', 'metawt_multi', 'metaxl', 'joint_training', 'jt-metaxl'],
                        type=str,
                        help="Method for meta learning.")
    parser.add_argument("--rest_all",
                        default=False,
                        action='store_true',
                        help='Use all train data for source langs (default: False).')
    parser.add_argument("--main_lr",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--meta_lr",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--sinkhorn_lr",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--reweighting_lr",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epochs",
                        default=10.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=5e-4, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--data_seed',
                        type=int,
                        default=42,
                        help="random seed for data initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--amp', type=int, default=-1,
                        help="For fp16: Apex AMP optimization level selected in [0, 1, 2, and 3]."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--layers', type=str, default=None,
                        help="Layer numbers concatenated with ',', e.g., 1,2,3")
    parser.add_argument('--meta_per_lang', action="store_true", default=False,
                        help="Whether to construct a meta network per language.")
    parser.add_argument('--struct', type=str, default="transformer",
                        help="The stacked structure of transfer component.")
    parser.add_argument('--tokenizer_dir', type=str, default=None,
                        help="The directory of tokenizer for unseen bert languages.")
    parser.add_argument('--bert_model_type', type=str, default="ori",
                        choices=["ori", "empty", "reinitialize_vocab"])
    parser.add_argument('--add_permutation', action="store_true", default=False,
                        help="Whether to add sinkhorn network for token level permutation.")
    parser.add_argument('--permutation_hidden_size', type=int, default=768,
                        help="The hidden size of the permutation network.")
    parser.add_argument('--no_skip_connection', action="store_true", default=False,
                        help="add skip connection or not")
    parser.add_argument('--temp', type=float, default=0.1,
                        help="The temperature of the permutation network.")
    parser.add_argument('--num_source_langs', type=int, default=1,
                        help='The number of source languages used.')
    parser.add_argument('--source_language_strategy', type=str, default="random", choices=["random", "language_family", "specified", "random2"],
                        help='The strategy to select source languages.')
    parser.add_argument('--portion', type=int, default=2,
                        help="1/n used for training")
    parser.add_argument('--source_languages', type=str,
                        help='Source languages that delimited by ,')
    parser.add_argument('--add_instance_weights', action="store_true",
                        help='Whether to reweight instances or not.')
    parser.add_argument('--weights_from', type=str, default="features",
                        help="Where does the feature come from?")
    parser.add_argument('--tied', action="store_true",
                        help="whether the weights are tied or not with the feature network.")
    parser.add_argument('--transfer_component_add_weights', action="store_true",
                        help="add weights for perceptron")
    parser.add_argument('--bottle_size', type=int, default=768)
    #parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    #parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    args = parser.parse_args()
    args.magic = 1.0
    args.every = 1

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    '''
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    '''

    processors = {'conll': NerProcessor,
                  'panx': PANXNerProcessor,
                  'panx_100': PANXNerProcessor,
                  'pos': POSProcessor,
                  'sent': SentClassProcessor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        #torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, APEX training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.amp))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.batch_size = args.batch_size // args.gradient_accumulation_steps

    if not args.do_train and not args.do_eval and not args.do_finetune:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    '''
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    '''

    if not args.do_finetune:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    else:
        args.output_dir = os.path.dirname(args.output_dir)

    # print arguments
    for arg in vars(args):
        logger.info(f"{arg} = {getattr(args, arg)}")

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    logger.info(f"There are {num_labels} labels. {label_list}")

    tokenizer = BERTSequenceTokenizer(args.bert_model, max_len=args.max_seq_length, cache_dir='cache', tokenizer_dir=args.tokenizer_dir)

    if task_name == 'panx':
        # langs = ['af', 'ar']
        langs = ['af', 'ar', 'bg', 'bn', 'bs', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fa',
             'fi', 'fr', 'he', 'hi', 'hr', 'hu', 'id', 'it', 'lt', 'lv', 'mk', 'ms', 'nl', 'no',
             'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sq', 'sv', 'ta', 'tl', 'tr', 'uk', 'vi']
        if args.num_source_langs < 41:
            if args.source_language_strategy == 'random':
                langs = "vi da ar hi fr fi de cs ca no".split() # random.sample(langs, args.num_source_langs)
            elif args.source_language_strategy == 'language_family':
                langs = ['he', 'it', 'bn', 'ms', 'vi', 'et', 'ta', 'fi', 'pl', 'tr']
            elif args.source_language_strategy == 'specified':
                langs = args.source_languages.split(",")
    elif task_name == 'panx_100':
        langs = ['ace', 'als', 'am', 'ang', 'arc', 'arz', 'as', 'ay', 'ba', 'bar', 'bat-smg', 'bh', 'bo', 'cbk-zam', 'cdo', 'ce', 'ceb', 'co', 'crh', 'csb', 'cv', 'diq', 'dv', 'eml', 'ext', 'fiu-vro', 'fo', 'frr', 'fur', 'gan', 'gd', 'gn', 'gu', 'hak', 'hsb', 'ia', 'ig', 'ilo', 'io', 'jbo', 'jv', 'km', 'kn', 'ksh', 'ku', 'ky', 'li', 'lij', 'lmo', 'ln', 'map-bms', 'mg', 'mhr', 'mi', 'min', 'mn', 'mt', 'mwl', 'my', 'mzn', 'nap', 'nds', 'ne', 'nov', 'oc', 'or', 'os', 'pa', 'pdc', 'pms', 'pnb', 'ps', 'qu', 'rm', 'rw', 'sa', 'sah', 'scn', 'sco', 'sd', 'si', 'so', 'su', 'szl', 'tg', 'tk', 'ug', 'vec', 'vep', 'vls', 'vo', 'wa', 'war', 'wuu', 'xmf', 'yi', 'yo', 'zea', 'zh-classical', 'zh-min-nan']
    elif task_name == 'conll':
        langs = ['eng', 'esp', 'ned', 'deu']
    elif task_name == 'pos':
        if args.source_language_strategy == "random":
            langs = ['Vietnamese-VTB', 'Basque-BDT', 'Estonian-EDT', 'Arabic-PADT', 'Japanese-BCCWJ', 'Tamil-TTB', 'Korean-GSD', 'Turkish-IMST', 'German-GSD', 'Chinese-GSDSimp']
        else:
            langs = ['Irish-IDT', 'Latin-Perseus', 'Latvian-LVTB', 'Galician-CTG', 'Japanese-GSD', 'Finnish-FTB', 'Latin-ITTB', 'Afrikaans-AfriBooms', 'Japanese-BCCWJ', 'Spanish-GSD']
    elif task_name == 'sent':
        if args.source_language_strategy == 'specified':
            langs = args.source_languages.split(",")
        else:
            langs = ["zh", "es", "en", "de", "ja", "fr"]
    else:
        raise Exception('invalid task name %s!' % task_name)


    lang2id = {k:v for v, k in enumerate(langs)}
    logging.info("source languages: " + " ".join(langs))

    tgt_lang = args.tgt_lang # target languages
    src_langs = [x for x in langs if x != tgt_lang]

    # load all data


    if args.do_train or args.do_finetune:
        # note for train, we may sample the data specified by args.train_size
        if args.method == 'gold_all':

            train_t_data = merge_data(args.data_dir, processor, tokenizer, langs, 'train', args.max_seq_length, args.bert_model, args.bert_model_type, args.target_train_size, seed=args.data_seed, rest_all=args.rest_all, tgt_lang=tgt_lang)
            # dev all also needs to subsample
            dev_data = merge_data(args.data_dir, processor, tokenizer, langs, 'dev', args.max_seq_length, args.bert_model, args.bert_model_type, -1, seed=args.data_seed)

            # not used by gold_all
            train_s_data = train_t_data
            #read_data(args.data_dir, processor, tokenizer, tgt_lang, 'train', args.max_seq_length, args.train_size, seed=args.seed)
        elif args.method == 'metawt_multi':
            train_s_data = []
            train_t_data = read_data(args.data_dir, processor, tokenizer, tgt_lang, 'train', args.max_seq_length, args.bert_model, args.bert_model_type, args.target_train_size, seed=args.data_seed)
            # dev all also needs to subsample
            for lang in src_langs:
                train_s_data.append(read_data(args.data_dir, processor, tokenizer, lang, 'train', args.max_seq_length, args.bert_model, args.bert_model_type, args.train_size, seed=args.data_seed))

            # same subsample size for dev, as using a tiny train + a full dev doesn't seem to make sense
            dev_data = read_data(args.data_dir, processor, tokenizer, tgt_lang, 'dev', args.max_seq_length, args.bert_model, args.bert_model_type, -1, seed=args.data_seed)
        else: # for method == gold_only, mlt_multi
            train_t_data = read_data(args.data_dir, processor, tokenizer, tgt_lang, 'train', args.max_seq_length, args.bert_model, args.bert_model_type, args.target_train_size, seed=args.data_seed)
            # train_s will be much larger than train_t as it contains multiple languages
            # train_s not used by gold_only
            if args.method != "gold_only":
                train_s_data = merge_data(args.data_dir, processor, tokenizer, src_langs, 'train', args.max_seq_length, args.bert_model, args.bert_model_type, -1 if args.rest_all else args.train_size, seed=args.data_seed)

            # same subsample size for dev, as using a tiny train + a full dev doesn't seem to make sense
            dev_data = read_data(args.data_dir, processor, tokenizer, tgt_lang, 'dev', args.max_seq_length, args.bert_model, args.bert_model_type, -1, seed=args.data_seed)

        # no subsample for dev and test
        test_data = read_data(args.data_dir, processor, tokenizer, tgt_lang, 'test',  args.max_seq_length, args.bert_model, args.bert_model_type, )

        logger.info(f"First example: {train_t_data[0][0][:10]}")

        if args.local_rank == -1:
            train_t_sampler = None
            train_s_sampler = None
            dev_sampler = None
            test_sampler = None
            batch_size = args.batch_size
        else:
            train_t_sampler = DistributedRandomSampler(train_t_data)
            train_s_sampler = DistributedRandomSampler(train_s_dataa)
            dev_sampler = DistributedSampler(dev_data)
            test_sampler = DistributedSampler(test_data)
            batch_size = int(args.batch_size / int(os.environ['NGPU']))

        train_t_loader = DataIterator(DataLoader(train_t_data, sampler=train_t_sampler, batch_size=batch_size, shuffle=(train_t_sampler is None)))
        if args.method == 'metawt_multi': # this only supports single GPU mode
            train_s_loaders = [DataLoader(train_s_data[i], sampler=None, batch_size=batch_size, shuffle=True) for i in range(len(src_langs))]
        elif args.method == "metawt" or args.method == "metaxl" or args.method == "jt-metaxl"  or args.method == "joint_training":
            train_s_loaders = [DataLoader(train_s_data, sampler=train_s_sampler, batch_size=batch_size, shuffle=(train_s_sampler is None))]
        dev_loader = DataIterator(DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size, shuffle=(dev_sampler is None)))
        test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, shuffle=(test_sampler is None))
    elif args.do_eval:
        dev_data = read_data(args.data_dir, processor, tokenizer, tgt_lang, 'dev', args.max_seq_length, args.bert_model, args.bert_model_type, -1, seed=args.data_seed)

        # no subsample for dev and test
        test_data = read_data(args.data_dir, processor, tokenizer, tgt_lang, args.bert_model, args.bert_model_type, 'test', args.max_seq_length)
        dev_loader = DataIterator(
            DataLoader(dev_data, sampler=None, batch_size=args.batch_size, shuffle=False))
        test_loader = DataLoader(test_data, sampler=None, batch_size=args.batch_size, shuffle=False)

    # Prepare model
    is_xlmr = args.bert_model.startswith("xlm")
    ConfigClass = XLMRobertaConfig if is_xlmr else BertConfig
    SequenceTagger = XLMRSequenceTagger if is_xlmr else BERTSequenceTagger
    if not args.do_train and (args.do_finetune or args.do_eval):
        config = ConfigClass.from_json_file(os.path.join(args.output_dir, "config.json"))
        model = SequenceTagger(config)
        logger.info(f"Loading an empty bert model with a vocab size {config.vocab_size}")
    elif args.do_train:
        config = ConfigClass.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name,
                                            output_hidden_states=True, cache_dir='cache')
        if args.bert_model_type == "empty":
            config.vocab_size = tokenizer.tokenizer.vocab_size
            model = SequenceTagger(config)
            logger.info(f"Loading an empty bert model with a vocab size {config.vocab_size}")
        else:
            model = SequenceTagger.from_pretrained(args.bert_model, config=config, cache_dir='cache')
            embeddings = model.roberta.embeddings if is_xlmr else model.bert.embeddings
            if args.bert_model_type == "reinitialize_vocab":
                config.vocab_size = tokenizer.tokenizer.vocab_size
                pretrained_embeddings = embeddings.word_embeddings.weight.data.clone()
                embeddings.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
                original_tokenizer = BERTSequenceTokenizer(args.bert_model, max_len=args.max_seq_length, cache_dir='cache')
                for i, word in enumerate(tokenizer.tokenizer.vocab):
                    if word in original_tokenizer.tokenizer.vocab:
                        index = original_tokenizer.tokenizer.convert_tokens_to_ids(word)
                        embeddings.word_embeddings.weight[i].data.copy_(pretrained_embeddings[index])
                logger.info(f"Reloaded bert embeddings with a vocab size {config.vocab_size}")



    if args.layers is not None:
        layers = args.layers.split(",")
    else:
        layers = []

    if args.method in ['metaw', 'metawt']:
        raptors = VNet(1, 512, 1)
    elif args.method in ['metaw_multi', 'metawt_multi']:
        raptors = WNets(512, len(src_langs))
    elif args.method == "metaxl" or args.method == "jt-metaxl":
        raptors = Raptors(config, len(layers), len(src_langs) if args.meta_per_lang else 1, struct=args.struct, add_weights=args.transfer_component_add_weights, tied=args.tied, bottle_size=args.bottle_size)
    else:
        raptors = None # Raptors vs Raptor

    # permutate_network = None
    # if args.add_permutation:
    #     permutate_network = Permutation(config=config, in_dim=config.hidden_size, h_dim=args.permutation_hidden_size,
    #                                     out_dim=config.max_position_embeddings, temp=args.temp, no_skip_connection=args.no_skip_connection)

    reweighting_module = None
    if args.add_instance_weights:
        if args.weights_from == "features":
            reweighting_module = VNet(config.hidden_size, args.bottle_size, 1)
        elif args.weights_from == "loss":
            reweighting_module = VNet(1, args.bottle_size, 1)
    if not args.do_train and (args.do_finetune or args.do_eval):
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "best.pt")))
        if raptors is not None:
            raptors.load_state_dict(torch.load(os.path.join(args.output_dir, "best_meta.pt")))
            logging.info(f"Reloaded model and raptors from best.pt, best_meta.pt.")
        # if permutate_network is not None:
        #     permutate_network.load_state_dict(torch.load(os.path.join(args.output_dir, "best_permutation.pt")))
        #     logging.info(f"Reloaded permutate network from best_permutation.pt.")
        if reweighting_module is not None:
            reweighting_module.load_state_dict(torch.load(os.path.join(args.output_dir), "best_weights.pt"))

    num_model_parameters = calculate_parameters(model)
    num_meta_network_parameters = 0
    num_permutate_network = 0
    num_reweighting_network = 0
    if raptors is not None:
        num_meta_network_parameters = calculate_parameters(raptors)
    # if permutate_network is not None:
    #     num_permutate_network = calculate_parameters(permutate_network)
    if reweighting_module is not None:
        num_reweighting_network = calculate_parameters(reweighting_module)
    total_parameters = num_model_parameters + num_meta_network_parameters + num_permutate_network + num_reweighting_network
    logging.info(f"Model parameters: {num_model_parameters}")
    logging.info(f"Meta network parameters: {num_meta_network_parameters}")
    logging.info(f"Permutation network parameters: {num_permutate_network}")
    logging.info(f"Reweighting network parameters: {num_reweighting_network}")
    logging.info(f"Total parameters: {total_parameters}")

    # if args.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)
    if raptors is not None:
        raptors.to(device)
    # if permutate_network is not None:
    #     permutate_network.to(device)
    if reweighting_module is not None:
        reweighting_module.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.main_lr, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    meta_opt = None
    if raptors is not None:
        meta_opt = torch.optim.Adam(raptors.parameters(), lr=args.meta_lr, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    # if permutate_network is not None:
    #     sinkhorn_opt = torch.optim.Adam(permutate_network.parameters(), lr=args.sinkhorn_lr, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    #     logging.info("Initialized sinkhorn optimizer.")
    reweighting_opt = None
    if reweighting_module is not None:
        reweighting_opt = torch.optim.Adam(reweighting_module.parameters(), lr=args.reweighting_lr, eps=args.adam_epsilon, weight_decay=args.weight_decay)
        logging.info("Initialized reweighting optimizer.")

    # change this for multiple train_s loader settings
    if args.do_train or args.do_finetune:
        if args.method != "gold_only":
            if type(train_s_data) is list:
                num_train_optimization_steps = sum([len(x) for x in train_s_data]) * args.epochs / batch_size
            else:
                num_train_optimization_steps = len(train_s_data) * args.epochs / batch_size # note the steps is counted based on train_s, which is 40/41 langs
        else:
            num_train_optimization_steps = len(
                train_t_data) * args.epochs / batch_size  # note the steps is counted based on train_s, which is 40/41 langs

    if args.do_train:
        warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)
        if raptors is not None:
            meta_scheduler = get_linear_schedule_with_warmup(meta_opt, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)
        # if args.add_permutation:
        #     permutation_scheduler = get_linear_schedule_with_warmup(sinkhorn_opt, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)
        if args.add_instance_weights:
            reweighting_scheduler = get_linear_schedule_with_warmup(reweighting_opt, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)


    if args.amp > -1:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    if args.do_train:
        # naive logging file
        _logw=open(os.path.join(args.output_dir, 'logging.txt'), 'w')

        best_val_metric = float('inf')
        model.train()

        for epoch in trange(int(args.epochs), desc="Epoch"):

            if args.method == "gold_only":
                for step, batch_train_t in enumerate(tqdm(train_t_loader.loader, desc="Iteration")):
                    batch_train_t = tuple(t.to(device) for t in batch_train_t)  # tgt train

                    train_t_ids, train_t_mask, train_t_labels = batch_train_t
                    train_t_ids, train_t_mask, train_t_labels = trim_input(train_t_ids, train_t_mask, train_t_labels, args.train_max_seq_length)

                    if len(train_t_ids) == 1:
                        continue

                    if args.augcopy > 0:
                        train_t_ids, train_t_mask, train_t_labels = permute_aug(train_t_ids, train_t_mask, train_t_labels, args.augcopy)


                    loss_meta, loss_train = step_gold_only(model, optimizer,
                                                           train_t_ids, train_t_mask, train_t_labels,
                                                           args)
                    if step % args.every == 0 and args.local_rank <= 0:  # only do eval on first GPU
                            tqdm.write('Step:%d\tLoss_Meta:%.6f\tLoss_Train:%.6f' % (step, (loss_meta).item(), (loss_train).item()))
                    scheduler.step()
            else:
                for j, train_s_loader in enumerate(train_s_loaders): # note here
                    logger.info(f"Updating on language {src_langs[j]} ..")
                    logger.info(f"Training batch size = {batch_size}")
                    for step, batch_train_s in enumerate(tqdm(train_s_loader, desc="Iteration")): # count epoch based on merged src langs loader
                        batch_train_t = next(train_t_loader)

                        batch_train_s = tuple(t.to(device) for t in batch_train_s) # src train
                        batch_train_t = tuple(t.to(device) for t in batch_train_t) # tgt train

                        train_s_ids, train_s_mask, train_s_labels = batch_train_s
                        train_s_ids, train_s_mask, train_s_labels = trim_input(train_s_ids, train_s_mask, train_s_labels, args.train_max_seq_length)
                        # print(train_s_labels)

                        train_t_ids, train_t_mask, train_t_labels = batch_train_t
                        train_t_ids, train_t_mask, train_t_labels = trim_input(train_t_ids, train_t_mask, train_t_labels, args.train_max_seq_length)

                        eta = scheduler.get_last_lr()[0]


                        # print("*"*20, "source", "*"*20)
                        # print(train_s_ids.shape, train_s_labels.shape)
                        # print("*" * 20, "target", "*" * 20)
                        # print(train_t_ids.shape, train_t_labels.shape)

                        if args.method == "metaxl":
                            half = int(len(train_t_ids)/args.portion)
                            eval_ids, eval_mask, eval_labels = train_t_ids[half:], train_t_mask[half:], train_t_labels[half:]
                            train_t_ids, train_t_mask, train_t_labels = train_t_ids[:half], train_t_mask[:half], train_t_labels[:half]
                            if len(eval_ids) == 0 or len(train_t_ids) == 0:
                                continue

                            print("*"*20, "source", "*"*20)
                            print(eval_ids.shape, eval_labels.shape)
                            print("*" * 20, "target", "*" * 20)
                            print(train_t_ids.shape, train_t_labels.shape)

                            layers = [int(l) for l in layers]
                            loss_meta, loss_train = step_metaxl(model, optimizer,
                                                                    raptors, meta_opt,
                                                                    reweighting_module, reweighting_opt,
                                                                    train_s_ids, train_s_mask, train_s_labels,
                                                                    train_t_ids, train_t_mask, train_t_labels,
                                                                    eval_ids, eval_mask, eval_labels,
                                                                    j if args.meta_per_lang else 0, layers, eta, args)

                            # logger.info(raptors.nets[0][0].weight)
                            # logger.info(meta_opt)
                            # logger.info(optimizer)
                        elif args.method == "jt-metaxl":
                            layers = [int(l) for l in layers]
                            loss_meta, loss_train = step_jt_metaxl(model, optimizer,
                                                                    raptors, meta_opt,
                                                                    reweighting_module, reweighting_opt,
                                                                    train_s_ids, train_s_mask, train_s_labels,
                                                                    train_t_ids, train_t_mask, train_t_labels,
                                                                    j if args.meta_per_lang else 0, layers, eta, args)

                        elif args.method == 'metawt':
                            half = int(len(train_t_ids) / 2)
                            eval_ids, eval_mask, eval_labels = train_t_ids[half:], train_t_mask[half:], train_t_labels[
                                                                                                        half:]
                            train_t_ids, train_t_mask, train_t_labels = train_t_ids[:half], train_t_mask[
                                                                                            :half], train_t_labels[
                                                                                                    :half]
                            loss_meta, loss_train = step_metawt_mix(model, optimizer, raptors, meta_opt,
                                                                    train_s_ids, train_s_mask, train_s_labels,
                                                                    train_t_ids, train_t_mask, train_t_labels,
                                                                    eval_ids, eval_mask, eval_labels,
                                                                    eta, args)
                        elif args.method == 'metawt_multi':
                            half = int(len(train_t_ids) / 2)
                            eval_ids, eval_mask, eval_labels = train_t_ids[half:], train_t_mask[half:], train_t_labels[half:]
                            train_t_ids, train_t_mask, train_t_labels = train_t_ids[:half], train_t_mask[:half], train_t_labels[:half]
                            loss_meta, loss_train = step_metawt_multi_mix(model, optimizer, raptors, meta_opt,
                                                                          train_s_ids, train_s_mask, train_s_labels,
                                                                          train_t_ids, train_t_mask, train_t_labels,
                                                                          eval_ids, eval_mask, eval_labels,
                                                                          eta, args, j)
                        elif args.method in "joint_training":
                            loss_meta, loss_train = step_gold_mix(model, optimizer,
                                                                  data_s=train_s_ids, mask_s=train_s_mask, target_s=train_s_labels,
                                                                  data_g=train_t_ids, mask_g=train_t_mask, target_g=train_t_labels,
                                                                  args=args)
                        else:
                            raise Exception('Method %s not implemented yet.' % args.method)


                        logger.info("Step: " + str(step) + "\n")
                        if step % args.every == 0 and args.local_rank <= 0: # only do eval on first GPU
                            tqdm.write('Step:%d\tLoss_Meta:%.6f\tLoss_Train:%.6f' % (step, loss_meta.item(), loss_train.item()))
                            logger.info('Step:%d\tLoss_Meta:%.6f\tLoss_Train:%.6f\n' % (step, loss_meta.item(), loss_train.item()))


                        # scheduler update per step
                        scheduler.step()
                        if raptors is not None:
                            meta_scheduler.step()
                        # if args.add_permutation:
                        #     permutation_scheduler.step()
                        if args.add_instance_weights:
                            reweighting_scheduler.step()

            model.eval()
            if raptors is not None:
                raptors.eval()
            # if permutate_network is not None:
            #     permutate_network.eval()
            if reweighting_module is not None:
                reweighting_module.eval()
            val_score, val_acc, val_precision, val_recall = eval(model, dev_loader.loader, processor, for_classification=(args.task_name=="sent"))
            test_score, test_acc, test_precision, test_recall = eval(model, test_loader, processor, for_classification=(args.task_name=="sent"))
            model.train()
            if raptors is not None:
                raptors.train()
            # if permutate_network is not None:
            #     permutate_network.train()
            if reweighting_module is not None:
                reweighting_module.train()

            if args.local_rank <=0 and -val_score < best_val_metric: # val_acc: the larger the better
                best_val_metric = -val_score
                # torch.save(model.state_dict(), os.path.join(args.output_dir, 'best.pt'))
                # if raptors is not None:
                #     torch.save(raptors.state_dict(), os.path.join(args.output_dir, 'best_meta.pt'))
                # if args.add_permutation:
                #     torch.save(permutate_network.state_dict(), os.path.join(args.output_dir, 'best_permutate.pt'))
                # if args.add_instance_weights:
                #     torch.save(reweighting_module.state_dict(), os.path.join(args.output_dir, 'best_weights.pt'))

            '''
            alphas = raptors.get_alpha().detach().cpu().numpy()
            '''

            tqdm.write('Loss_Meta:%.4f\tLoss_Train:%.4f\tDev F1:%.4f\tDev ACC:%.4f\tDev Precision:%.4f\tDev Recall:%.4f\tBest Dev F1 so far:%.4f' % (loss_meta.item(), loss_train.item(), val_score, val_acc, val_precision, val_recall, -best_val_metric))
            tqdm.write('Loss_Meta:%.4f\tLoss_Train:%.4f\tTest F1:%.4f\tTest ACC:%.4f\tTest Precision:%.4f\tTest Recall:%.4f' % (loss_meta.item(), loss_train.item(), test_score, test_acc, test_precision, test_recall))
            _logw.write('%s\t%d\tDev F1: %.4f\tTest F1: %.4f\n' % (tgt_lang, epoch, val_score, test_score))
            _logw.flush()
            os.fsync(_logw)

        # eval on best model saved so far
        print ('====== Final performance =======')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best.pt')))
        model.eval()
        score, acc, precision, recall = eval(model, test_loader, processor, for_classification=(args.task_name == "sent"))
        print ('Best Dev F1:', -best_val_metric)
        print ('Test F1:', score, 'Test ACC:', acc, 'Precision:', precision, 'Recall:', recall)
        _logw.write('%s\tFinal best Dev F1: %.4f\tTest F1: %.4f\n' % (tgt_lang, -best_val_metric, score))
        _logw.flush()
        os.fsync(_logw)
        # close ad-hoc log
        _logw.close()

        with open(os.path.join(args.output_dir, 'result.txt'), 'w') as w:
            w.write('Test F1: %.4f\tTest ACC: %.4f\tPrecision: %.4f\tRecall: %.4f\n' % (score, acc, precision, recall))
            w.write('Best Dev F1: %.4f\n' % (-best_val_metric))
            w.write('Test F1: %.4f\n' % (score))

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_save.save_pretrained(args.output_dir)
        #tokenizer.save_pretrained(args.output_dir)
        label_map = {i : label for i, label in enumerate(label_list,1)}
        model_config = {'bert_model':args.bert_model, 'max_seq_length':args.max_seq_length,'num_labels':len(label_list)+1,'label_map':label_map}
        json.dump(model_config,open(os.path.join(args.output_dir,'model_config.json'),'w'))
        # Load a trained model and config that you have fine-tuned
    elif args.do_finetune:
        _logw = open(os.path.join(args.output_dir, 'logging_finetune.txt'), 'w')

        best_val_metric = float('inf')
        model.train()
        raptors.train()
        # if args.add_permutation:
        #     permutate_network.train()

        for epoch in trange(int(args.epochs), desc="Epoch"):
            for j, train_s_loader in enumerate(train_s_loaders):  # note here
                logger.info(f"Updating on language {src_langs[j]} ..")
                logger.info(f"Training batch size = {batch_size}")
                for step, batch_train_s in enumerate(
                        tqdm(train_s_loader, desc="Iteration")):  # count epoch based on merged src langs loader
                    batch_train_t = next(train_t_loader)

                    batch_train_s = tuple(t.to(device) for t in batch_train_s)  # src train
                    batch_train_t = tuple(t.to(device) for t in batch_train_t)  # tgt train

                    train_s_ids, train_s_mask, train_s_labels = batch_train_s
                    train_s_ids, train_s_mask, train_s_labels = trim_input(train_s_ids, train_s_mask, train_s_labels,
                                                                           args.train_max_seq_length)

                    train_t_ids, train_t_mask, train_t_labels = batch_train_t
                    train_t_ids, train_t_mask, train_t_labels = trim_input(train_t_ids, train_t_mask, train_t_labels,
                                                                           args.train_max_seq_length)

                    layers = [int(l) for l in layers]
                    loss_t, loss_s = step_metaxl_finetune(model, optimizer, raptors,

                                                            train_s_ids, train_s_mask, train_s_labels,
                                                            train_t_ids, train_t_mask, train_t_labels,
                                                            j if args.meta_per_lang else 0, layers, args)

            model.eval()
            val_score, val_acc, val_precision, val_recall = eval(model, dev_loader.loader, processor)
            test_score, test_acc, test_precision, test_recall = eval(model, test_loader, processor)
            model.train()

            if args.local_rank <= 0 and -val_score < best_val_metric:  # val_acc: the larger the better
                best_val_metric = -val_score
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_finetune.pt'))

            '''
            alphas = raptors.get_alpha().detach().cpu().numpy()
            '''

            tqdm.write(
                'Loss_s:%.4f\tLoss_t:%.4f\tLoss:%.4f\tDev F1:%.4f\tDev ACC:%.4f\tDev Precision:%.4f\tDev Recall:%.4f\tBest Dev F1 so far:%.4f' % (
                loss_s.item(), loss_t.item(), loss_s.item() + loss_t.item(), val_score, val_acc, val_precision, val_recall, -best_val_metric))
            tqdm.write(
                'Test F1:%.4f\tTest ACC:%.4f\tTest Precision:%.4f\tTest Recall:%.4f' % (test_score, test_acc, test_precision, test_recall))
            _logw.write('%s\t%d\tDev F1: %.4f\tTest F1: %.4f\n' % (tgt_lang, epoch, val_score, test_score))
            _logw.flush()
            os.fsync(_logw)

        # eval on best model saved so far
        print('====== Final performance =======')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_finetune.pt')))
        model.eval()
        score, acc, precision, recall = eval(model, test_loader, processor)
        print('Best Dev F1:', -best_val_metric)
        print('Test F1:', score, 'Test ACC:', acc, 'Precision:', precision, 'Recall:', recall)
        _logw.write('%s\tFinal best Dev F1: %.4f\tTest F1: %.4f\n' % (tgt_lang, -best_val_metric, score))
        _logw.flush()
        os.fsync(_logw)
        # close ad-hoc log
        _logw.close()

        with open(os.path.join(args.output_dir, 'result_finetune.txt'), 'w') as w:
            w.write(
                'Test F1: %.4f\tTest ACC: %.4f\tPrecision: %.4f\tRecall: %.4f\n' % (score, acc, precision, recall))
            w.write('Best Dev F1: %.4f\n' % (-best_val_metric))
            w.write('Test F1: %.4f\n' % (score))

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        # model_to_save.save_pretrained(args.output_dir)
    elif args.do_eval:
        # Load a trained model and vocabulary that you have fine-tuned
        model.eval()
        _logw = open(os.path.join(args.output_dir, 'all_result.txt'), 'w')
        score, acc, precision, recall = eval(model, dev_loader.loader, processor)
        print('%s\tDev F1: %.4f\tDev ACC: %.4f\tPrecision: %.4f\tRecall: %.4f' % (tgt_lang, score, acc, precision, recall))
        _logw.write('%s\tDev F1: %.4f\tDev ACC: %.4f\tPrecision: %.4f\tRecall: %.4f\n' % (tgt_lang, score, acc, precision, recall))

        score, acc, precision, recall = eval(model, test_loader, processor)
        print('%s\tTest F1: %.4f\tTest ACC: %.4f\tPrecision: %.4f\tRecall: %.4f' % (tgt_lang, score, acc, precision, recall))
        _logw.write('%s\tTest F1: %.4f\tTest ACC: %.4f\tPrecision: %.4f\tRecall: %.4f\n' % (tgt_lang, score, acc, precision, recall))

        _logw.close()

if __name__ == '__main__':
    main()
