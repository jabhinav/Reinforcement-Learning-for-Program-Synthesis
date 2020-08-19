from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.token import Token
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.seq2seq_decoders import SeqDecoder
from allennlp.modules.attention import Attention
from allennlp.nn import util
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data.dataset import Batch

import torch.nn as nn
import torch
import time
import datetime
import json
from typing import Dict, List, Iterator
from overrides import overrides

torch.manual_seed(1)


@DatasetReader.register("StringProgramReader")
class StringProgramReader(DatasetReader):

    def __init__(self,
                 target_namespace: str,
                 lazy: bool = False,
                 string_indexers: Dict[str, TokenIndexer] = None,
                 program_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy)
        self.string_indexers = string_indexers or {"tokens": SingleIdTokenIndexer()}
        self.program_indexers = program_indexers or {"tokens": SingleIdTokenIndexer(namespace=target_namespace)}

    def text_to_instance(self,
                         ip_token: List[Token],
                         op_token: List[Token],
                         program_token: List[Token] = None) -> Instance:

        # # to convert the tokens into indices. Each `TokenIndexer` could represent each token as a single ID,
        # or a list of character IDs, or something else
        input_field = TextField(ip_token, self.string_indexers)
        output_field = TextField(op_token, self.string_indexers)
        fields = {"input_str": input_field, "output_str": output_field}
        if program_token:
            program_field = TextField(program_token, self.program_indexers)
            fields["program_str"] = program_field

        return Instance(fields)

    def _read(self,
              file_path: str) -> Iterator[Instance]:
        """
        :param file_path: path to json containing dataset samples
        :return: Iterator over instances from dataset
        """
        with open(file_path) as f:
            for line in f:
                data = json.loads(line.strip())
                input_str, output_str = data["input"], data["output"]
                program_str = data["program"]

                # # TOKENIZE raw strings using Token() class
                program_token = [Token(START_SYMBOL)] + [Token(c) for c in program_str.split()] + [Token(END_SYMBOL)]
                for ip, op in zip(input_str, output_str):
                    ip_token = [Token(START_SYMBOL)] + [Token(c) for c in ip] + [Token(END_SYMBOL)]
                    op_token = [Token(START_SYMBOL)] + [Token(c) for c in op] + [Token(END_SYMBOL)]
                    yield self.text_to_instance(ip_token, op_token, program_token)

    def read_json(self, json_dict: Dict) -> List[Instance]:
        """
        Reads data from a single sample containing input and output strings and optionally a transformation
        program
        :param json_dict: data from single sample containing single program and multiple ip-op pairs
        :return: List of instances from json_dict
        """
        input_str, output_str = json_dict["input"], json_dict["output"]
        if 'program' in json_dict.keys():
            program_str = json_dict['program']
            program_token = [Token(START_SYMBOL)] + [Token(c) for c in program_str.split()] + [Token(END_SYMBOL)]
        else:
            program_token = None
        instance_list = []
        for ip, op in zip(input_str, output_str):
            ip_token = [Token(START_SYMBOL)] + [Token(c) for c in ip] + [Token(END_SYMBOL)]
            op_token = [Token(START_SYMBOL)] + [Token(c) for c in op] + [Token(END_SYMBOL)]
            instance_list.append(self.text_to_instance(ip_token, op_token, program_token))
        return instance_list

    def read_batch_json(self, json_dicts: List[Dict]) -> List[Instance]:
        """
        Reads data from multiple samples
        :param json_dicts: data from multiple sample
        :return: List of instances from json_dicts
        """
        instance_list = []
        for json_dict in json_dicts:
            input_str, output_str = json_dict["input"], json_dict["output"]
            if 'program' in json_dict.keys():
                program_str = json_dict['program']
                program_token = [Token(START_SYMBOL)] + [Token(c) for c in program_str.split()] + [Token(END_SYMBOL)]
            else:
                program_token = None

            for ip, op in zip(input_str, output_str):
                ip_token = [Token(START_SYMBOL)] + [Token(c) for c in ip] + [Token(END_SYMBOL)]
                op_token = [Token(START_SYMBOL)] + [Token(c) for c in op] + [Token(END_SYMBOL)]
                instance_list.append(self.text_to_instance(ip_token, op_token, program_token))
        return instance_list


@Model.register("robust_fill")
class RobustFill(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 string_embedder: TextFieldEmbedder,
                 input_encoder: Seq2SeqEncoder,
                 output_encoder: Seq2SeqEncoder,
                 program_decoder: SeqDecoder,
                 num_examples: int,
                 held_out_examples: int) -> None:
        super().__init__(vocab)

        self.string_embedder = string_embedder
        self.inputEncoder = input_encoder
        self.outputEncoder = output_encoder
        self.programDecoder = program_decoder
        self._num_examples = num_examples  # number of observed example pairs per sample to generate the prog
        self._num_held_out = held_out_examples  # number of held-out example pairs per sample to test the generated prog

    @overrides
    def forward_on_instances(self, instances: List[Instance]):
        """
               Takes a list of  :class:`~allennlp.data.instance.Instance`s, converts that text into
               arrays using this model's :class:`Vocabulary`, passes those arrays through
               :func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
               and returns the result.  Before returning the result, we convert any
               ``torch.Tensors`` into numpy arrays and separate the
               batched output into a list of individual dicts per instance. Note that typically
               this will be faster on a GPU (and conditionally, on a CPU) than repeated calls to
               :func:`forward_on_instance`.

               Parameters
               ----------
               instances : List[Instance], required
                   The instances to run the model on.

               Returns
               -------
               A list of the models output for each instance.
               """
        # _batch_size = len(instances)
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = self.decode(self(**model_input))
        return outputs

    # get_text_field_mask() Takes the dictionary of tensors produced by a `TextField`
    # and returns a mask with 0 where the tokens are padded, and 1 otherwise
    @overrides
    def forward(self,
                input_str: Dict[str, torch.Tensor],
                output_str: Dict[str, torch.Tensor],
                program_str: Dict[str, torch.Tensor] = None):
        """
        Forward pass of the seq-to-seq encoder decoder model
        :param input_str: list of input strings
        :param output_str: list of corresponding strings
        :param program_str: list of transformation programs for each input-output example pair
        :return: loss/reward for model training stored in the key 'loss' of returned dictionary 'output_dict'
        """

        # start_time = time.time()

        # Split input_strings/output_strings into observed and held-out examples
        assert input_str['tokens'].size()[0] % (self._num_examples + self._num_held_out) == 0
        train_indices: List[int] = []
        test_indices: List[int] = []
        train_input_str: Dict[str, torch.Tensor] = {}
        train_output_str: Dict[str, torch.Tensor] = {}
        # Test input-output strings will only be used to initialise Generalisation Environment for RL-based training
        test_input_str: Dict[str, torch.Tensor] = {}
        test_output_str: Dict[str, torch.Tensor] = {}
        cuda_device = self._get_prediction_device()  # Get cuda device number (int)

        # Compute set of indices to split the passed dataset into training dataset and held-out dataset
        for start in range(0, input_str['tokens'].size()[0], self._num_examples + self._num_held_out):
            train_indices.extend(range(start, start + self._num_examples))
            test_indices.extend(range(start + self._num_examples, start + self._num_examples + self._num_held_out))
        train_indices = util.move_to_device(torch.tensor(train_indices), cuda_device)
        # Split the batched dataset using computed indices
        train_input_str['tokens'] = input_str['tokens'].index_select(0, train_indices)
        train_output_str['tokens'] = output_str['tokens'].index_select(0, train_indices)
        if test_indices:
            test_indices = util.move_to_device(torch.tensor(test_indices), cuda_device)
            test_input_str['tokens'] = input_str['tokens'].index_select(0, test_indices)
            test_output_str['tokens'] = output_str['tokens'].index_select(0, test_indices)
        if program_str:
            program_str['tokens'] = program_str['tokens'].index_select(0, train_indices)

        # Input String Encoder: Encode observed input_strings only
        embedded_input_str = self.string_embedder(train_input_str)
        input_str_mask = util.get_text_field_mask(train_input_str)
        input_all_hidden = self.inputEncoder(embedded_input_str, input_str_mask)

        # Output String Encoder attending to input String Encoder: Encode observed output_strings only
        embedded_output_str = self.string_embedder(train_output_str)
        output_str_mask = util.get_text_field_mask(train_output_str)
        output_all_hidden = self.outputEncoder(
            embedded_output_str,
            output_str_mask,
            input_all_hidden,
            input_str_mask
        )

        # Compute model state: Duplicating certain variables 'encoder_outputs' and 'source_mask' for ease of use
        model_state = {'input_encoder_outputs': input_all_hidden,
                       'input_encoder_outputs_mask': input_str_mask,
                       'output_encoder_outputs': output_all_hidden,
                       'output_encoder_outputs_mask': output_str_mask}

        # Program Decoder attending to both input_string and output_string encoder
        output = self.programDecoder(model_state, train_input_str, train_output_str,
                                     test_input_str, test_output_str, program_str)

        # print("Fwd Pass time: {}".format(datetime.timedelta(seconds=(time.time() - start_time))))

        return output

    def no_loss_validate_fwd(self, input_str: Dict[str, torch.Tensor],
                             output_str: Dict[str, torch.Tensor],
                             program_str: Dict[str, torch.Tensor] = None) -> Dict[str, List[int]]:
        """
        Forward pass during model validation. Here, we don't worry about the XE loss but the metrics like
        consistency, generalisation etc. during validation, which can better tell the generalisation capability of
        our trained model.
        :param input_str: list of input strings
        :param output_str: list of corresponding strings
        :param program_str: list of transformation programs for each input-output example pair
        :return: dictionary
        """
        assert input_str['tokens'].size()[0] % (self._num_examples + self._num_held_out) == 0
        train_indices: List[int] = []
        test_indices: List[int] = []

        train_input_str: Dict[str, torch.Tensor] = {}
        train_output_str: Dict[str, torch.Tensor] = {}
        # Test input-output strings will only be used to initialise Generalisation Environment for RL-based training
        test_input_str: Dict[str, torch.Tensor] = {}
        test_output_str: Dict[str, torch.Tensor] = {}
        cuda_device = self._get_prediction_device()  # Get cuda device number (int)

        for start in range(0, input_str['tokens'].size()[0], self._num_examples + self._num_held_out):
            train_indices.extend(range(start, start + self._num_examples))
            test_indices.extend(range(start + self._num_examples, start + self._num_examples + self._num_held_out))
        train_indices = util.move_to_device(torch.tensor(train_indices), cuda_device)
        train_input_str['tokens'] = input_str['tokens'].index_select(0, train_indices)
        train_output_str['tokens'] = output_str['tokens'].index_select(0, train_indices)
        if test_indices:
            test_indices = util.move_to_device(torch.tensor(test_indices), cuda_device)
            test_input_str['tokens'] = input_str['tokens'].index_select(0, test_indices)
            test_output_str['tokens'] = output_str['tokens'].index_select(0, test_indices)
        if program_str:
            program_str['tokens'] = program_str['tokens'].index_select(0, train_indices)

        # Input String Encoder
        embedded_input_str = self.string_embedder(train_input_str)
        input_str_mask = util.get_text_field_mask(train_input_str)
        input_all_hidden = self.inputEncoder(embedded_input_str, input_str_mask)

        # Output String Encoder attending to input String Encoder
        embedded_output_str = self.string_embedder(train_output_str)
        output_str_mask = util.get_text_field_mask(train_output_str)
        output_all_hidden = self.outputEncoder(
            embedded_output_str,
            output_str_mask,
            input_all_hidden,
            input_str_mask
        )
        state = {'input_encoder_outputs': input_all_hidden,
                 'input_encoder_outputs_mask': input_str_mask,
                 'output_encoder_outputs': output_all_hidden,
                 'output_encoder_outputs_mask': output_str_mask}

        return self.programDecoder.no_loss_validate_decoder(state, train_input_str, train_output_str,
                                                            test_input_str, test_output_str, program_str)


# For seq2seq output string encoder with attention mechanism
@Seq2SeqEncoder.register("attention_encoder")
class EncoderWithAttention(Seq2SeqEncoder):

    def __init__(self,
                 input_embed_size: int,
                 hidden_size: int,
                 bidirectional: bool,
                 attention: Attention) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self._num_directions = 2 if self.bidirectional else 1
        self.hidden_size = hidden_size*self._num_directions

        # For each LSTM cell, the input is concatenation of: 
        # (1) previous cell's prediction in teacher forcing, y_hat(t-1) or y_true(t) (2) attended hidden vector
        self.input_dim = input_embed_size+self.hidden_size
        self.rnn_cell = nn.LSTMCell(
                                    input_size=self.input_dim,  # The number of expected features in the input x
                                    hidden_size=self.hidden_size  # The number of features in the hidden state h
                                )
        self.attention = attention

    def forward(self,
                output_batch,
                output_batch_mask,
                prev_encoder_outputs,
                prev_encoder_outputs_mask) -> torch.Tensor:
        """
        :param output_batch: Embedded output strings of a batch (batch, seq, embedding_feature)
        :param output_batch_mask: seq of 0 and 1. 0 for padding and 1 for no padding (batch, seq)
        :param prev_encoder_outputs: Input string encoder hidden states (batch, seq, hidden_feature)
        :param prev_encoder_outputs_mask: seq of 0 and 1. 0 for padding and 1 for no padding (batch, seq)
        :return:
        """

        batch_size = prev_encoder_outputs_mask.size(0)

        # Extract the final hidden state of the input_string_encoder from 'prev_encoder_outputs'.
        # This will serve as the initial state for the output_string_encoder.
        # h_0, encoder_hidden = (batch, final_hidden_feature)
        encoder_hidden = util.get_final_encoder_states(
            prev_encoder_outputs, prev_encoder_outputs_mask, self.bidirectional
        )

        # Initialise the LSTM cell memory state, c_0 for output_string_encoder
        encoder_cell = prev_encoder_outputs.new_zeros(
            batch_size, self.get_output_dim()
        )

        # Initialise a list that will contain hidden states (outputs) for the output_string_encoder
        # [h_1, h_2, .... , h_T]
        encoder_all_hidden = []
        prev_encoder_outputs_mask = prev_encoder_outputs_mask.float()

        # Iterate over the sequence of output_string_embeddings to compute the hidden vectors using attention mechanism.
        # (1) Extract the output string embedding vector at position i. (2) Use encoder_hidden i.e. h_{i-1} of
        # output_string_encoder and hidden states of input_string_encoder to compute the attended vector.
        # (3) Concat embed_vec and attn_vec to serve as the input to LSTM at time 'i'.
        # (4) Compute h_i and c_i. Syntax: h_i, c_i = nn.LSTMCell(input_vec, (h_{i-1}, c_{i-1}))
        for i in range(output_batch.size(1)):
            embed_vec = output_batch[:, i]
            attn_vec = self.get_attn_vec(encoder_hidden, prev_encoder_outputs, prev_encoder_outputs_mask)
            input_vec = torch.cat((embed_vec, attn_vec), -1)
            encoder_hidden, encoder_cell = self.rnn_cell(input_vec, (encoder_hidden, encoder_cell))
            encoder_all_hidden.append(encoder_hidden)
        return torch.stack(encoder_all_hidden, 1)

    def get_attn_vec(self,
                     encoder_hidden,
                     prev_encoder_outputs,
                     prev_encoder_outputs_mask):
        input_weights = self.attention(encoder_hidden, prev_encoder_outputs, prev_encoder_outputs_mask)
        attended_input = util.weighted_sum(encoder_hidden, input_weights)
        return attended_input

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.hidden_size

    @overrides
    def is_bidirectional(self) -> bool:
        return self.bidirectional


"""
util.get_final_encoder_states():
    Given the output from a `Seq2SeqEncoder`, with shape `(_batch_size, sequence_length,
    encoding_dim)`, this method returns the final hidden state for each element of the batch,
    giving a tensor of shape `(_batch_size, encoding_dim)`.  This is not as simple as
    `encoder_outputs[:, -1]`, because the sequences could have different lengths.  We use the
    mask (which has shape `(_batch_size, sequence_length)`) to find the final state for each batch
    instance.
    Additionally, if `bidirectional` is `True`, we will split the final dimension of the
    `encoder_outputs` into two and assume that the first half is for the forward direction of the
    encoder and the second half is for the backward direction.  We will concatenate the last state
    for each encoder dimension, giving `encoder_outputs[:, -1, :encoding_dim/2]` concatenated with
    `encoder_outputs[:, 0, encoding_dim/2:]`.

"""