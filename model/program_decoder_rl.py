from typing import Dict, List, Tuple, Optional
from overrides import overrides

import numpy
import torch
import torch.nn.functional as F
from torch.nn import Linear, LSTMCell

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.modules.seq2seq_decoders.seq_decoder import SeqDecoder
from allennlp.data import Vocabulary
from allennlp.modules import Embedding, Attention
from allennlp.modules.seq2seq_decoders.decoder_net import DecoderNet
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import Metric

from model.reinforce_tools import Rolls, batch_rolls_reinforce
from model.reinforce_tools import EnvironmentClasses, Simulator, check_currently_active_tensors
from model.reinforce_tools import debug_memory, check_tensor_memory, stat_cuda
from model.reinforce_tools import RewardCombinationFun
from model.tokens import build_token_tables
from model.validate_beam_search import ValBeamSearch


def gen_samples(vocab, num_examples, input_strings, output_strings, target_tokens):
    """
    Decode back the ip/op strings and target prog to initialise the RL environment for simulating generated programs
    :param vocab:
    :param num_examples: Num examples per sample to be used for training
    :param input_strings: Tensor consisting of encoded input strings
    :param output_strings: Tensor consisting of encoded output strings
    :param target_tokens: Tensor consisting of target encoded program sequences repeated after every 'num_example'
    :return:
    """

    ignore_symbols = [START_SYMBOL, END_SYMBOL, '@@PADDING@@']

    # Obtain single program per ip-op pairs set. Decode using vocab of AllenNLP and remove ignore_symbols.
    # Decoded token is in 'string' format, while token_op_table has keys in 'int' format
    # Shape: (_batch_size/num_examples,)
    expected_programs = [[int(vocab.get_token_from_index(token_id.item(), "program_tokens")) for token_id in prog
                          if vocab.get_token_from_index(token_id.item(), "program_tokens") not in ignore_symbols]
                         for prog in target_tokens[::num_examples]]

    # Decode input-output strings, merge characters to form string, and group 'num_examples' ip-op pairs
    input_strings = [''.join([vocab.get_token_from_index(token_id.item()) for token_id in inp
                              if vocab.get_token_from_index(token_id.item()) not in ignore_symbols])
                     for inp in input_strings]
    set_example_input_strings = [input_strings[i:i+num_examples] for i in range(0, len(input_strings), num_examples)]

    output_strings = [''.join([vocab.get_token_from_index(token_id.item()) for token_id in out
                               if vocab.get_token_from_index(token_id.item()) not in ignore_symbols])
                      for out in output_strings]
    set_example_output_strings = [output_strings[i:i + num_examples]
                                  for i in range(0, len(output_strings), num_examples)]

    return expected_programs, set_example_input_strings, set_example_output_strings


@SeqDecoder.register("rl_attention_decoder")
class DecoderWithAttention(SeqDecoder):
    """
    An autoregressive decoder that can be used for most seq2seq tasks.

    # Parameters

    vocab : `Vocabulary`, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    decoder_net : `DecoderNet`, required
        Module that contains implementation of neural network for decoding output elements
    max_decoding_steps : `int`, required
        Maximum length of decoded sequences.
    target_embedder : `Embedding`
        Embedder for target tokens.
    target_namespace : `str`, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    beam_size : `int`, optional (default = 4)
        Width of the beam for beam search.
    tensor_based_metric : `Metric`, optional (default = None)
        A metric to track on validation data that takes raw tensors when its called.
        This metric must accept two arguments when called: a batched tensor
        of predicted token indices, and a batched tensor of gold token indices.
    token_based_metric : `Metric`, optional (default = None)
        A metric to track on validation data that takes lists of lists of tokens
        as input. This metric must accept two arguments when called, both
        of type `List[List[str]]`. The first is a predicted sequence for each item
        in the batch and the second is a gold sequence for each item in the batch.
    scheduled_sampling_ratio : `float` optional (default = 0)
        Defines ratio between teacher forced training and real output usage. If its zero
        (teacher forcing only) and `decoder_net`supports parallel decoding, we get the output
        predictions in a single forward pass of the `decoder_net`.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        decoder_net: DecoderNet,
        max_decoding_steps: int,
        target_embedder: Embedding,
        num_examples: int,
        held_out_examples: int,
        nb_rollouts: int = 100,
        target_namespace: str = "tokens",
        tie_output_embedding: bool = False,
        scheduled_sampling_ratio: float = 0,
        label_smoothing_ratio: Optional[float] = None,
        beam_size: int = 4,
        top_k: int = 100,
        tensor_based_metric: Metric = None,
        token_based_metric: Metric = None,
        training_signal: str = "supervised",
        rl_environment: str = None,
        reward_comb: str = "RenormExpected",
        rl_inner_batch: int = 1
    ) -> None:
        super().__init__(target_embedder)

        self._vocab = vocab

        # Decodes(single-step) a new hidden state from previous hidden state.
        self._decoder_net = decoder_net

        # Max num of steps we want to take while sampling a program from decoder in model_sample
        self._max_decoding_steps = max_decoding_steps

        # Name space for token encoding of program which is different from that of strings
        self._target_namespace = target_namespace

        # For cross-entropy loss. To train for some mislabelled samples in the training dataset. Converts hard labels
        # into soft labels. For example: with a value of 0.2, a 4-class classification target, say [0,0,1,0] ->
        # [0.05, 0,05, 0.85, 0.05] using new_labels= onehot_labels*(1-label_smoothing)+label_smoothing / num_classes
        self._label_smoothing_ratio = label_smoothing_ratio

        # Num of observed examples and number of held-out examples per program for training/evaluation
        self._num_examples = num_examples
        self._num_held_out = held_out_examples

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        # We need the start symbol to provide as the input at the first time-step of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self._vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self._vocab.get_token_index(END_SYMBOL, self._target_namespace)

        # Set of symbols to be ignored while recovering the program/string from sequences using encoded ALLenNLP vocab
        self._ignore_symbols = [START_SYMBOL, END_SYMBOL, '@@PADDING@@', "@@UNKNOWN@@"]

        # Beam size for beam search and specify top_k, if we want top-k (< beam size) samples from beam search
        self.beam_size = beam_size
        if top_k is None or top_k >= beam_size:
            self.top_k = beam_size
        else:
            self.top_k = top_k
        self._beam_search = BeamSearch(
            self._end_index, max_steps=max_decoding_steps, beam_size=self.beam_size
        )  # optional argument: per_node_beam_size = beam_size(default)

        # For evaluation(testing), we use validation beam search which is replica of AllenNLP's beam search.
        # We use a copy so as to tune its code as per the requirements if need arises in the future.
        # Beam-size = 100 to evaluate top-100 (Set as default)
        self.val_beam_size = 100
        self.val_max_steps = 50
        self._val_beam_search = ValBeamSearch(self._end_index, max_steps=self.val_max_steps,
                                              beam_size=self.val_beam_size)

        # target_vocab is not the same as vocab of op_token_table/token_op_table
        target_vocab_size = self._vocab.get_vocab_size(self._target_namespace)

        if self.target_embedder.get_output_dim() != self._decoder_net.target_embedding_dim:
            raise ConfigurationError(
                "Target Embedder output_dim doesn't match decoder module's input."
            )

        # Feed-forward layer before Max Pooling
        self.max_pool_linear = Linear(
            self._decoder_net.get_output_dim(), self._decoder_net.get_output_dim()
        )

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(
            self._decoder_net.get_output_dim(), target_vocab_size
        )

        # Number of trajectories to be sampled from the decoder per training instance (in training signal = "rl")
        self._nb_samples = nb_rollouts

        if tie_output_embedding:
            if self._output_projection_layer.weight.shape != self.target_embedder.weight.shape:
                raise ConfigurationError(
                    "Can't tie embeddings with output linear layer, due to shape mismatch"
                )
            self._output_projection_layer.weight = self.target_embedder.weight

        # These metrics will be updated during training and validation
        self._tensor_based_metric = tensor_based_metric
        self._token_based_metric = token_based_metric

        # Exposure Bias: For a sampled ratio less than smoothing, previous prediction is the input else GT prediction
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        # Have to use this for beam search for multi-example pooling
        # we will set batch size dynamically since provided number of input strings do not correspond to actual
        # batch size which depends on num training and held-out examples.
        self._batch_size = None

        # Training signal sets a training objective. Choose out of supervised, rl or beam_rl
        self.training_signal = training_signal

        # RL environment which generates rewards.
        # BlackBoxConsistency: to generate reward by evaluating programs on observed examples
        # BlackBoxGeneralization: to generate reward by evaluating programs on held-out examples
        self._rl_environment = rl_environment

        # How to combine reward for each decoded program with its probability. 'RenormExpected'
        self.reward_comb_fn = RewardCombinationFun[reward_comb]

        # Not required
        self._rl_inner_batch = rl_inner_batch

        # Simulator to run each decoded program on set of input strings
        token_table = build_token_tables()
        self.simulator = Simulator(token_table.token_op_table)

        # Set it as true when evaluating/testing (will stay False during model validation) a trained model.
        # We want to bypass the custom_metric computation during the forward pass of the Decoder when validating
        # since, we exclusively call no_loss_validate_decoder to do that. See custom_trainer.py
        self.evaluating = False
        self.cuda_device = -1  # default

    def _beam_sample(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Used for training_signal = beam_rl
        Prepare inputs for the beam search, does beam search and returns beam search results.
        """
        batch_size = state["output_encoder_outputs_mask"].size()[0]
        assert batch_size % self._num_examples == 0
        self._batch_size = batch_size // self._num_examples

        start_predictions = state["output_encoder_outputs_mask"].new_full((batch_size//self._num_examples,),
                                                                          fill_value=self._start_index)

        # Reshape tensors in the state dict | shape: (actual_batch_size, num_examples, *dims)
        for key, state_tensor in state.items():
            _, *dims = state_tensor.size()
            state[key] = state_tensor.view(batch_size//self._num_examples, self._num_examples, *dims)

        # TODO: Debug
        # stat_cuda("Before Beam Sampling")

        # shape (all_top_k_predictions): (_batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (_batch_size, beam_size)
        if self.evaluating:
            all_top_k_predictions, log_probabilities = self._val_beam_search.search(start_predictions, state,
                                                                                    self.take_step)
        else:
            all_top_k_predictions, log_probabilities = self._beam_search.search(start_predictions, state,
                                                                                self.take_step)
        # TODO: Debug
        # stat_cuda("After Beam Sampling")

        return {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }

    def _model_sample(self, state: Dict[str, torch.Tensor], nb_samples: int, max_len: int) -> List[Rolls]:
        """
        Used for training_signal = rl
        :param state: Dict of input_encoder_outputs, output_encoder_outputs, input_encoder_outputs_mask,
        output_encoder_outputs_mask, source_mask, encoder_outputs, decoder_hidden, decoder_context
        :param nb_samples: number of programs to decode per set of input-output example pairs
        :param max_len: maximum length of decoded programs
        :return: Rolls which contain per-step predicted program token, associated reward and gradient value
        """
        batch_size, self.max_input_sequence_length, self.input_encoder_dim = state['input_encoder_outputs'].size()
        _, self.max_output_sequence_length, self.output_encoder_dim = state['output_encoder_outputs'].size()
        _, self.decoder_output_dim = state['decoder_hidden'].size()
        source_mask = state["output_encoder_outputs_mask"]
        cuda_device = util.get_device_of(source_mask)

        # Reshape State for ease of replicability
        for key, state_tensor in state.items():
            _, *last_dims = state_tensor.size()
            state[key] = state_tensor.view(batch_size // self._num_examples, self._num_examples, *last_dims)

        # rolls will hold the sample output programs that we are going to collect for each set of input-output strings
        # in the batch. full_probability is the initial prob for what will be sampled.
        full_probability = util.move_to_device(torch.tensor([1.]), cuda_device)
        rolls = [Rolls(action=-1, proba=full_probability, multiplicity=nb_samples, depth=-1)
                 for _ in range(batch_size // self._num_examples)]

        sm = torch.nn.Softmax(dim=1)

        # Batch size will vary as we go along. It will indicate how many inputs we will pass to decoder at once.
        # Initialise it with actual _batch_size
        curr_batch_size = batch_size//self._num_examples

        # Initial input for the decoder step
        # batch_inputs = torch.LongTensor(_batch_size//self.num_examples, 1).fill_(self._start_index)
        # batch_list_inputs = [[self._start_index]]*_batch_size//self.num_examples

        # # Information to be maintained at each time-step:
        # (1) multiplicity: List[ int ] -> How many of this trace have we sampled
        multiplicity = [nb_samples for _ in range(batch_size // self._num_examples)]

        # (2) List[ List[idx] ] -> Trajectory for each trace that we are currently expanding
        trajectories = [[] for _ in range(batch_size // self._num_examples)]

        # (3) cr_list: List[ idx ] -> Which roll/sample is it a trace for
        cr_list = [roll_idx for roll_idx in range(batch_size // self._num_examples)]

        last_predictions = source_mask.new_full((batch_size // self._num_examples,), fill_value=self._start_index)
        # last_predictions = torch.LongTensor(_batch_size // self._num_examples, ).fill_(self._start_index)

        # shape: (steps, _batch_size, target_embedding_dim)
        # steps_embeddings = torch.Tensor([])

        for time_step in range(max_len):
            # Expected shape: (group_size, steps, target_embedding_dim.
            # Here, our group_size is changing which does not allow concatentaion of previous step embeddings with last
            # prediction's embedding. steps_embeddings will be set as None to make it equal to last_predictions_embed
            # state["previous_steps_predictions"] = steps_embeddings
            effective_last_prediction = last_predictions

            # shape: output_projections = (group_size, num_classes)
            # state["previous_steps_predictions"] = effective_last_prediction
            output_projections, state = self._prepare_output_projections(
                effective_last_prediction, state)
            output_projections_prob = sm(output_projections)
            to_sample_from = output_projections_prob

            # Prepare the container for what is needed to be given to the next steps
            new_trajectories = []
            new_multiplicity = []
            new_cr_list = []
            new_batch_list_inputs = []
            # new_batch_checker = []

            # This needs to be collected for each of the samples we do
            parent = []  # -> idx of the parent trace (in current batch) for the sampled output
            next_input = []  # -> sampled output
            sp_proba = []  # -> probability of the sampled output

            for trace_idx in range(curr_batch_size):

                new_batch_list_inputs.append([])

                # For grouping same samples. Key: Word from vocab, Value: Position in new_trajectories.
                # Is required to increment multiplicity of already seen sampled choices for a given trace_idx
                idx_per_sample = {}

                # sampled outputs at time 'time_step' for sample 'trace_idx', choices = (multiplicity,)
                choices = torch.multinomial(to_sample_from.data[trace_idx], num_samples=multiplicity[trace_idx],
                                            replacement=True)

                # To avoid computation for duplicate samples
                for sampled_choice in choices:

                    if sampled_choice in idx_per_sample:
                        # Increase the multiplicity for already seen samples
                        new_multiplicity[idx_per_sample[sampled_choice]] += 1
                    else:
                        idx_per_sample[sampled_choice] = len(new_trajectories)
                        # (1) Each new sample will have a multiplicity of 1
                        new_multiplicity.append(1)

                        # (2) Trajectory for new sample: old_prefix + sampled choice
                        new_traj = trajectories[trace_idx] + [sampled_choice]
                        new_trajectories.append(new_traj)

                        sp_proba.append(output_projections_prob[trace_idx, sampled_choice])

                        # (3) Each sampled choice belongs to the same training sample that his prefix belongs to
                        new_cr_list.append(cr_list[trace_idx])

                        parent.append(trace_idx)
                        next_input.append(sampled_choice)

            # Use 'cr' as the original roll_id of a training sample and expand its trajectory
            for traj, _multiplicity, cr, sp_pb in zip(new_trajectories, new_multiplicity, new_cr_list, sp_proba):
                rolls[cr].expand_samples(traj, _multiplicity, sp_pb)

            # Ignore the trajectories that have reached the end symbol.
            # Each non-terminal trajectory needs to be expanded, so consider it separately by updating curr_batch_size.
            to_continue_mask = [inp != self._end_index for inp in next_input]
            curr_batch_size = sum(to_continue_mask)
            if curr_batch_size == 0:
                # There is nothing left to sample from
                break

            # Update <trajectories>, <multiplicity>, <cr_list> for newly constructed trajectories
            joint = [(mul, traj, cr) for mul, traj, cr, to_cont
                     in zip(new_multiplicity,
                            new_trajectories,
                            new_cr_list,
                            to_continue_mask)
                     if to_cont]
            multiplicity, trajectories, cr_list = zip(*joint)

            # For next decoder step
            next_batch_inputs = [inp for inp in next_input if inp != self._end_index]
            # batch_inputs = torch.LongTensor(next_batch_inputs).view(-1, 1)
            # batch_list_inputs = [[inp] for inp in next_batch_inputs]

            # Parent Id in curr_batch - for which next state is required
            parents_to_continue = [parent_idx for (parent_idx, to_cont)
                                   in zip(parent, to_continue_mask) if to_cont]
            parent = util.move_to_device(torch.tensor(parents_to_continue), cuda_device)

            # Prepare state for the next step: Current batch size has changed.
            for key, state_tensor in state.items():
                state[key] = state_tensor.index_select(0, parent)

            # Update last_predictions. shape: (new_group_size,)
            last_predictions = util.move_to_device(torch.tensor(next_batch_inputs).view(-1,), cuda_device)

        return rolls

    def _forward_loss(
        self, state: Dict[str, torch.Tensor], target_tokens: Dict[str, torch.LongTensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """

        if not self._scheduled_sampling_ratio > 0:
            raise ConfigurationError("Scheduled sampling ratio should be non-zero for the model logic to hold")

        batch_size, self.max_input_sequence_length, self.input_encoder_dim = state['input_encoder_outputs'].size()
        _, self.max_output_sequence_length, self.output_encoder_dim = state['output_encoder_outputs'].size()
        _, self.decoder_output_dim = state['decoder_hidden'].size()

        # Reshape State for ease of ability to replicate
        for key, state_tensor in state.items():
            _, *last_dims = state_tensor.size()
            state[key] = state_tensor.view(batch_size // self._num_examples, self._num_examples, *last_dims)

        # shape: (_batch_size, max_target_sequence_length) -> (_batch_size/num_examples, max_target_sequence_length)
        targets = target_tokens["tokens"][::self._num_examples]

        # _batch_size = source_mask.size()[0]
        _, target_sequence_length = targets.size()

        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.
        num_decoding_steps = target_sequence_length - 1

        # Initialize target predictions with the start index.
        # shape: (_batch_size/num_examples,)
        # last_predictions = torch.LongTensor(_batch_size // self._num_examples, ).fill_(self._start_index)
        last_predictions = state['output_encoder_outputs_mask'].new_full(size=(batch_size // self._num_examples,),
                                                                         fill_value=self._start_index)

        step_logits: List[torch.Tensor] = []

        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:

                # shape: (_batch_size/num_examples, )
                effective_last_prediction = last_predictions
            else:
                # shape: (_batch_size/num_examples,)
                effective_last_prediction = targets[:, timestep]

            # shape: output_projections = (_batch_size/self.num_examples=4, num_classes)
            output_projections, state = self._prepare_output_projections(
                effective_last_prediction, state
            )

            # list of tensors, shape: (_batch_size/num_examples, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # shape (predicted_classes): (_batch_size/num_examples,)
            _, predicted_classes = torch.max(output_projections, 1)

            # shape (predicted_classes): (_batch_size,)
            # Duplicating each element by a factor of self.num_examples
            last_predictions = predicted_classes

        # shape: (_batch_size/num_examples, num_decoding_steps, num_classes)
        logits = torch.cat(step_logits, 1)

        # Compute loss.
        # shape: (_batch_size, max_target_sequence_length)
        target_mask = util.get_text_field_mask(target_tokens)
        # shape: (_batch_size/num_examples, max_target_sequence_length)
        target_mask = target_mask[::self._num_examples]

        # Slicing both targets and its mask since targets is duplication by a factor of self.number_examples
        # loss = self._get_loss(logits, targets[::self.num_examples], target_mask[::self.num_examples])
        loss = self._get_loss(logits, targets, target_mask)

        output_dict = {"loss": loss}

        return output_dict

    def _prepare_output_projections(
            self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # TODO: DEBUG
        # stat_cuda("1) _prepare_output_projections | Start")

        # In beam sample, tensors might be shuffled, that's why calling contiguous to reallocated memory and
        # then re-shaping the tensor
        for key, state_tensor in state.items():
            group_size, num_examples, *dims = state_tensor.size()
            state[key] = state_tensor.contiguous().view(group_size*num_examples, *dims)

        assert num_examples == self._num_examples

        # TODO: DEBUG
        # stat_cuda("2) _prepare_output_projections | State Values Extracted and Reshaped")

        # Repeat prediction for each example | shape: -> (group_size*num_examples,)
        output = last_predictions.repeat_interleave(num_examples, dim=0)
        # Embed the predictions | shape: (group_size*num_examples, 1, target_embedding_dim)
        output = self.target_embedder(output).unsqueeze(1)

        # TODO: DEBUG
        # stat_cuda("3) _prepare_output_projections | Last Predictions Embedded")

        # Decoder output (decoder_hidden) is hidden vector of shape: (group_size*num_examples, decoder_output_dim)
        # Decoder net only requires last_prediction_embedding which we provide directly in previous_steps_predictions
        state = self._decoder_net(
            state=state,
            last_predictions_embedding=output,
        )

        # TODO: DEBUG
        # stat_cuda("4) _prepare_output_projections | New Decoder state computed")
        # check_currently_active_tensors("4 _prepare_output_projections | After | New Decoder state computed")

        output = state["decoder_hidden"]
        if self._decoder_net.decodes_parallel:
            output = output[:, -1, :]

        # Preparing decoder output (i.e. decoder_hidden) before max-pooling
        output = torch.tanh(self.max_pool_linear(output))\
            .view(-1, self._num_examples, self._decoder_net.hidden_size)\
            .permute(0, 2, 1)
        # Pool decoder output to prepare for output projections
        output = F.max_pool1d(output, kernel_size=self._num_examples).squeeze(2)

        # Compute output projections | shape: (group_size, num_classes)
        output = self._output_projection_layer(output)

        # Reshape tensors (No contiguity of memory reqd.) | shape: (orig_group_size, num_examples, *dims)
        for key, state_tensor in state.items():
            group_size, *dims = state_tensor.size()
            state[key] = state_tensor.view(group_size//self._num_examples, self._num_examples, *dims)

        # TODO: DEBUG
        # stat_cuda("5) _prepare_output_projections | Completed")

        return output, state

    def _get_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (_batch_size,
        num_decoding_steps, num_classes), target indices of size (_batch_size, num_decoding_steps+1)
        and corresponding masks of size (_batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of `targets` is expected to be greater than that of `logits` because the
        decoder does not need to compute the output corresponding to the last timestep of
        `targets`. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (_batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (_batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(
            logits, relevant_targets, relevant_mask, label_smoothing=self._label_smoothing_ratio
        )

    def get_output_dim(self):
        return self._decoder_net.get_output_dim()

    def take_step(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        # Parameters

        last_predictions : `torch.Tensor`
            A tensor of shape `(group_size,)`, which gives the indices of the predictions
            during the last time step.
        state : `Dict[str, torch.Tensor]`
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape `(group_size, *)`, where `*` can be any other number
            of dimensions.

        # Returns

        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of `(log_probabilities, updated_state)`, where `log_probabilities`
            is a tensor of shape `(group_size, num_classes)` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while `updated_state` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though `group_size` is not necessarily
            equal to `_batch_size`, since the group may contain multiple states
            for each source sentence in the batch.
        """

        # Doing this beam search messes up positions of examples that belong to same program. (Not true now)
        shuffle_tensors = (last_predictions.size(0) == (self._batch_size * self.beam_size))
        # shuffle_tensors = False

        if shuffle_tensors:
            last_predictions = last_predictions.view(self._batch_size, -1)\
                .transpose(1, 0).reshape(last_predictions.shape)
            for key, state_tensor in state.items():
                _, *last_dims = state_tensor.size()
                state[key] = state_tensor.view(self._batch_size, -1, *last_dims)\
                    .transpose(1, 0).reshape(state_tensor.shape)
        # Compute output projections and update decoder hidden and context of state | shape: (group_size, num_classes)
        output, state = self._prepare_output_projections(last_predictions, state)
        # Compute class log probabilities | shape: (group_size, num_classes)
        output = F.log_softmax(output, dim=-1)
        # shape: (group_size, num_classes)
        # class_log_probabilities = class_log_probabilities.repeat_interleave(self._num_examples, dim=0)

        if shuffle_tensors:
            _, *last_dims = output.size()
            output = output.view(-1, self._batch_size, *last_dims)\
                .transpose(1, 0).reshape(output.shape)
            for key, state_tensor in state.items():
                _, *last_dims = state_tensor.size()
                state[key] = state_tensor.view(-1, self._batch_size, *last_dims)\
                    .transpose(1, 0).reshape(state_tensor.shape)

        return output, state

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(
                    self._tensor_based_metric.get_metric(reset=reset)  # type: ignore
                )
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))  # type: ignore
        return all_metrics

    @overrides
    def forward(self,
                model_state: Dict[str, torch.LongTensor],
                train_input_strings: Dict[str, torch.LongTensor],
                train_output_strings: Dict[str, torch.LongTensor],
                test_input_strings: Dict[str, torch.LongTensor],
                test_output_strings: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None):

        output_dict = {}
        self.cuda_device = util.get_device_of(model_state['output_encoder_outputs_mask'])

        # stat_cuda("\n1. Decoder start")

        train_input_strings = self.decode_string_encoding(train_input_strings['tokens'], self._num_examples)
        train_output_strings = self.decode_string_encoding(train_output_strings['tokens'], self._num_examples)
        if test_input_strings and test_output_strings:
            test_input_strings = self.decode_string_encoding(test_input_strings['tokens'], self._num_held_out)
            test_output_strings = self.decode_string_encoding(test_output_strings['tokens'], self._num_held_out)

        if target_tokens:
            expected_progs = self.decode_trg_prog_encoding(target_tokens['tokens'])

        # Initialise decoder initial state such that it is equal to dictionary containing hidden and context vector
        decoder_init_state = self._decoder_net.init_decoder_state(model_state)

        # Update the 'decoder_state' and 'decoder_context' of state dictionary with initialised values
        model_state.update(decoder_init_state)

        # State is the dictionary containing tensors for hidden state and context
        state = model_state.copy()

        # If target tokens are provided, compute the loss in output_dict = {"loss": loss}
        # We have added extra bool flag -
        # a) During validation (evaluating = False) compute loss; (b) During evaluation (evaluating = True), bypass loss
        if target_tokens and not self.evaluating:

            expected_progs = self.decode_trg_prog_encoding(target_tokens['tokens'])

            # Max length of allowed programs. We allow 10 additional tokens other than the end symbol to be decoded.
            _, target_seq_len = target_tokens['tokens'].size()
            # max_length = (target_seq_len - 1) + 10  # we are separately providing this value as max_decoding_steps

            if self.training_signal == "rl" or self.training_signal == "beam_rl":
                env_class = EnvironmentClasses[self._rl_environment]

                # Declare rewards to receive.
                if self.training_signal == "rl":
                    # Here normalisation is required because we are back propagating reward for each sample in the batch
                    reward_norm = 1 / float(self._nb_samples)
                elif self.training_signal == "beam_rl":
                    # Here no normalisation since we back-propagate the reward for the whole batch and normalise later
                    reward_norm = float(1)
                else:
                    raise NotImplementedError("Unknown training signal")

                if "Consistency" in self._rl_environment:
                    # For generating the reward, here we create a simulation environment which will run the predicted
                    # program on observed example-pairs
                    envs = [env_class(reward_norm, trg_prog, example_input_strings, example_output_strings,
                                      self.simulator, self._vocab.get_index_to_token_vocabulary(self._target_namespace))
                            for trg_prog, example_input_strings, example_output_strings
                            in zip(expected_progs, train_input_strings, train_output_strings)]

                elif "Generalization" in self._rl_environment:
                    # For generating the reward, here we create a simulation environment which will run the predicted
                    # program on held-out example-pairs
                    if not test_input_strings and not test_output_strings:
                        raise ConfigurationError("No held-out examples provided for Generalisation environment based RL"
                                                 "training. Check if whether a non-zero value is provided in your JSON "
                                                 "config or the data contains prescribed number of held-out examples.")

                    envs = [env_class(reward_norm, trg_prog, example_input_strings, example_output_strings,
                                      self.simulator, self._vocab.get_index_to_token_vocabulary(self._target_namespace))
                            for trg_prog, example_input_strings, example_output_strings
                            in zip(expected_progs, test_input_strings, test_output_strings)]

                else:
                    raise NotImplementedError("Unknown environment type")

                if self.training_signal == "rl":
                    # Sample 'nb_samples'/'nb_rollouts' samples from the decoder model
                    rolls = self._model_sample(state, self._nb_samples, self._max_decoding_steps)

                    for roll, env in zip(rolls, envs):
                        # Assign the rewards for each sample
                        roll.assign_rewards(env, [])

                    # Compute the total reward for the mini-batch
                    batch_reward = sum(roll.dep_reward for roll in rolls)

                    # Get all variables and all gradients from all the rolls - CAN BE TIME CONSUMING
                    # variables (tensor): Each contain proba of taking an action at some time-step for some sample in
                    #                     the batch.
                    # grad_variable (scalar): Returned by reinforce_gradient() of each variable's roll
                    #                       = self.dep_reward / (1e-6 + self.proba.data)
                    variables, grad_variables = zip(*batch_rolls_reinforce(rolls))

                    # For each of the sampling probability, we know their gradients.
                    # See https://arxiv.org/abs/1506.05254 for what we are doing,
                    # simply using the probability of the choice made, times the reward of all successors.
                    output_dict = {'loss': util.move_to_device(-torch.tensor([batch_reward]), self.cuda_device),
                                   'variables': variables,
                                   'grad_variables': grad_variables}

                elif self.training_signal == "beam_rl":

                    batch_reward = 0

                    # Beam sample. Keys present: class_log_probabilities, predictions
                    # shape: (predictions): (_batch_size, beam_size, num_decoding_steps)
                    # shape: (class_log_probabilities): (_batch_size, beam_size)
                    beam_decoded = self._beam_sample(state)

                    # TODO: DEBUG
                    # stat_cuda("2. After Beam sampling")

                    batch_size, _ = beam_decoded['class_log_probabilities'].size()

                    # shape (predictions): (group_size, beam_size, num_decoding_steps)
                    preds = beam_decoded['predictions']

                    # shape: (predictions): (group_size, beam_size)
                    preds_lp = beam_decoded['class_log_probabilities']
                    scorers = envs

                    # shape: List[len=batch_size] -> each element is a tensor of rewards/log_prob of size beam_size
                    per_sample_reward = []
                    per_sample_log_prob = []
                    for batch_index in range(batch_size):
                        env = scorers[batch_index]
                        to_score = preds.index_select(0, util.move_to_device(torch.tensor([batch_index]),
                                                                             self.cuda_device)).squeeze(0)
                        sp_rewards = []
                        for trace in to_score:
                            sp_rewards.append(env.step_reward(list(trace), True))

                        per_sample_reward.append(util.move_to_device(torch.tensor(sp_rewards), self.cuda_device))
                        per_sample_log_prob.append(preds_lp
                                                   .index_select(0, util.move_to_device(torch.tensor([batch_index]),
                                                                                        self.cuda_device)).squeeze(0))
                    # The prob distribution of q(theta) is computed by normalising prob distribution of top-N samples
                    # sampled from original distribution, p(theta) over transformation programs. In our case, we have
                    # sampled using beam_sample which already provides normalised log prob of each program.
                    for pred_lpbs, pred_rewards in zip(per_sample_log_prob, per_sample_reward):
                        batch_reward += self.reward_comb_fn(pred_lpbs, pred_rewards)

                    # To maximise the reward, we pass its negative as loss and minimise the loss
                    output_dict = {
                        'loss': -batch_reward
                    }

            elif self.training_signal == "supervised":
                # If target tokens are provided, compute the cross-entropy loss in output_dict = {"loss": loss}
                if target_tokens:
                    output_dict = self._forward_loss(state, target_tokens)

            else:
                raise NotImplementedError("Unknown training signal")
        else:
            output_dict = {}

        # For evaluation during test time (during validation, we run no_loss_validate_decoder to compute custom metrics)
        # use beam search to get top-k predictions and their log-probabilities.
        if not self.training and self.evaluating:
            # Decode using beam_search
            beam_decoded = self._beam_sample(model_state)
            # output_dict.update(beam_decoded)

            # We may not provide target_tokens for some test cases like Flash-Fill

            batch_size, _ = beam_decoded['class_log_probabilities'].size()
            predictions = beam_decoded['predictions']
            # preds_lp = beam_decoded['class_log_probabilities']

            # Retrieve top_K predictions. preds_lp are already in decreasing order.
            # top_k_predictions = predictions.narrow(1, 0, self.top_k)
            top_k = self.val_beam_size
            top_k_predictions = predictions.narrow(1, 0, top_k)
            # topk_preds_lp = preds_lp.narrow(1, 0, self.top_k)

            nb_correct = [[0 for _ in range(top_k)] for _ in range(batch_size)]
            nb_semantic_correct = [[0 for _ in range(top_k)] for _ in range(batch_size)]
            nb_syntax_correct = [[0 for _ in range(top_k)] for _ in range(batch_size)]
            nb_generalize_correct = [[0 for _ in range(top_k)] for _ in range(batch_size)]
            total_nb = 0
            parsed_prg = [[None for _ in range(top_k)] for _ in range(batch_size)]

            index_to_token_vocab = self._vocab.get_index_to_token_vocabulary(self._target_namespace)

            for batch_index in range(batch_size):
                total_nb += 1

                # shape: (beam_size, num_decoding_steps)
                per_sample_predictions = top_k_predictions.\
                    index_select(0, util.move_to_device(torch.tensor([batch_index]), self.cuda_device)).squeeze(0)
                for rank, prediction in enumerate(per_sample_predictions):

                    # First Decode each prediction of the Beam to tokens from op_token_table
                    decoded_pred: List[int] = []
                    for token_id in prediction:
                        token = index_to_token_vocab[token_id.item()]
                        if token == START_SYMBOL or token == "@@PADDING@@" or token == "@@UNKNOWN@@":
                            continue
                        elif token == END_SYMBOL:
                            break
                        decoded_pred.append(int(token))

                    # Exact Matches: determine whether generated program matches the GT program
                    if target_tokens is not None:
                        if decoded_pred == expected_progs[batch_index]:
                            nb_correct[batch_index][rank] += 1

                    # 1. Correct Syntax?
                    parse_success, candidate_prog = self.simulator.get_prog_obj(decoded_pred)
                    if not parse_success:
                        continue
                    nb_syntax_correct[batch_index][rank] += 1
                    parsed_prg[batch_index][rank] = candidate_prog

                    # 2. Consistent? (Semantic matches).
                    semantically_correct = True
                    for inp_string, out_string in zip(train_input_strings[batch_index],
                                                      train_output_strings[batch_index]):
                        xform_string_obj = self.simulator.run_prog(candidate_prog, inp_string)
                        if xform_string_obj.status != "OK" or xform_string_obj.xformed_string != out_string:
                            semantically_correct = False
                            break
                    if not semantically_correct:
                        continue
                    nb_semantic_correct[batch_index][rank] += 1

                    # 3. Check for Generalisation.
                    generalizes = True
                    if test_input_strings and test_output_strings:
                        for inp_string, out_string in zip(test_input_strings[batch_index],
                                                          test_output_strings[batch_index]):
                            xform_string_obj = self.simulator.run_prog(candidate_prog, inp_string)
                            if xform_string_obj.status != "OK" or xform_string_obj.xformed_string != out_string:
                                generalizes = False
                                break
                        if not generalizes:
                            continue
                        nb_generalize_correct[batch_index][rank] += 1

            evaluated_metrics = {
                "total_nb": total_nb,
                "nb_correct": nb_correct,
                "nb_semantic_correct": nb_semantic_correct,
                "nb_syntax_correct": nb_syntax_correct,
                "nb_generalize_correct": nb_generalize_correct
            }
            output_dict.update(evaluated_metrics)
            output_dict.update({
                'parsed_prog': parsed_prg
            })
            # output_dict.update({
            #     "top_predictions": top_k_predictions.detach()
            # })

        return output_dict

    def no_loss_validate_decoder(self, model_state: Dict[str, torch.LongTensor],
                                 train_input_strings: Dict[str, torch.LongTensor],
                                 train_output_strings: Dict[str, torch.LongTensor],
                                 test_input_strings: Dict[str, torch.LongTensor],
                                 test_output_strings: Dict[str, torch.LongTensor],
                                 target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, List[int]]:

        cuda_device = util.get_device_of(model_state['output_encoder_outputs_mask'])
        # We do not provide target_tokens for test cases from Flash-Fill
        expected_progs: List = []
        if target_tokens:
            expected_progs = self.decode_trg_prog_encoding(target_tokens['tokens'])
        train_input_strings = self.decode_string_encoding(train_input_strings['tokens'], self._num_examples)
        train_output_strings = self.decode_string_encoding(train_output_strings['tokens'], self._num_examples)
        if test_input_strings and test_output_strings:
            test_input_strings = self.decode_string_encoding(test_input_strings['tokens'], self._num_held_out)
            test_output_strings = self.decode_string_encoding(test_output_strings['tokens'], self._num_held_out)

        # Initialise decoder initial state such that it is equal to dictionary containing hidden and context vector
        decoder_init_state = self._decoder_net.init_decoder_state(model_state)

        # Update the 'decoder_state' and 'decoder_context' of state dictionary with initialised values
        model_state.update(decoder_init_state)

        # Decode using beam_search
        beam_decoded = self._beam_sample(model_state)
        batch_size, _ = beam_decoded['class_log_probabilities'].size()
        predictions = beam_decoded['predictions']
        # preds_lp = beam_decoded['class_log_probabilities']

        # Retrieve top_K predictions. preds_lp are already in decreasing order.
        top_k_predictions = predictions.narrow(1, 0, self.top_k)
        # topk_preds_lp = preds_lp.narrow(1, 0, self.top_k)

        # Will use this with batch size of 1
        nb_correct = [0 for _ in range(self.top_k)]
        nb_semantic_correct = [0 for _ in range(self.top_k)]
        nb_syntax_correct = [0 for _ in range(self.top_k)]
        nb_generalize_correct = [0 for _ in range(self.top_k)]
        total_nb = 0

        index_to_token_vocab = self._vocab.get_index_to_token_vocabulary(self._target_namespace)

        for batch_index in range(batch_size):
            total_nb += 1

            # shape: (beam_size, num_decoding_steps)
            per_sample_predictions = top_k_predictions.index_select(0, util.move_to_device(torch.tensor([batch_index]),
                                                                                           cuda_device)).squeeze(0)

            for rank, prediction in enumerate(per_sample_predictions):

                # First Decode each prediction of the Beam to tokens from op_token_table
                decoded_pred: List[int] = []
                for token_id in prediction:
                    token = index_to_token_vocab[token_id.item()]
                    if token == START_SYMBOL or token == "@@PADDING@@" or token == "@@UNKNOWN@@":
                        continue
                    elif token == END_SYMBOL:
                        break
                    decoded_pred.append(int(token))

                # Exact Matches: determine whether generated program matches the GT program
                if target_tokens is not None:
                    if decoded_pred == expected_progs[batch_index]:
                        nb_correct[rank] += 1

                # Correct Syntax?
                parse_success, candidate_prog = self.simulator.get_prog_obj(decoded_pred)
                if not parse_success:
                    continue
                nb_syntax_correct[rank] += 1

                # Consistent? (Semantic matches)
                semantically_correct = True
                for inp_string, out_string in zip(train_input_strings[batch_index],
                                                  train_output_strings[batch_index]):
                    xform_string_obj = self.simulator.run_prog(candidate_prog, inp_string)
                    if xform_string_obj.status != "OK" or xform_string_obj.xformed_string != out_string:
                        semantically_correct = False
                        break
                if not semantically_correct:
                    continue
                nb_semantic_correct[rank] += 1

                # Check for Generalisation
                generalizes = True
                if test_input_strings and test_output_strings:
                    for inp_string, out_string in zip(test_input_strings[batch_index],
                                                      test_output_strings[batch_index]):
                        xform_string_obj = self.simulator.run_prog(candidate_prog, inp_string)
                        if xform_string_obj.status != "OK" or xform_string_obj.xformed_string != out_string:
                            generalizes = False
                            break
                    if not generalizes:
                        continue
                    nb_generalize_correct[rank] += 1

        return {
            "total_nb": total_nb,
            "nb_correct": nb_correct,
            "nb_semantic_correct": nb_semantic_correct,
            "nb_syntax_correct": nb_syntax_correct,
            "nb_generalize_correct": nb_generalize_correct
        }

    @overrides
    def post_process(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called `predicted_tokens` to the `output_dict`.
        """
        predicted_indices = output_dict["predictions"]
        all_predicted_tokens = self.indices_to_tokens(predicted_indices)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def indices_to_tokens(self, batch_indeces) -> List[List[str]]:

        if not isinstance(batch_indeces, numpy.ndarray):
            batch_indeces = batch_indeces.detach().cpu().numpy()

        all_tokens = []
        for indices in batch_indeces:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[: indices.index(self._end_index)]
            tokens = [
                self._vocab.get_token_from_index(x, namespace=self._target_namespace)
                for x in indices
            ]
            all_tokens.append(tokens)

        return all_tokens

    def decode_trg_prog_encoding(self, target_tokens: torch.LongTensor, ) -> List[List[int]]:
        """
        Decode program tokens into executable transformation program
        :param target_tokens:
        :return:
        """
        target_tokens = target_tokens.detach()

        all_prog = []
        for prog_enc in target_tokens[::self._num_examples]:
            prog_seq = [
                int(self._vocab.get_token_from_index(token_id.item(), self._target_namespace))
                for token_id in prog_enc
                if self._vocab.get_token_from_index(token_id.item(), self._target_namespace) not in self._ignore_symbols
            ]
            all_prog.append(prog_seq)
        return all_prog

    def decode_string_encoding(self, string_encs: torch.LongTensor, num_examples: int) -> List[List[str]]:
        """
        Decodes string encodings back into their original str form. Groups consecutive 'num_examples' strings together
        :param string_encs: self-explanatory
        :param num_examples: number of strings (denoted by m in paper) to be used for decoding a transformation program
        :return: [[str_1,str_2...str_m][.....]]
        """

        all_strings = []
        for string_enc in string_encs:
            string = ''.join([
                self._vocab.get_token_from_index(token_id.item())
                for token_id in string_enc
                if self._vocab.get_token_from_index(token_id.item()) not in self._ignore_symbols
            ])
            all_strings.append(string)

        string_set_form = [
            all_strings[i: i + num_examples]
            for i in range(0, len(all_strings), num_examples)
        ]

        return string_set_form


# Implementing each call to an LSTM cell. Attend to input_string_encoder, output_string_encoder and Concatenate with
# previous time stamp's prediction, y_hat/GT , y. Generate the new hidden state and context vector.
@DecoderNet.register("rl_lstm_cell_double_attention")
class LstmCellDecoderNetDoubleAttention(DecoderNet):
    """
    This decoder net implements simple decoding network with LSTMCell and Double Attention.

    # Parameters

    hidden_size : `int`, required
        Defines dimensionality of output vectors.
    input_embed_size : `int`, required
        Defines dimensionality of input target embeddings.  Since this model takes it's output on a previous step
        as input of following step, this is also an input dimensionality.
    attention_input_encoder : `Attention`,
        If you want to use attention to get a dynamic summary of the input encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and input encoder outputs.
    attention_output_encoder : `Attention`,
        If you want to use attention to get a dynamic summary of the output encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and output encoder outputs.
    """

    def __init__(
        self,
        input_embed_size: int,
        hidden_size: int,
        bidirectional: bool,
        attention_input_encoder: Attention,
        attention_output_encoder: Attention,
    ) -> None:
        self.bidirectional = bidirectional
        self._num_directions = 2 if self.bidirectional else 1
        self.hidden_size = hidden_size*self._num_directions
        super().__init__(
            decoding_dim=self.hidden_size,
            target_embedding_dim=input_embed_size,
            decodes_parallel=False,
        )
        # 2*hidden_size since double attention is used.
        self.input_dim = input_embed_size+(2*self.hidden_size)
        # Attention mechanism applied to the btoh input and output encoders' output for each step.
        self.attention_input_encoder = attention_input_encoder
        self.attention_output_encoder = attention_output_encoder
        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        self._decoder_cell = LSTMCell(self.input_dim, self.hidden_size)

    # Attend to outputEncoder_outputs using decoder_hidden_state and attend to inputEncoder_outputs using concatentaed
    # decoder_hidden_state and attended_outputEncoder.
    # <Output> = <_batch_size, inputEncoder_output_dim + outputEncoder_output_dim>
    def _prepare_attended_input(
        self,
        state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply attention over both encoder outputs and decoder state."""
        # Compute weights for output encoder | shape: (_batch_size, max_sequence_length)
        att_output_encoder = self.attention_output_encoder(state["decoder_hidden"],
                                                           state['output_encoder_outputs'],
                                                           state['output_encoder_outputs_mask'])
        # Attend to output encoder | shape: (_batch_size, outputEncoder_output_dim)
        att_output_encoder = util.weighted_sum(state['output_encoder_outputs'], att_output_encoder)

        # Compute weights for input encoder | shape: (_batch_size, max_sequence_length)
        att_input_encoder = self.attention_input_encoder(torch.cat((state["decoder_hidden"], att_output_encoder), -1),
                                                         state['input_encoder_outputs'],
                                                         state['input_encoder_outputs_mask'])
        # Attend to input encoder | shape: (_batch_size, inputEncoder_output_dim)
        att_input_encoder = util.weighted_sum(state['input_encoder_outputs'], att_input_encoder)

        return torch.cat((att_output_encoder, att_input_encoder), -1)

    @overrides
    def init_decoder_state(
        self, encoder_out: Dict[str, torch.LongTensor]
    ) -> Dict[str, torch.Tensor]:

        batch_size = encoder_out["output_encoder_outputs_mask"].size(0)

        # Initialize the decoder hidden state with the final output of the encoder,
        # and the decoder context with zeros.
        # shape: (_batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
            encoder_out["output_encoder_outputs"],
            encoder_out["output_encoder_outputs_mask"],
            bidirectional=self.bidirectional,
        )

        return {
            "decoder_hidden": final_encoder_output,  # shape: (_batch_size, decoder_output_dim)
            "decoder_context": final_encoder_output.new_zeros(batch_size, self.decoding_dim)
            #                  shape: (_batch_size, decoder_output_dim)
        }

    @overrides
    def forward(
        self,
        state: Dict[str, torch.Tensor],
        last_predictions_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        # TODO: Debug
        # stat_cuda("3a) DecoderNet.forward | Calling _prepare_attended_input")

        # Create attended input | shape: (group_size, inputEncoder_output_dim + outputEncoder_output_dim
        decoder_input = self._prepare_attended_input(state=state)

        # TODO: Debug
        # stat_cuda("3b) DecoderNet.forward | attended_input prepared")

        # Concat attended input with last time-step's predictions |
        # shape: (group_size, (inputEncoder_output_dim + outputEncoder_output_dim) + target_embedding_dim)
        decoder_input = torch.cat((last_predictions_embedding[:, -1], decoder_input), -1)

        # Compute new hidden & context | shape (decoder_context, decoder_hidden): (_batch_size, decoder_output_dim)
        state["decoder_hidden"], state["decoder_context"] = self._decoder_cell(
            decoder_input, (state["decoder_hidden"], state["decoder_context"])
        )

        # TODO: Debug
        # stat_cuda("3c) DecoderNet.forward | New decode_cell state generated")
        # check_currently_active_tensors("3c) DecoderNet.forward | Before | Exiting DecoderNet")

        return state
