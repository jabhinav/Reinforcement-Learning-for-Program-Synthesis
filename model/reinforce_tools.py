from model import operators as op
from model.tokens import build_token_tables
from typing import Dict, List, Tuple, Optional
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.models.model import Model
import copy
# import torchcl
import torch.nn.functional as F
import torch
import gc


def check_currently_active_tensors(msg):
    """
    Print what tensors currently reside in the memory. For Debugging.
    :param msg:
    """
    print('--', msg)
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


def stat_cuda(msg):
    """
    Print state of the memory. For Debugging.
    :param msg:
    """
    print('--', msg)
    print('allocated: %dM, max allocated: %dM, cached: %dM, max cached: %dM' % (
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_cached() / 1024 / 1024,
        torch.cuda.max_memory_cached() / 1024 / 1024
    ))


def debug_memory(msg, device=None):
    print("\n --- Debug --- {}".format(msg))
    print("GPU memory occupied: {} GB".format(torch.cuda.memory_allocated() / (1024 ** 3)))
    print(torch.cuda.memory_summary())


def check_tensor_memory(x: torch.Tensor, name: str = "x"):
    print("\n--- Debug --- Tensor {} occupies {} MB".format(name, x.element_size()*x.nelement() / (1024**2)))


class progEvalObj():
    def __init__(self, xformed_string, status):
        self.xformed_string = xformed_string
        self.status = status


class ProgParseException(BaseException):
    def __init__(self, error_message):
        self.error_message = error_message

    def __str__(self):
        return self.error_message


class ProgParser(object):
    """
    Example Usage:
    prog = [Concat, SubStr, 2, 4, ConstStr, “a”]
    parser = ProgParser()
    executable_prog = parser.parse(prog) <- Concat(SubStr(2,4), ConstStr('a'))
    inputString = “AFGDTH”
    outputString = executable_prog.eval(inputString)
    The output generated will be “GDa”
    """

    def __init__(self):
        token_table = build_token_tables()
        self.token_op_table = token_table.token_op_table

    def generate_prog_from_tokens(self, prg_tokens):
        return [self.token_op_table[token] for token in prg_tokens]

    def parse(self, program):
        """
        Parse an un-padded and decoded sequence of program tokens else raise exception
        :param program: Seq of tokens (integers) or seq of operators/operations (strings) (both types allowed)
        :return: op.Concat object implementing the program decoded
        """
        # If program is a seq of tokens (integers), convert them into seq of operators/operations
        # print("Parsing program: {}".format(program))
        # if all(isinstance(x, int) for x in program):
        #     program = self.generate_prog_from_tokens(program)

        program = copy.deepcopy(program)

        if len(program) > 1 and program[-1] == 'EOS':
            program.pop()

        index = 0
        # num_func_args = 0
        expressions = []
        string_expressions = []

        while index < len(program):

            operation = program[index]

            ############################
            #   SubString Expression   #
            ############################

            # Function Parsed -> Substr(int: pos1, int: pos2), pos between op.POSITION
            if operation == op.SubStr:
                num_func_args = 2
                if index + num_func_args < len(program):

                    # # Check if next two are positional arguments i.e. integers
                    if isinstance(program[index + 1], int) and isinstance(program[index + 2], int):
                        func_obj = op.SubStr(program[index + 1], program[index + 2])
                        expressions.append(func_obj)
                    else:
                        self.__raise_exception(program, "Positional arguments {}, {} are not integers".
                                               format(program[index + 1], program[index + 2]))
                        break
                else:
                    self.__raise_exception(program, "Not enough arguments for {}".format(operation))
                    break

            # Function Parsed -> GetSpan(op.TYPE/str: dsl_regex1, int: index1, op.BOUNDARY: boundary1,
            #                            op.TYPE/str: dsl_regex1, int: index2, op.BOUNDARY: boundary2)
            elif operation == op.GetSpan:
                num_func_args = 6
                if index + num_func_args < len(program):
                    # # Check next six arguments i.e. dsl_regex1, index1, boundary1, dsl_regex2, index2, boundary2
                    if (isinstance(program[index + 1], str) or isinstance(program[index + 1], op.Type)) \
                            and isinstance(program[index + 2], int) \
                            and isinstance(program[index + 3], op.Boundary) \
                            and (isinstance(program[index + 4], str) or isinstance(program[index + 4], op.Type)) \
                            and isinstance(program[index + 5], int) \
                            and isinstance(program[index + 6], op.Boundary):
                        if program[index+1] in (list(op.Type) + list(op.DELIMITER)) and \
                                program[index+2] in list(op.INDEX) and \
                                program[index+3] in list(op.Boundary) and \
                                program[index+4] in (list(op.Type) + list(op.DELIMITER)) and \
                                program[index+5] in list(op.INDEX) and \
                                program[index+6] in list(op.Boundary):
                            func_obj = op.GetSpan(*program[index + 1:index + 7])
                            expressions.append(func_obj)
                        else:
                            self.__raise_exception(program, "GetSpan Arguments: '{}' not in defined range".format(
                                program[index + 1:index + 7]))
                            break
                    else:
                        self.__raise_exception(program, "GetSpan Arguments: '{}' not of expected type".format(
                            program[index + 1:index + 7]))
                        break
                else:
                    self.__raise_exception(program, "Not enough arguments for {}".format(operation))
                    break

            ###########################
            #   ConstStr Expression   #
            ###########################

            # Function Parsed -> ConstStr(str: char), where char should be in op.CHARACTER
            elif operation == op.ConstStr:
                num_func_args = 1
                if index + num_func_args < len(program):
                    # # Check if next element is a character from the specified list
                    if isinstance(program[index + 1], str):
                        if program[index + 1] in op.CHARACTER:
                            func_obj = op.ConstStr(program[index + 1])
                            expressions.append(func_obj)
                        else:
                            self.__raise_exception(program, "Constant String argument '{}' not in defined set of "
                                                            "Characters".format(program[index + 1]))
                            break
                    else:
                        self.__raise_exception(program, "Expected Argument for ConstStr should be of type 'str' but "
                                                        "found: {}".format(type(program[index+1])))
                        break
                else:
                    self.__raise_exception(program, "Not enough arguments for {}".format(operation))
                    break

            ##################################
            #        Nested Expressions      #
            ##################################

            # tuple = <function, arg1, arg2 ..> Includes GetToken, GetUpto, GetFrom, GetFirst, GetAll, ToCase, Replace
            elif isinstance(operation, tuple):
                num_func_args = 0
                func_obj = operation[0](*operation[1:])
                expressions.append(func_obj)

            # Function parsed -> Trim(), removes trailing and leading whitespaces
            elif operation == op.Trim:
                num_func_args = 0
                func_obj = op.Trim()
                expressions.append(func_obj)

            ###############################
            #      Concat Operation       #
            ###############################
            # While tokenizing a transformation program, Concat operation is tokenized by adding its token between
            # string expressions passed as its arguments. Rules:-
            # 1. If two string expressions are followed by a Concat operation, then use Compose to allow nesting
            # 2. In GT programs, each string expression (other than Compose) is followed by a Concat token.
            # 3. But we will allow multiple (>=1, except 2) to be concatenated during program construction & evaluation
            elif operation == op.Concat:
                num_func_args = 0
                if len(expressions) == 2:
                    string_expressions.append(op.Compose(*expressions))
                # elif len(expressions) == 1:
                #     string_expressions.extend(expressions)
                else:
                    string_expressions.extend(expressions)
                expressions = []

            else:
                self.__raise_exception(program, "Unknown command: {0}".format(program[index]))
                break

            index += (num_func_args + 1)

        ###########################
        #      Reaching EOS       #
        ###########################
        if len(expressions) == 2:
            string_expressions.append(op.Compose(*expressions))
        else:
            string_expressions.extend(expressions)

        return op.Concat(*string_expressions)

    @staticmethod
    def __raise_exception(t, error_message):
        # msg = "Error parsing tokens: '{0}'. Error message: {1}".format(" ".join(t), error_message)
        msg = "Error parsing tokens: '{0}'. Error message: {1}".format(t, error_message)
        raise ProgParseException(msg)


class Simulator(object):
    def __init__(self, token_op_table):
        super(Simulator, self).__init__()
        self.token_op_table = token_op_table
        self.prog_parser = ProgParser()

    def generate_prog_from_tokens(self, prg_tokens):
        return [self.token_op_table[token] for token in prg_tokens]

    def get_prog_obj(self, prg_tokens):
        """
        Parse the program.
        :param prg_tokens: List of program token IDs
        :return: Parsed Program in executable form if syntactically correct, else None is returned
        """
        prg_ops = self.generate_prog_from_tokens(prg_tokens)
        try:
            prog_obj = self.prog_parser.parse(prg_ops)
        except ProgParseException:
            return False, None
        # prog_ast = Ast(prg_ast_json)
        return True, prog_obj

    def run_prog(self, prog_obj, inp_string):
        """
        Runs the program on the input string and wraps the output as a separate object.
        The returned object has a 'status' to check if generated program raised any error while transforming
        the input string.
        :param prog_obj:
        :param inp_string:
        :return:
        """
        try:
            xformed_string = prog_obj.eval(inp_string)
        # Modify to catch errors specific to eval() only to avoid catching errors like keyboard_interrupt etc.
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            return progEvalObj(None, status="FAIL")

        xformed_string_obj = progEvalObj(xformed_string, status="OK")
        return xformed_string_obj


class Rolls(object):

    def __init__(self, action, proba, multiplicity, depth):
        self.successor = {}
        # The action that this node in the tree corresponds to
        self.action = action  # -> what was the sample taken here
        self.proba = proba  # -> Variable containing the proba of taking this action
        self.multi_of_this = multiplicity  # -> How many times was this prefix (until this point of the sequence) seen
        self.depth = depth  # -> How far along are we in the sequence

        # Has no successor to this sample
        self.is_final = True

        # This is the reward that is obtained by evaluating current trajectory's prefix once.
        # [ This is only to use for bookkeeping.]
        # In our case, if prefix is partial, then its own_reward is zero (no reward is collected).
        # If prefix is complete i.e. end token is sampled, we evaluate it on input strings and collect the reward
        self.own_reward = 0

        # The one to be used for compute gradients is the following:
        # This contains `self.own_reward * self.multi_of_this` + sum of all the dependents' self.dep_reward
        self.dep_reward = 0

    def expand_samples(self, trajectory, end_multiplicity, end_proba):
        """
        The assumption here is that all but the last steps of the trajectory
        have already been created.
        You pick the first action from the trajectory, see whether it exists as a successor.
        If not, initialise the action with a Roll object, else, recursively expand.
        """
        assert(len(trajectory) > 0)

        pick = trajectory[0]
        if pick in self.successor:
            self.successor[pick].expand_samples(trajectory[1:],
                                                end_multiplicity,
                                                end_proba)
        else:
            # We add a successor so we are necessarily not final anymore
            self.is_final = False
            # We don't expand the samples by several steps at a time so verify
            # that we are done
            assert(len(trajectory) == 1)
            self.successor[pick] = Rolls(pick, end_proba,
                                         end_multiplicity,
                                         self.depth + 1)

    def yield_final_trajectories(self):
        """
        Yields 3-tuples:
        -> Trajectory
        -> Multiplicity of this trajectory
        -> Proba of this trajectory
        -> Final reward of this trajectory
        """
        if self.is_final:
            yield [], self.multi_of_this, self.proba, self.own_reward
        else:
            for key, succ in self.successor.items():
                for final_traj, multi, proba_suffix, reward in succ.yield_final_trajectories():
                    yield ([key] + final_traj,
                           multi,
                           self.proba * proba_suffix,
                           reward)

    def yield_var_and_grad(self):
        """
        Yields 2-tuples:
        -> Proba: Variable corresponding to the probability of this last choice
        -> Grad: Gradients for each of those variables
        """
        for succ in self.successor.values():
            for var, grad in succ.yield_var_and_grad():
                yield var, grad
        yield self.proba, self.reinforce_gradient()

    def assign_rewards(self, reward_assigner, trace):
        """
        Using the `reward_assigner` scorer, go depth first to assign the
        reward at each timestep, and then collect back all the "depending
        rewards"
        """
        if self.depth == -1:
            # This is the root from which all the samples come from, ignore
            pass
        else:
            # Assign to this step its own reward. If this step is not final (~partial trajectory), reward will be 0.
            # Reward will be non-zero (+norm_reward or -norm_reward) only if the current step is final.
            self.own_reward = reward_assigner.step_reward(trace, self.is_final)

        # Assign their own score to each of the successor
        for next_step, succ in self.successor.items():
            new_trace = trace + [next_step]
            succ.assign_rewards(reward_assigner, new_trace)

        # If this is a final node, there is no successor, so I can already
        # compute the dep-reward.
        if self.is_final:
            self.dep_reward = self.multi_of_this * self.own_reward
        else:
            # On the other hand, all my child nodes have already computed their
            # dep_reward so I can collect them to compute mine
            self.dep_reward = self.multi_of_this * self.own_reward
            for succ in self.successor.values():
                self.dep_reward += succ.dep_reward

    def reinforce_gradient(self):
        """
        At each decision, compute a reinforce gradient estimate to the
        parameter of the probability that was sampled from.
        """
        if self.depth == -1:
            return None
        else:
            # We haven't put in a baseline so just ignore this.
            # Ideally it would have been baselined_reward = (dep_reward - baseline_reward)
            # The grad value = 1/S * R(.) * grad(log(prob)) = normalised_reward * 1/(prob)
            baselined_reward = self.dep_reward
            grad_value = baselined_reward / (1e-6 + self.proba.data)

            # We return a negative here because we want to maximize the rewards
            # And the pytorch optimizers try to minimize them, so this puts them
            # in agreement
            return -grad_value


class Environment(object):

    def __init__(self, reward_norm, environment_data):
        """
        reward_norm: float -> Value of the reward for correct answer
        environment_data: anything -> Data/Ground Truth to be used for the reward evaluation


        Note: To create different types of reward, subclass it and modify the
        `should_skip_reward` and `reward_value` function.
        """
        self.reward_norm = reward_norm
        self.environment_data = environment_data

    def step_reward(self, trace, is_final):
        """
        trace: List[int] -> all prediction of the sample to score.
        is_final: bool -> Is the sample finished.
        Return the reward only if the sample is finished.
        """
        if self.should_skip_reward(trace, is_final):
            return 0
        else:
            return self.reward_value(trace, is_final)

    def should_skip_reward(self, trace, is_final):
        raise NotImplementedError

    def reward_value(self, trace, is_final):
        raise NotImplementedError


class MultiIO01(Environment):
    """
    This only gives rewards at the end of the prediction.
    +1 if the two programs lead to same outputs.
    -1 if the two programs lead to different outputs
    """
    def __init__(self, reward_norm: float, target_program: List[int],
                 input_strings: List[List[str]], output_strings: List[List[str]],
                 simulator: Simulator, network_vocab: Dict[int, str]):
        """
        reward_norm: float -> Value of the reward for correct answer
        target_prog = sequence of integers representing operations/operands from token_table not from network_vocab
        input_strings, output_strings: Reference IO specification for program synthesis
        """
        super(MultiIO01, self).__init__(reward_norm,
                                        (target_program,
                                         input_strings,
                                         output_strings,
                                         simulator))
        self.target_program = target_program
        self.input_strings = input_strings
        self.output_strings = output_strings
        self.simulator = simulator
        self._vocab = network_vocab

        # Parse the target program and check whether it is syntactically correct i.e. parse_success == True
        parse_success, ref_prog = self.simulator.get_prog_obj(self.target_program)
        if not parse_success:
            print("\nThe target Program {} is not parsable on string pairs:-\nInput: {}\nOutput:{}".format(
                self.target_program, self.input_strings, self.output_strings))
        assert parse_success

        # Make sure that the reference program works for the given IO
        self.correct_reference = True
        for inp_string, out_string in zip(self.input_strings, self.output_strings):

            xform_string_obj = self.simulator.run_prog(ref_prog, inp_string)
            self.correct_reference = self.correct_reference and (xform_string_obj.status == 'OK')
            self.correct_reference = self.correct_reference and (out_string == xform_string_obj.xformed_string)

    def should_skip_reward(self, trace, is_final):
        return not is_final

    def decode_trace(self, trace: List[torch.Tensor]) -> List[int]:
        # do not detach the trace since during training time, we need to back-propagate its gradients - Right???
        # print("Trace to be decoded for associated reward generation: {}".format(trace))
        # print("Corresponding vocab output: {}".format([self._vocab[token_id.item()] for token_id in trace]))
        decoded_trace = []
        for token_id in trace:
            token = self._vocab[token_id.item()]
            if token == START_SYMBOL or token == "@@PADDING@@" or token == "@@UNKNOWN@@":
                continue
            elif token == END_SYMBOL:
                break
            decoded_trace.append(int(token))
        return decoded_trace

    def reward_value(self, trace, is_final):
        """
        Generate Reward for a sample program. The trace to be evaluate should be unpadded.
        :param trace: sample program sequence(of tensors) consisting of tokens from AllenNLP vocab (needs to be decoded)
        :param is_final:
        :return: reward value
        """
        if not self.correct_reference:
            # There is some problem with the data because the reference program
            # crashed. Ignore it.
            return 0

        rew = 0
        # trace = [token.item() for token in trace]
        trace = self.decode_trace(trace)
        parse_success, cand_prog = self.simulator.get_prog_obj(trace)
        if not parse_success:
            # Program is not syntactically correct. Un-normalised Reward = -1
            rew = -self.reward_norm
        else:
            for inp_string, out_string in zip(self.input_strings, self.output_strings):
                xform_string_obj = self.simulator.run_prog(cand_prog, inp_string)
                # if res_emu.status != 'OK' or res_emu.crashed:
                if xform_string_obj.status != "OK":

                    # Crashed or failed the simulator
                    # Set the reward to negative and stop looking
                    rew = -self.reward_norm
                    break
                elif xform_string_obj.xformed_string != out_string:
                    # Generated a wrong state
                    # Set the reward to negative and stop looking
                    rew = -self.reward_norm
                    break
                else:
                    rew = self.reward_norm
        return rew


EnvironmentClasses = {
    "BlackBoxGeneralization": MultiIO01,
    "BlackBoxConsistency": MultiIO01,
    # "PerfRewardMul": PerfRewardMul,
    # "PerfRewardDiff": PerfRewardDiff
}


def batch_rolls_reinforce(rolls):
    for roll in rolls:
        for var, grad in roll.yield_var_and_grad():
            if grad is None:
                assert var.requires_grad is False
            else:
                yield var, grad


def expected_rew_renorm(prediction_lpbs, prediction_reward):
    """
    Simplest Reward Combination Function
    Takes as input:
    `prediction_lpbs`: The log probabilities of each sampled programs
    `prediction_reward_list`: The reward associated with each of these
                              sampled programs.
    Returns the expected reward under the (renormalized so that it sums to 1)
    probability distribution defined by prediction_lbps.
    """

    # # Method 1:
    # pbs = prediction_lpbs.exp()
    # pb_sum = pbs.sum()
    # pbs = pbs.div(pb_sum.expand_as(pbs))

    # Method 2:
    prediction_pbs = F.softmax(prediction_lpbs, dim=0)  # Use exponential from softmax to convert log_prob back to prob
    return torch.dot(prediction_pbs, prediction_reward)


def n_samples_expected_1m1rew(nb_samples_in_bag):
    """Generates a Reward Combination Function
    based on sampling with replacement `nb_samples_in_bag` programs from the
    renormalized probability distribution and keeping the one with the best
    reward."""
    def fun(prediction_lpbs, prediction_reward):
        """Takes as input:
        `prediction_lpbs`: The log probabilities of each sampled programs
        `prediction_reward_list`: The reward associated with each of these
                                  sampled programs, assumed to be 1 or -1
        Returns the expected reward when you sample with replacement
        `nb_samples_in_bag` programs from the (renormalized) probability
        distribution defined by prediction_lbps and keep the best reward
        out of those `nb_samples_in_bag`."""
        prediction_pbs = F.softmax(prediction_lpbs, dim=0)
        negs_mask = (prediction_reward == -1)
        prob_negs = prediction_pbs.masked_select(negs_mask)
        prob_of_neg_rew_per_sp = prob_negs.sum()

        prob_of_neg_rew_for_bag = prob_of_neg_rew_per_sp.pow(nb_samples_in_bag)
        prob_of_pos_rew_for_bag = 1 - prob_of_neg_rew_for_bag

        expected_bag_rew = prob_of_pos_rew_for_bag - prob_of_neg_rew_per_sp
        return expected_bag_rew
    return fun


RewardCombinationFun = {
    "RenormExpected": expected_rew_renorm
}
for bag_size in [5, 50]:
    key_name = str(bag_size) + "1m1BagExpected"
    RewardCombinationFun[key_name] = n_samples_expected_1m1rew(bag_size)


def get_custom_metrics(model: Model, total_loss: float, num_batches: int, reset: bool = False) -> Dict[str, float]:
    """
    Gets the metrics but sets ``"loss"`` to
    the total loss divided by the ``num_batches`` so that
    the ``"loss"`` metric is "average loss per batch".
    """
    metrics = model.get_metrics(reset=reset)
    metrics["loss"] = float(total_loss / num_batches) if num_batches > 0 else 0.0
    return metrics
