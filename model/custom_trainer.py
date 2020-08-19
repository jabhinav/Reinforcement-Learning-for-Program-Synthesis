# import logging
import math
import os
import time
import datetime
import traceback
from typing import Optional, Tuple, Iterable
# from typing import Dict, List, Union, Any
import torch
import torch.optim.lr_scheduler
import torch.autograd as autograd
# import torchviz

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, parse_cuda_device
from allennlp.common.util import (dump_metrics, gpu_memory_mb, peak_memory_mb,
                                  lazy_groups_of)
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.nn import util as nn_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
# from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.training.trainer_base import TrainerBase
from allennlp.training.trainer import Trainer
from allennlp.training.trainer_pieces import TrainerPieces
from allennlp.training import util as training_util
from allennlp.training.moving_average import MovingAverage

from overrides import overrides
import logging
import numpy as np
from typing import Dict, List, Union, Any

logger = logging.getLogger(__name__)
# Learning rate schedulers type: step, exponential, multi_step, reduce_on_plateau


@TrainerBase.register("rl_trainer")
class RLTrainer(Trainer):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 train_dataset: Iterable[Instance],
                 validation_dataset: Optional[Iterable[Instance]] = None,
                 patience: Optional[int] = None,
                 validation_metric: str = "-loss",
                 validation_iterator: DataIterator = None,
                 shuffle: bool = True,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 num_serialized_models_to_keep: int = 20,
                 keep_serialized_model_every_num_seconds: int = None,
                 checkpointer: Checkpointer = None,
                 model_save_interval: float = None,
                 cuda_device: Union[int, List] = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[LearningRateScheduler] = None,
                 momentum_scheduler: Optional[MomentumScheduler] = None,
                 summary_interval: int = 100,
                 histogram_interval: int = None,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = False,
                 log_batch_size_period: Optional[int] = None,
                 moving_average: Optional[MovingAverage] = None,
                 training_signal: str = "supervised",
                 rl_inner_batch: int = 1,
                 fine_tune_rf: bool = False,
                 weights_file: str = _DEFAULT_WEIGHTS) -> None:

        super().__init__(model,
                         optimizer,
                         iterator,
                         train_dataset,
                         validation_dataset,
                         patience,
                         validation_metric,
                         validation_iterator,
                         shuffle,
                         num_epochs,
                         serialization_dir,
                         num_serialized_models_to_keep,
                         keep_serialized_model_every_num_seconds,
                         checkpointer,
                         model_save_interval,
                         cuda_device,
                         grad_norm,
                         grad_clipping,
                         learning_rate_scheduler,
                         momentum_scheduler,
                         summary_interval,
                         histogram_interval,
                         should_log_parameter_statistics,
                         should_log_learning_rate,
                         log_batch_size_period,
                         moving_average)

        self.training_signal = training_signal

        # For RL based training - to save memory
        self.accumulate_grads = True
        self.rl_inner_batch = rl_inner_batch  # use default value of 1 i.e. call loss.backward() after every iter

        # We assume that weights_file contains the RobustFill model's state_dict when supervised training was used.
        # Also, self.model is initialised first, we can view its current state_dict as self.model.state_dict().
        # To load weights available in model_state, we use self.model.load_state_dict().
        # Note: model_state (loaded dictionary) and self.model.state_dict() (curr dictionary) should match.

        # Note: TrainerPieces is a helper class, where model is instantiated, data-sets are loaded, iterators and
        # vocab are created
        if training_signal == 'beam_rl' or training_signal == "rl":
            if os.path.exists("./{}".format(weights_file)):
                model_state = torch.load(weights_file, nn_util.device_mapping(cuda_device))
                self.model.load_state_dict(model_state)
                print("Model weights loaded successfully for {}".format(training_signal))
            else:
                raise ConfigurationError("For beam_rl/rl training signal only fine-tuning is supported, "
                                         "please provide a valid weights path")
        elif training_signal == "supervised":
            if fine_tune_rf:
                if os.path.exists("./{}".format(weights_file)):
                    model_state = torch.load(weights_file, nn_util.device_mapping(cuda_device))
                    self.model.load_state_dict(model_state)
                    print("Model weights loaded successfully for {}".format(training_signal))
                else:
                    raise ConfigurationError("No saved weights detected for fine-tuning the model in supervised setting"
                                             "")

    def rescale_gradients(self) -> Optional[float]:
        return training_util.rescale_gradients(self.model, self._grad_norm)

    @overrides
    def batch_loss(self, batch_group: List[TensorDict], for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batches and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        if self._multiple_gpu:
            output_dict = training_util.data_parallel(batch_group, self.model, self._cuda_devices)
        else:
            assert len(batch_group) == 1
            batch = batch_group[0]
            batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            output_dict = self.model(**batch)

        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self.model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError("The model you are trying to optimize does not contain a"
                                   " 'loss' key in the output of model.forward(inputs).")
            loss = None

        return loss

    def batch_reward(self, batch_group: List[TensorDict], for_training: bool):
        """
        Designed specifically for training signals rl and beam_rl
        :param batch_group: Batched data
        :return: Reward as the loss for back propagation
        """

        if self._multiple_gpu:
            if self.training_signal == "rl":
                # {'loss': losses.mean()} is computed in data_parallel which is not possible for rl signal. Concat reqd.
                raise ConfigurationError("Multi-GPU training currently not available for training signal 'rl'. "
                                         "Use single GPU")
            output_dict = training_util.data_parallel(batch_group, self.model, self._cuda_devices)
        else:
            assert len(batch_group) == 1
            batch = batch_group[0]
            batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            output_dict = self.model(**batch)

        if self.training_signal == "rl":
            try:
                batch_reward = output_dict["loss"]
                variables = output_dict['variables']
                grad_variables = output_dict['grad_variables']
            except KeyError:
                if for_training:
                    raise RuntimeError("The model you are trying to optimize does not contain one of the keys"
                                       " 'loss'/'variables'/'grad_variables' in the output of model.forward(inputs).")
                batch_reward = None
                variables = None
                grad_variables = None
            return batch_reward, variables, grad_variables

        elif self.training_signal == "beam_rl":
            try:
                loss = output_dict["loss"]
            except KeyError:
                if for_training:
                    raise RuntimeError("The model you are trying to optimize does not contain one of the keys"
                                       " 'loss' key in the output of model.forward(inputs).")
                loss = None
            return loss

    def no_loss_custom_validation_metrics(self) -> Dict[str, float]:
        """
        Computes the exact match accuracy, syntactic correctness, consistency and generalisation accuracy
        """
        logger.info("Validating")

        val_metrics: Dict[str, float] = {}
        self.model.eval()

        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        num_gpus = len(self._cuda_devices)
        raw_val_generator = val_iterator(self._validation_data,
                                         num_epochs=1,
                                         shuffle=False)
        val_generator = lazy_groups_of(raw_val_generator, num_gpus)
        num_validation_batches = math.ceil(val_iterator.get_num_batches(self._validation_data) / num_gpus)
        val_generator_tqdm = Tqdm.tqdm(val_generator,
                                       total=num_validation_batches)
        # batches_this_epoch = 0
        total_nb = 0
        top1_nb_correct = 0
        top1_nb_semantic_correct = 0
        top1_nb_syntax_correct = 0
        top1_nb_generalize_correct = 0
        for batch_group in val_generator_tqdm:
            if self._multiple_gpu:
                raise ConfigurationError("Evaluation metrics cannot be computed on multiple GPUs as of now")

            assert len(batch_group) == 1
            batch = batch_group[0]
            batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            computed_metric_dict = self.model.no_loss_validate_fwd(**batch)
            try:
                total_nb += computed_metric_dict["total_nb"]
                top1_nb_correct += computed_metric_dict["nb_correct"][0]
                top1_nb_semantic_correct += computed_metric_dict["nb_semantic_correct"][0]
                top1_nb_syntax_correct += computed_metric_dict["nb_syntax_correct"][0]
                top1_nb_generalize_correct += computed_metric_dict["nb_generalize_correct"][0]
            except KeyError:
                raise RuntimeError("The returned output metric dictionary does not contain one of the computed metric"
                                   "keys")

            # Update the description with the latest validation metrics (we show the running average)
            val_metrics = {
                "acc_exact_match": float(top1_nb_correct / total_nb) if total_nb > 0 else 0.0,
                "acc_consistency": float(top1_nb_semantic_correct / total_nb) if total_nb > 0 else 0.0,
                "acc_syntax": float(top1_nb_syntax_correct / total_nb) if total_nb > 0 else 0.0,
                "acc_generalisation": float(top1_nb_generalize_correct / total_nb) if total_nb > 0 else 0.0
            }
            description = training_util.description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()

        return val_metrics

    @overrides
    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self.model.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        num_gpus = len(self._cuda_devices)

        raw_val_generator = val_iterator(self._validation_data,
                                         num_epochs=1,
                                         shuffle=False)
        val_generator = lazy_groups_of(raw_val_generator, num_gpus)
        num_validation_batches = math.ceil(val_iterator.get_num_batches(self._validation_data)/num_gpus)
        val_generator_tqdm = Tqdm.tqdm(val_generator,
                                       total=num_validation_batches)
        batches_this_epoch = 0
        val_loss = 0.0
        for batch_group in val_generator_tqdm:
            if self.training_signal == "supervised":
                loss = self.batch_loss(batch_group, for_training=False)
                if loss is not None:
                    # You shouldn't necessarily have to compute a loss for validation, so we allow for
                    # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                    # currently only used as the divisor for the loss function, so we can safely only
                    # count those batches for which we actually have a loss.  If this variable ever
                    # gets used for something else, we might need to change things around a bit.
                    batches_this_epoch += 1
                    val_loss += loss.detach().cpu().numpy()
            elif self.training_signal == "rl":
                loss, _, _ = self.batch_reward(batch_group, for_training=False)
                if loss is not None:
                    batches_this_epoch += 1
                    # Returned loss is a manually tensored scalar, so no need to detach, just use .item()
                    val_loss += np.array(loss.item(), dtype=float)
            elif self.training_signal == "beam_rl":
                loss = self.batch_reward(batch_group, for_training=False)
                if loss is not None:
                    batches_this_epoch += 1
                    val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest validation metrics
            val_metrics = training_util.get_metrics(self.model, val_loss, batches_this_epoch)
            description = training_util.description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()

        return val_loss, batches_this_epoch

    @overrides
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        # Set the model to "train" mode.
        self.model.train()

        num_gpus = len(self._cuda_devices)

        # Get tqdm for the training batches
        raw_train_generator = self.iterator(self.train_data,
                                            num_epochs=1,
                                            shuffle=self.shuffle)
        train_generator = lazy_groups_of(raw_train_generator, num_gpus)
        num_training_batches = math.ceil(self.iterator.get_num_batches(self.train_data)/num_gpus)
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())

        logger.info("Training with signal {}".format(self.training_signal))
        # print("Training with signal {}".format(self.training_signal))
        train_generator_tqdm = Tqdm.tqdm(train_generator,
                                         total=num_training_batches)
        cumulative_batch_size = 0
        for batch_group in train_generator_tqdm:
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self.optimizer.zero_grad()

            if self.training_signal == "supervised":
                loss = self.batch_loss(batch_group, for_training=True)

                if torch.isnan(loss):
                    raise ValueError("nan loss encountered")

                loss.backward()

                train_loss += loss.item()

            elif self.training_signal == "rl":
                if self.accumulate_grads:
                    num_samples = sum([training_util.get_batch_size(batch) for batch in batch_group])
                    eff_batch_size = num_samples // (self.model._num_examples + self.model._num_held_out)
                    _len = self.rl_inner_batch * (self.model._num_examples + self.model._num_held_out)
                    for i in range(0, eff_batch_size, self.rl_inner_batch):
                        start = i * (self.model._num_examples + self.model._num_held_out)

                        # Split batch_group into mini batch_groups (Hard Coded - will only work for single GPU)
                        mini_batch_group = [{
                            "input_str": {
                                "tokens": batch_group[0]["input_str"]["tokens"].narrow(0, start, _len)
                            },
                            "output_str": {
                                "tokens": batch_group[0]["output_str"]["tokens"].narrow(0, start, _len)
                            },
                            "program_str": {
                                "tokens": batch_group[0]["program_str"]["tokens"].narrow(0, start, _len)
                            }
                        }]
                        loss, variables, grad_variables = self.batch_reward(mini_batch_group, for_training=True)
                        autograd.backward(variables, grad_variables)
                        train_loss += loss.item()
                else:
                    loss, variables, grad_variables = self.batch_reward(batch_group, for_training=True)
                    # When the .backwards method is called on a scalar value i.e. variables is a scalar value like loss,
                    # PyTorch preempts the grad_variable argument to be Torch.Tensor([1]).
                    # For calling backward() on tensors, alternate usage: variables.backward(grad_variables).
                    # In our case, all sampled actions will constitute 'variables' tensor and 'grad_variables' will contain
                    # the gradient values corresponding to each action taken at every time step of the decoded program
                    autograd.backward(variables, grad_variables)
                    train_loss += loss.item()

            elif self.training_signal == "beam_rl":

                if self.accumulate_grads:
                    num_samples = sum([training_util.get_batch_size(batch) for batch in batch_group])
                    eff_batch_size = num_samples//(self.model._num_examples + self.model._num_held_out)
                    _len = self.rl_inner_batch * (self.model._num_examples + self.model._num_held_out)
                    for i in range(0, eff_batch_size, self.rl_inner_batch):

                        start = i*(self.model._num_examples + self.model._num_held_out)

                        # Split batch_group into mini batch_groups (Hard Coded - will only work for single GPU)
                        mini_batch_group = [{
                            "input_str": {
                                "tokens": batch_group[0]["input_str"]["tokens"].narrow(0, start, _len)
                            },
                            "output_str": {
                                "tokens": batch_group[0]["output_str"]["tokens"].narrow(0, start, _len)
                            },
                            "program_str": {
                                "tokens": batch_group[0]["program_str"]["tokens"].narrow(0, start, _len)
                            }
                        }]
                        loss = self.batch_reward(mini_batch_group, for_training=True)
                        if torch.isnan(loss):
                            raise ValueError("nan loss encountered")
                        # Reward is maximised so loss = -reward.
                        loss.backward()
                        train_loss += loss.item()

                else:
                    loss = self.batch_reward(batch_group, for_training=True)
                    if torch.isnan(loss):
                        raise ValueError("nan loss encountered")
                    # Reward is maximised so loss = -reward.
                    loss.backward()
                    train_loss += loss.item()

            else:
                raise NotImplementedError("Unknown training method")

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using a
            # scheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(batch_num_total)

            if self._tensorboard.should_log_histograms_this_batch():
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                param_updates = {name: param.detach().cpu().clone()
                                 for name, param in self.model.named_parameters()}
                self.optimizer.step()
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1, ))
                    param_norm = torch.norm(param.view(-1, )).cpu()
                    self._tensorboard.add_train_scalar("gradient_update/" + name,
                                                       update_norm / (param_norm + 1e-7))
            else:
                self.optimizer.step()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics.
            # training_util.get_metrics() first internally calls model.get_metrics() which returns empty dict.
            # Then running avg loss is computed and stored in metrics dict.
            metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch)  # provides running avg.loss
            description = training_util.description_from_metrics(metrics)  # A string representation of metrics
            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if self._tensorboard.should_log_this_batch():
                self._tensorboard.log_parameter_and_gradient_statistics(self.model, batch_grad_norm)
                self._tensorboard.log_learning_rates(self.model, self.optimizer)

                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"])
                self._tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self._tensorboard.should_log_histograms_this_batch():
                self._tensorboard.log_histograms(self.model, histogram_parameters)

            if self._log_batch_size_period:
                cur_batch = sum([training_util.get_batch_size(batch) for batch in batch_group])
                cumulative_batch_size += cur_batch
                if (batches_this_epoch - 1) % self._log_batch_size_period == 0:
                    average = cumulative_batch_size/batches_this_epoch
                    logger.info(f"current batch size: {cur_batch} mean batch size: {average}")
                    self._tensorboard.add_train_scalar("current_batch_size", cur_batch)
                    self._tensorboard.add_train_scalar("mean_batch_size", average)

            # Save model if needed.
            if self._model_save_interval is not None and (
                    time.time() - last_save_time > self._model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                        '{0}.{1}'.format(epoch, training_util.time_to_str(int(last_save_time)))
                )
        metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch, reset=True)
        metrics['cpu_memory_MB'] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics['gpu_'+str(gpu_num)+'_memory_MB'] = memory
        return metrics

    @overrides
    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError("Could not recover training from the checkpoint.  Did you mean to output to "
                                     "a different serialization directory or delete the existing serialization "
                                     "directory?")

        print("Resuming training from epoch {}".format(epoch_counter))
        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        # MetricTracker initialised in 'Trainer' base class (with 1. patience value and 2. validation_metric (assigned
        # with +/-)). +/- is taken of the validation_metric before assigning it to self._validation_metric.
        # Dict val_metrics have keys without +/-.
        metrics['best_epoch'] = self._metric_tracker.best_epoch  # int
        for key, value in self._metric_tracker.best_epoch_metrics.items():  # Dict[str, float]
            metrics["best_validation_" + key] = value

        # TODO DEBUG: Check model parameters
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print("---DEBUG--- ", name)

        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)  # Contains cpu/gpu memory in Mb and avg. batch loss/reward

            # get peak of memory usage for CPU and all GPUs. Populate metrics from train_metrics
            if 'cpu_memory_MB' in train_metrics:
                metrics['peak_cpu_memory_MB'] = max(metrics.get('peak_cpu_memory_MB', 0),
                                                    train_metrics['cpu_memory_MB'])
            for key, value in train_metrics.items():
                if key.startswith('gpu_'):
                    metrics["peak_"+key] = max(metrics.get("peak_"+key, 0), value)

            if self._validation_data is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it. Following statement populates
                    # val_metrics with 'loss'. We will avoid the unnecessary computation if loss is not the specified
                    # metric to be monitored
                    if self._validation_metric == "loss":
                        val_loss, num_batches = self._validation_loss()
                        val_metrics = training_util.get_metrics(self.model, val_loss, num_batches, reset=True)
                    else:
                        # val_loss, num_batches = self._validation_loss()
                        # val_metrics = training_util.get_metrics(self.model, val_loss, num_batches, reset=True)
                        # Custom metrics computed to monitor training
                        val_metrics.update(self.no_loss_custom_validation_metrics())

                    # Check validation metric for early stopping. add_metric() will compare the computed val_metric with
                    # best_so_far and will correspondingly update its value. If there is no improvement,
                    # epochs_with_no_improvement is updated. If you want to monitor multiple validation metrics, use
                    # add_metrics()
                    if self._validation_metric in val_metrics.keys():
                        this_epoch_val_metric = val_metrics[self._validation_metric]
                    else:
                        raise KeyError("Specified validation metric: {} to be monitored is not present in computed set "
                                       "of metrics: {}".format(self._validation_metric, val_metrics.keys()))
                    self._metric_tracker.add_metric(this_epoch_val_metric)

                    # This happens when epochs_with_no_improvement becomes >= patience
                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience. Stopping training.")
                        break

            self._tensorboard.log_metrics(train_metrics,
                                          val_metrics=val_metrics,
                                          log_to_console=True,
                                          epoch=epoch + 1)  # +1 because tensor board doesn't like 0

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            # Populate metrics with training and validation metrics
            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            # If the current computed validation_metric is best so far. Here, we are monitoring only one val_metric.
            # Remaining val_metrics will also be recorded as best irrespective of whether they are best or not.
            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics['best_epoch'] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir:
                dump_metrics(os.path.join(self._serialization_dir, f'metrics_epoch_{epoch}.json'), metrics)

            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)
            if self._momentum_scheduler:
                self._momentum_scheduler.step(this_epoch_val_metric, epoch)

            self._save_checkpoint(epoch)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * \
                    ((self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1)
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        # make sure pending events are flushed to disk and files are closed properly
        self._tensorboard.close()

        # Load the best model state before returning
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return metrics

    def _save_checkpoint(self, epoch: Union[int, str]) -> None:
        """
        Saves a checkpoint of the model to self._serialization_dir.
        Is a no-op if self._serialization_dir is None.

        Parameters
        ----------
        epoch : Union[int, str], required.
            The epoch of training.  If the checkpoint is saved in the middle
            of an epoch, the parameter is a string with the epoch and timestamp.
        """
        # If moving averages are used for parameters, we save
        # the moving average values into checkpoint, instead of the current values.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        # These are the training states we need to persist.
        training_states = {
                "metric_tracker": self._metric_tracker.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "batch_num_total": self._batch_num_total
        }

        # If we have a learning rate or momentum scheduler, we should persist them too.
        if self._learning_rate_scheduler is not None:
            training_states["learning_rate_scheduler"] = self._learning_rate_scheduler.state_dict()
        if self._momentum_scheduler is not None:
            training_states["momentum_scheduler"] = self._momentum_scheduler.state_dict()

        self._checkpointer.save_checkpoint(
                model_state=self.model.state_dict(),
                epoch=epoch,
                training_states=training_states,
                is_best_so_far=self._metric_tracker.is_best_so_far())

        # Restore the original values for parameters so that training will not be affected.
        if self._moving_average is not None:
            self._moving_average.restore()

    def _restore_checkpoint(self) -> int:
        """
        Restores the model and training state from the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.

        Returns
        -------
        epoch: int
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        model_state, training_state = self._checkpointer.restore_checkpoint()

        if not training_state:
            # No checkpoint to restore, start at 0
            return 0

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(training_state["optimizer"])
        if self._learning_rate_scheduler is not None and "learning_rate_scheduler" in training_state:
            self._learning_rate_scheduler.load_state_dict(training_state["learning_rate_scheduler"])
        if self._momentum_scheduler is not None and "momentum_scheduler" in training_state:
            self._momentum_scheduler.load_state_dict(training_state["momentum_scheduler"])
        training_util.move_optimizer_to_cuda(self.optimizer)

        # Currently the ``training_state`` contains a serialized ``MetricTracker``.
        if "metric_tracker" in training_state:
            self._metric_tracker.load_state_dict(training_state["metric_tracker"])
        # It used to be the case that we tracked ``val_metric_per_epoch``.
        elif "val_metric_per_epoch" in training_state:
            self._metric_tracker.clear()
            self._metric_tracker.add_metrics(training_state["val_metric_per_epoch"])
        # And before that we didn't track anything.
        else:
            self._metric_tracker.clear()

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1

        # For older checkpoints with batch_num_total missing, default to old behavior where
        # it is unchanged.
        batch_num_total = training_state.get('batch_num_total')
        if batch_num_total is not None:
            self._batch_num_total = batch_num_total

        return epoch_to_return

    # Requires custom from_params.
    @classmethod
    def from_params(cls, params: Params, serialization_dir: str, recover: bool = False, cache_directory: str = None,
                    cache_prefix: str = None, **kwargs) -> 'Trainer':
        # pylint: disable=arguments-differ

        pieces = TrainerPieces.from_params(params, serialization_dir, recover)
        model = pieces.model
        iterator = pieces.iterator
        train_data = pieces.train_dataset
        validation_data = pieces.validation_dataset
        params = pieces.params
        validation_iterator = pieces.validation_iterator

        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)
        momentum_scheduler_params = params.pop("momentum_scheduler", None)

        if isinstance(cuda_device, list):
            model_device = cuda_device[0]
        else:
            model_device = cuda_device
        if model_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(model_device)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))
        if "moving_average" in params:
            moving_average = MovingAverage.from_params(params.pop("moving_average"), parameters=parameters)
        else:
            moving_average = None

        if lr_scheduler_params:
            lr_scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            lr_scheduler = None
        if momentum_scheduler_params:
            momentum_scheduler = MomentumScheduler.from_params(optimizer, momentum_scheduler_params)
        else:
            momentum_scheduler = None

        if 'checkpointer' in params:
            if 'keep_serialized_model_every_num_seconds' in params or \
                    'num_serialized_models_to_keep' in params:
                raise ConfigurationError(
                        "Checkpointer may be initialized either from the 'checkpointer' key or from the "
                        "keys 'num_serialized_models_to_keep' and 'keep_serialized_model_every_num_seconds'"
                        " but the passed config uses both methods.")
            checkpointer = Checkpointer.from_params(params.pop("checkpointer"))
        else:
            num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
            keep_serialized_model_every_num_seconds = params.pop_int(
                    "keep_serialized_model_every_num_seconds", None)
            checkpointer = Checkpointer(
                    serialization_dir=serialization_dir,
                    num_serialized_models_to_keep=num_serialized_models_to_keep,
                    keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds)

        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
        log_batch_size_period = params.pop_int("log_batch_size_period", None)
        training_signal = params.pop("training_signal", "supervised")
        rl_inner_batch = params.pop_int("rl_inner_batch", 1)
        fine_tune_rf = params.pop_bool("fine_tune_rf", False)

        params.assert_empty(cls.__name__)
        return cls(model, optimizer, iterator,
                   train_data, validation_data,
                   patience=patience,
                   validation_metric=validation_metric,
                   validation_iterator=validation_iterator,
                   shuffle=shuffle,
                   num_epochs=num_epochs,
                   serialization_dir=serialization_dir,
                   cuda_device=cuda_device,
                   grad_norm=grad_norm,
                   grad_clipping=grad_clipping,
                   learning_rate_scheduler=lr_scheduler,
                   momentum_scheduler=momentum_scheduler,
                   checkpointer=checkpointer,
                   model_save_interval=model_save_interval,
                   summary_interval=summary_interval,
                   histogram_interval=histogram_interval,
                   should_log_parameter_statistics=should_log_parameter_statistics,
                   should_log_learning_rate=should_log_learning_rate,
                   log_batch_size_period=log_batch_size_period,
                   moving_average=moving_average,
                   training_signal=training_signal,
                   rl_inner_batch=rl_inner_batch,
                   fine_tune_rf=fine_tune_rf)
