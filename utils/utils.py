from collections import defaultdict
import time
from datetime import datetime
import json
import shutil
from typing import Any, Callable, Dict
import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm


class CyclicCounter(object):
    def __init__(self, n, include_start=True):
        self.n = n
        self.i = n if include_start else 0
        self.iter_idx = -1

    def next(self):
        self.i += 1
        if self.i >= self.n:
            self.i = 0

        self.iter_idx += 1

    def active(self):
        return self.i == 0


class CyclicTimeBasedCounter(object):
    def __init__(self, seconds, include_start=True):
        self.seconds = seconds
        self.last_trigger = time.time() - seconds - 100000 if include_start else time.time()
        self.triggered = False

    def next(self):
        self.triggered = False

        current_elapsed = time.time() - self.last_trigger
        if current_elapsed >= self.seconds:
            self.triggered = True
            self.last_trigger = time.time()

    def active(self):
        return self.triggered


def save_model(model, path):
    model.save(os.path.join(path, 'model'))

def save_optimizer(optimizer, model, path):
    if not optimizer.get_weights():
        # https://stackoverflow.com/a/64671177/3308553
        model_train_vars = model.trainable_variables
        zero_grads = [tf.zeros_like(w) for w in model_train_vars]
        saved_vars = [tf.identity(w) for w in model_train_vars]
        optimizer.apply_gradients(zip(zero_grads, model_train_vars))
        [x.assign(y) for x, y in zip(model_train_vars, saved_vars)]

    np.save(os.path.join(path, 'optimizer.npy'),
            np.array(optimizer.get_weights(), dtype=object), allow_pickle=True)

def load_model(path):
    return tf.keras.models.load_model(os.path.join(path, 'model'))

def load_optimizer(optimizer, model, path):
    # https://stackoverflow.com/a/64671177/3308553
    model_train_vars = model.trainable_variables
    zero_grads = [tf.zeros_like(w) for w in model_train_vars]
    saved_vars = [tf.identity(w) for w in model_train_vars]
    optimizer.apply_gradients(zip(zero_grads, model_train_vars))
    [x.assign(y) for x, y in zip(model_train_vars, saved_vars)]
    optimizer.set_weights(
        np.load(os.path.join(path, 'optimizer.npy'), allow_pickle=True))


class CheckpointManager(object):
    def __init__(self, experiment_dir, interval, max_retention=5):
        assert interval >= 10
        self.experiment_dir = experiment_dir
        self.timer = CyclicTimeBasedCounter(interval)
        self.max_retention = max_retention
        self.checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')

    @staticmethod
    def get_latest_checkpoint_in(dir):
        if not os.path.exists(dir):
            return None

        checkpoint_paths = [
            entry.path for entry in os.scandir(dir) if entry.is_dir()]

        if len(checkpoint_paths) == 0:
            return None

        checkpoint_paths.sort()
        latest_dir = checkpoint_paths[-1]
        return latest_dir

    def step(self, model, optimizer, step):
        self.timer.next()
        if not self.timer.active():
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # purge old checkpoints
        if self.max_retention > 0:
            result_dirs = [
                entry.path for entry in os.scandir(self.checkpoint_dir) if entry.is_dir()]
            if len(result_dirs) > self.max_retention:
                result_dirs.sort()
                for i in range(len(result_dirs) - self.max_retention):
                    result_dir = result_dirs[i]
                    shutil.rmtree(result_dir)

        # save
        checkpoint_path = os.path.join(self.checkpoint_dir, datetime.now().strftime(
            '%Y-%m-%d_%H-%M-%S' + '_' + str(step)))
        os.makedirs(checkpoint_path, exist_ok=True)

        save_model(model, checkpoint_path)
        save_optimizer(optimizer, model, checkpoint_path)

        with open(os.path.join(checkpoint_path, 'state.json'), 'w') as f:
            json.dump({'step': step}, f)

    def load(self, checkpoint_path, model, optimizer):
        print('Loading checkpoint from', checkpoint_path)

        load_model(checkpoint_path)
        load_optimizer(optimizer, model, checkpoint_path)

        with open(os.path.join(checkpoint_path, 'state.json'), 'r') as f:
            step = json.load(f)['step']

        return model, optimizer, step


class CheckpointDirManager(object):
    def __init__(self, checkpoint_dir, max_retention=5):
        self.max_retention = max_retention
        self.checkpoint_dir = checkpoint_dir

    @staticmethod
    def get_latest_checkpoint_in(dir):
        if not os.path.exists(dir):
            return None

        checkpoint_paths = [
            entry.path for entry in os.scandir(dir) if entry.is_dir()]

        if len(checkpoint_paths) == 0:
            return None

        checkpoint_paths.sort()
        latest_dir = checkpoint_paths[-1]
        return latest_dir

    def next_checkpoint_path(self, step=0):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # purge old checkpoints
        if self.max_retention > 0:
            result_dirs = [
                entry.path for entry in os.scandir(self.checkpoint_dir) if entry.is_dir()]
            if len(result_dirs) > self.max_retention:
                result_dirs.sort()
                for i in range(len(result_dirs) - self.max_retention):
                    result_dir = result_dirs[i]
                    shutil.rmtree(result_dir)

        checkpoint_path = os.path.join(self.checkpoint_dir, datetime.now().strftime(
            '%Y-%m-%d_%H-%M-%S' + '_' + str(step)))
        os.makedirs(checkpoint_path, exist_ok=True)

        return checkpoint_path


class ModelSummarizer(object):
    def __init__(self, model):
        self.model = model
        self.finished = False

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.finished:
            self.model.summary(expand_nested=True, show_trainable=True)
            self.finished = True


class ModelGraphVisualizer(object):
    def __init__(self, writer, model, experiment_dir, tensorboard_dir):
        self.writer = writer
        self.model = model
        self.experiment_dir = experiment_dir
        self.tensorboard_dir = tensorboard_dir
        self.finished = False

    def __enter__(self):
        if not self.finished:
            with self.writer.as_default():
                tf.summary.trace_on(graph=True, profiler=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.finished:
            # tf.keras.utils.plot_model(self.model, to_file=os.path.join(
            #     self.experiment_dir, 'model_graph.png'), show_shapes=True)
            with self.writer.as_default():
                tf.summary.trace_export(
                    name="graph_trace", step=0, profiler_outdir=self.tensorboard_dir)
                tf.summary.trace_off()

            self.finished = True


# context manager based
class DebugTimer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self.name, 'took', time.time() - self.start, 'seconds')


class DebugMultiTimer(object):
    def __init__(self, name):
        self.name = name
        self.times = []

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.times.append(time.time() - self.start)

    def print(self):
        print(f'{self.name} took {np.mean(self.times)} +- {np.std(self.times)} seconds ({len(self.times)} times)')


def normalize_dict(d: Dict[Any, float]):
    total = sum(d.values())
    if total == 0:
        return None

    mul = 1 / total
    return {k: v * mul for k, v in d.items()}


def argmax_dict(d: Dict[Any, Any]):
    return max(d, key=d.get) # type: ignore


class keydefaultdict(dict):
    def __init__(self, default_factory, *a, **kw):
        if not isinstance(default_factory, Callable):
            raise TypeError('first argument must be callable')
        super().__init__(*a, **kw)
        self.default_factory = default_factory

    def __missing__(self, key):
        ret = self[key] = self.default_factory(key)
        return ret


def pretty_print_dict(d: Dict[Any, Any]):
    # convert keys to string
    d = {str(k): v for k, v in d.items()}
    print(json.dumps(d, indent=4, sort_keys=True))


def process_offset_dict(offset_dict: Dict[str, int]) -> int:
    total = 0
    for key in offset_dict.keys():
        size = offset_dict[key]
        offset_dict[key] = total
        total += size

    return total