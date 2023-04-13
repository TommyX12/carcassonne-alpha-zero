import os
import tensorflow as tf
from sacred import Experiment as SacredExperiment
from sacred.observers import FileStorageObserver

from datetime import datetime

from pathlib import Path

import shutil

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

DEFAULT_MAX_RETENTION = 10


class Experiment(object):
    def __init__(self):
        self.name = None
        self.sacred: SacredExperiment = None
        self.summary_writer = None
        self.max_retention = DEFAULT_MAX_RETENTION

    def setup(self, name, sacred_ex, *, max_retention=None):
        self.name = name
        self.sacred = sacred_ex
        if max_retention is not None:
            self.max_retention = max_retention

    def setup_config(self, experiment_parent_dir, experiment_dir, tensorboard_dir):
        physical_devices = tf.config.list_physical_devices('GPU')
        print(f'Num GPUs: {len(physical_devices)}')

        os.makedirs(experiment_parent_dir, exist_ok=True)
        if self.max_retention > 0:
            result_dirs = [
                entry.path for entry in os.scandir(experiment_parent_dir) if entry.is_dir()]
            if len(result_dirs) > self.max_retention:
                print(f'Max retention ({self.max_retention}) exceeded.')
                result_dirs.sort()
                for i in range(len(result_dirs) - self.max_retention):
                    result_dir = result_dirs[i]
                    print(f'Removing: {result_dir}')
                    shutil.rmtree(result_dir)

        os.makedirs(experiment_dir, exist_ok=True)
        self.sacred.observers.append(FileStorageObserver(
            Path(experiment_dir),
            source_dir=Path(os.path.join(experiment_dir, 'sources'))))
        self.summary_writer = tf.summary.create_file_writer(tensorboard_dir)


ex = Experiment()


def init_experiment(name, sacred_ex, *args, **kwargs):
    ex.setup(name, sacred_ex, *args, **kwargs)

    @ex.sacred.config
    def setup_config():
        tag = 'draft'
        full_tag = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + tag
        experiment_parent_dir = os.path.join(
            ROOT_DIR, 'experiments', 'runs', ex.name)
        experiment_dir = os.path.join(experiment_parent_dir, full_tag)
        tensorboard_dir = os.path.join(experiment_dir, 'logs')

        # Setup
        ex.setup_config(experiment_parent_dir, experiment_dir, tensorboard_dir)
