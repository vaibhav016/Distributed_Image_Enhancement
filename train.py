# Copyright 2022 Niharika Krishnan (@niharikakrishnan)
# Copyright 2022 Vaibhav Singh (@vaibhav016)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf
tf.keras.backend.clear_session()
from ray.train import Trainer
import json
import os

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

from dataset import make_data
from model import build_and_compile_latent_model
from config import Config
import argparse

parser = argparse.ArgumentParser(prog="Distributed Image Enhancement: BigData Project")

parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")
args = parser.parse_args()
config = Config(args.config)
m = config.data_config

path = 'faces_data/*.png'


def train_func_distributed():
    per_worker_batch_size = 8
    # This environment variable will be set by Ray Train.
    tf_config = json.loads(os.environ['TF_CONFIG'])
    num_workers = len(tf_config['cluster']['worker'])

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    global_batch_size = per_worker_batch_size * num_workers
    multi_worker_dataset = make_data(global_batch_size=global_batch_size, **config.data_config)

    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = build_and_compile_latent_model()
    multi_worker_model.fit(multi_worker_dataset, epochs=config.train_config["epochs"])


trainer = Trainer(backend=config.train_config["backend"], num_workers=config.train_config["num_workers"])

# For GPU Training, set `use_gpu` to True.
# trainer = Trainer(backend="tensorflow", num_workers=4, use_gpu=True)

trainer.start()
results = trainer.run(train_func_distributed)
trainer.shutdown()
