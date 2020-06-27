from slgep_lib import wrap_config
from utils import Saver
from mfea import mfea
import argparse
import yaml


# Load configuration
config = yaml.load(open('config.yaml').read())

# Load benchmark
benchmark = yaml.load(open('atari_benchmark/multitask-benchmark.yaml').read())

instances = []
for i in range(1, 41):
	if i not in [100]:
		instances.append('multi-' + str(i))


seeds = range(1, 21)

for seed in seeds:
    for instance in instances:
        data = benchmark[instance]
        config.update(data)

        config = wrap_config(config)
        saver = Saver(config, instance, seed)

        mfea(config, saver.append)
        saver.save()
