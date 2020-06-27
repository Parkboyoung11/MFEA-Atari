from slgep_lib import wrap_config
from utils import Saver
from cea import cea
import argparse
import yaml


# Load configuration
config = yaml.load(open('config.yaml').read())

# Load benchmark
singletask_benchmark = yaml.load(open('atari_benchmark/singletask-benchmark.yaml').read())

instances = []
for i in range(1, 41):
	if i not in [100]:
		instances.append('single-' + str(i))
		instances.append('single-' + str(i) + 's')

seeds = range(1, 21)

for seed in seeds:
    for instance in instances:
        data = singletask_benchmark[instance]
        config.update(data)

        config = wrap_config(config)
        saver = Saver(config, instance, seed)

        cea(config, saver.append)
        saver.save()
