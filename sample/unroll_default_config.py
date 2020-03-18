import numpy
from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

argparser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

argparser.add_argument("--default_config", type=Path, default="hypparams/defaults.config")
argparser.add_argument("--output_dir", type=Path, default="hypparams")

args = argparser.parse_args()

default_config_file = args.default_config
config_root = args.output_dir

config_root.mkdir(exist_ok=True, parents=True)

# load default config
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read(default_config_file)

letter_num = parser.getint("model", "letter_num")

# unroll model
tmp_parser = ConfigParser()
tmp_parser.read_dict({"model": dict(parser["model"])})
with (config_root / "model.config").open("w") as f:
    tmp_parser.write(f)

# unroll pyhlm
tmp_parser = ConfigParser()
tmp_parser.read_dict({"pyhlm": dict(parser["pyhlm"])})
with (config_root / "pyhlm.config").open("w") as f:
    tmp_parser.write(f)

# unroll letter observation
tmp_parser = ConfigParser()
tmp_parser.read_dict({"DEFAULT": dict(parser["letter_observation"])})
for i in range(letter_num):
    tmp_parser.add_section(f"{i+1}_th")
with (config_root / "letter_observation.config").open("w") as f:
    tmp_parser.write(f)

# unroll letter duration
tmp_parser = ConfigParser()
tmp_parser.read_dict({"DEFAULT": dict(parser["letter_duration"])})
for i in range(letter_num):
    tmp_parser.add_section(f"{i+1}_th")
with (config_root / "letter_duration.config").open("w") as f:
    tmp_parser.write(f)

# unroll letter hsmm
tmp_parser = ConfigParser()
tmp_parser.read_dict({"letter_hsmm": dict(parser["letter_hsmm"])})
with (config_root / "letter_hsmm.config").open("w") as f:
    tmp_parser.write(f)

# unroll word length
tmp_parser = ConfigParser()
tmp_parser.read_dict({"word_length": dict(parser["word_length"])})
with (config_root / "word_length.config").open("w") as f:
    tmp_parser.write(f)

# unroll word superstate
tmp_parser = ConfigParser()
tmp_parser.read_dict({"DEFAULT": dict(parser["superstate"])})

sentence_files = numpy.loadtxt("files.txt", dtype=str)
for sentence in sentence_files:
    tmp_parser.add_section(sentence)
with (config_root / "superstate.config").open("w") as f:
    tmp_parser.write(f)
