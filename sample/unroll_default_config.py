import numpy
from configparser import ConfigParser, ExtendedInterpolation
import sys
from pathlib import Path

config_file = "hypparams/defaults.config"
if len(sys.argv) >= 2:
    config_file = sys.argv[1]

target_dir = "hypparams"
if len(sys.argv) >= 3:
    target_dir = sys.argv[2]

# ready the temp vars
config_root = Path(target_dir)

# load default config
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read(config_file)

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
