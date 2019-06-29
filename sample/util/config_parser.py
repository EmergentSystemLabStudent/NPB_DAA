from configparser import ConfigParser

class ConfigParser_with_eval(ConfigParser):
    def get(self, *argv, **kwargs):
        import numpy
        val = super(ConfigParser_with_eval, self).get(*argv, **kwargs)
        return eval(val)
