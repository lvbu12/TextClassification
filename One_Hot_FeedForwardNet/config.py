# -*- coding: utf-8 -*-
import json
import six
import copy


class Config(object):

    def __init__(self):
        pass

    @classmethod
    def from_dict(cls, json_object):
        config = Config()
        for (key, val) in six.iteritems(json_object):
            config.__dict__[key] = val

        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as f:
            text = f.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file):
        with open(json_file, 'w', encoding='utf-8') as f:
            f.write(self.to_json_string())

if __name__ == "__main__":

    config_dict = {
        'chars_size': 7555, 
        'hidden_size': 512, 
        'output_size': 14,
        'chars_path': 'corpus/chars.lst',
        'labels_path': 'corpus/labels.lst',
        'train_path': 'data/train.txt',
        'valid_path': 'data/valid.txt',
        'test_path': 'data/test.txt',
        'pred_path': 'data/pred.txt',
        'lr': 0.01,
        'cur_lr': 0.0001,
        'batch_size': 32,
        'save_model_path': 'Models',
        'load_model_path': 'Models',
        'best_acc': 0.94,
        'epochs': 30,
        'print_every': 10,
        'valid_every': 100,
    }
    config = Config.from_dict(config_dict)
    print(type(config))
    print(config.chars_size)
    output = config.to_dict()
    print(output)
    json_str = config.to_json_string()
    print(json_str)
    config.to_json_file('zz.json')
