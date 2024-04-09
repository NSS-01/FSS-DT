from copy import deepcopy

from torch import Tensor

from crypto.primitives.auxiliary_parameter.param_file_pipe import ParamFilePipe
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.tensor.RingTensor import RingTensor


class Parameter(object):
    @staticmethod
    def gen(*args) -> list:
        pass

    @classmethod
    def gen_and_save(cls, *args):
        params = cls.gen(*args)
        num_of_party = len(params)
        for party_id in range(num_of_party):
            params[party_id].save_to_file_pipe(party_id, num_of_party)

    @classmethod
    def from_dic(cls, dic):
        ret = cls()
        for key, value in ret.__dict__.items():
            if hasattr(value, 'from_dic'):
                setattr(ret, key, getattr(ret, key).from_dic(dic[key]))
            else:
                setattr(ret, key, dic[key])
        return ret

    def to_dic(self):
        dic = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dic'):
                dic[key] = value.to_dic()
            else:
                dic[key] = value
        return dic

    @classmethod
    def load_from_file_pipe(cls, party_id, num_of_party=2):
        return ParamFilePipe.read(cls, party_id, num_of_party)

    def save_to_file_pipe(self, party_id, num_of_party=2):
        ParamFilePipe.write(self, party_id, num_of_party)

    def __getstate__(self):
        return self.to_dic()

    def __setstate__(self, state):
        self.__dict__.update(self.from_dic(state).__dict__)  # TODO

    def __len__(self):
        for attr, value in self.__dict__.items():
            if isinstance(value, (ArithmeticSharedRingTensor, RingTensor, Tensor)):
                return value.shape[0]

    def __getitem__(self, item):
        ret = self.__class__()
        for attr, value in self.__dict__.items():
            if hasattr(value, 'getitem'):
                setattr(ret, attr, value.getitem(item))
            elif hasattr(value, '__getitem__'):
                setattr(ret, attr, value[item])
            else:
                setattr(ret, attr, value)
        return ret

    def __setitem__(self, key, new_value):
        for attr_name, attr_value in self.__dict__.items():
            if hasattr(attr_value, '__setitem__'):
                attr_value.__setitem__(key, getattr(new_value, attr_name, None))
            else:
                setattr(self, attr_name, getattr(new_value, attr_name, None))

    def clone(self):
        ret = self.__class__()
        for attr, value in self.__dict__.items():
            if hasattr(value, 'clone'):
                setattr(ret, attr, value.clone())
            else:
                setattr(ret, attr, deepcopy(value))
        return ret

    def to(self, device):
        for attr, value in self.__dict__.items():
            if hasattr(value, 'to'):
                setattr(self, attr, value.to(device))
            elif isinstance(value, list):
                for i in range(len(value)):
                    if hasattr(value[i], 'to'):
                        value[i] = value[i].to(device)
            elif isinstance(value, dict):
                for k, v in value:
                    if hasattr(v, 'to'):
                        value[k] = v.to(device)
        return self

    def pop(self):
        ret = self.__class__()
        for attr, value in self.__dict__.items():
            if hasattr(value, 'pop'):
                setattr(ret, attr, value.pop())
            elif hasattr(value, "__getitem__") and not isinstance(value, str):
                setattr(ret, attr, value[-1])
                setattr(self, attr, value[:-1])
            else:
                setattr(ret, attr, value)
        return ret

    def expand_as(self, input):
        ret = self.__class__()
        for attr, value in self.__dict__.items():
            if hasattr(value, 'expand_as'):
                setattr(ret, attr, value.expand_as(getattr(input, attr)))
            elif isinstance(value, list):
                for i in range(len(value)):
                    if hasattr(value[i], 'expand_as'):
                        value[i] = value[i].expand_as(getattr(input, attr))
            elif isinstance(value, dict):
                for k, v in value:
                    if hasattr(v, 'expand_as'):
                        value[k] = v.expand_as(getattr(input, attr))
            else:
                setattr(ret, attr, value)
        return ret

    # def unsqueeze(self, dim):
    #     ret = self.__class__()
    #     for attr, value in self.__dict__.items():
    #         if hasattr(value, 'unsqueeze'):
    #             setattr(ret, attr, value.unsqueeze(dim))
    #         elif isinstance(value, list):
    #             for i in range(len(value)):
    #                 if hasattr(value[i], 'unsqueeze'):
    #                     value[i] = value[i].unsqueeze(dim)
    #         elif isinstance(value, dict):
    #             for k, v in value:
    #                 if hasattr(v, 'unsqueeze'):
    #                     value[k] = v.unsqueeze(dim)
    #         else:
    #             setattr(ret, attr, value)
    #     return ret
