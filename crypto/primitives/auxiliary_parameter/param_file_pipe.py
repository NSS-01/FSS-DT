import os
import pickle

from config.base_configs import base_path


class ParamFilePipe:
    @staticmethod
    def write(param, party_id, num_of_party=2):
        path = f"{base_path}/aux_parameters/{type(param).__name__}/{num_of_party}party/"
        file_name = os.path.join(path, f"{type(param).__name__}_{party_id}.pth")

        if not os.path.exists(path):
            os.makedirs(path)
        dic = param.to('cpu').to_dic()
        with open(file_name, 'wb') as file:
            pickle.dump(dic, file)

    @staticmethod
    def read(param_type, party_id, num_of_party=2):
        path = f"{base_path}/aux_parameters/{param_type.__name__}/{num_of_party}party/"
        file_name = os.path.join(path, f"{param_type.__name__}_{party_id}.pth")
        with open(file_name, 'rb') as file:
            dic = pickle.load(file)
        param = param_type.from_dic(dic)
        return param

    @staticmethod
    def write_by_name(param, name, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        dic = param.to('cpu').to_dic()
        file_name = os.path.join(file_path, name)
        with open(file_name, 'wb') as file:
            pickle.dump(dic, file)

    @staticmethod
    def read_dic_by_name(name, file_path):
        file_name = os.path.join(file_path, name)
        with open(file_name, 'rb') as file:
            dic = pickle.load(file)
        return dic
