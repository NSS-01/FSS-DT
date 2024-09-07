"""
Improved Arithmetic Secret Sharing
3PC
"""

import torch

from crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
from crypto.tensor.RingTensor import RingTensor


class ImprovedSecretSharing(object):
    """
    支持增强的算术秘密共享运算的自定义tensor类

    属性:
        public:
        replicated_shared_tensor: 其张量值，ReplicatedSharedTensor类型
        party: 所属参与方
    """

    def __init__(self, public, replicated_shared_tensor, party):
        self.public = public
        self.replicated_shared_tensor = replicated_shared_tensor
        self.party = party
        self.shape = public.shape

    @staticmethod
    def generate_offline_random_mask(number_of_mask):
        """
        离线产生随机掩码，掩码经复制秘密分享后再保存到本地。

        :param number_of_mask: 产生掩码的位数。
        """
        mask = RingTensor.random([number_of_mask])
        shared_mask = ReplicatedSecretSharing.share(mask)
        for i in range(3):
            # save the mask to the file
            shared_mask[i][0].save('./data/mask_data/S' + str(i) + '/mask_0.pth')
            shared_mask[i][1].save('./data/mask_data/S' + str(i) + '/mask_1.pth')
        # return shared_mask

    def __str__(self):
        return "[{}\n public:{},\n rss :{},\n party:{}]".format(self.__class__.__name__,
                                                                self.public.__str__(),
                                                                self.replicated_shared_tensor.__str__(),
                                                                self.party.party_id)

    def __add__(self, other):
        new_public = self.public + other.public
        new_rss = self.replicated_shared_tensor + other.replicated_shared_tensor
        return ImprovedSecretSharing(new_public, new_rss, self.party)

    def __sub__(self, other):
        if isinstance(other, ImprovedSecretSharing):
            new_public = self.public - other.public
            new_rss = self.replicated_shared_tensor - other.replicated_shared_tensor
            return ImprovedSecretSharing(new_public, new_rss, self.party)
        elif isinstance(other, RingTensor):

            if self.party.party_id ==0:
                new_public = self.public - other
            else:
                new_public = self.public - other
            return  ImprovedSecretSharing(new_public, self.replicated_shared_tensor, self.party)

    def __mul__(self, other):
        if isinstance(other, ImprovedSecretSharing):
            phi_z = self.replicated_shared_tensor * other.replicated_shared_tensor  # 这个可以离线计算，但是需要提前知道计算图
            # get offline mask
            num_of_element = self.public.tensor.numel()
            # todo:增加对多维张量的支持
            # mask = self.party.get_pre_generated_mask(num_of_element)
            # mask = mask.reshape(self.public.tensor.shape)
            # print(mask.restore())

            mask = self.party.rss_provider.get(num_of_element)
            mask = mask.view(self.public.tensor.shape)

            # print("mask", mask.restore())
            #

            # 先算 新iss的第一个share
            if self.party.party_id == 0:
                mz_0 = self.public * other.public + self.public * \
                       other.replicated_shared_tensor.replicated_shared_tensor[0] + \
                       self.replicated_shared_tensor.replicated_shared_tensor[0] * other.public + \
                       phi_z.replicated_shared_tensor[0] - mask.replicated_shared_tensor[0]

                mz_1 = self.public * other.replicated_shared_tensor.replicated_shared_tensor[1] + \
                       self.replicated_shared_tensor.replicated_shared_tensor[1] * other.public + \
                       phi_z.replicated_shared_tensor[1] - mask.replicated_shared_tensor[1]
            elif self.party.party_id == 1:
                mz_0 = self.public * other.replicated_shared_tensor.replicated_shared_tensor[
                    0] + \
                       self.replicated_shared_tensor.replicated_shared_tensor[0] * other.public + \
                       phi_z.replicated_shared_tensor[0] - mask.replicated_shared_tensor[0]

                mz_1 = self.public * other.replicated_shared_tensor.replicated_shared_tensor[1] + \
                       self.replicated_shared_tensor.replicated_shared_tensor[1] * other.public + \
                       phi_z.replicated_shared_tensor[1] - mask.replicated_shared_tensor[1]

            else:
                mz_0 = self.public * other.replicated_shared_tensor.replicated_shared_tensor[
                    0] + \
                       self.replicated_shared_tensor.replicated_shared_tensor[0] * other.public + \
                       phi_z.replicated_shared_tensor[0] - mask.replicated_shared_tensor[0]

                mz_1 = self.public * other.public + self.public * \
                       other.replicated_shared_tensor.replicated_shared_tensor[1] + \
                       self.replicated_shared_tensor.replicated_shared_tensor[1] * other.public + \
                       phi_z.replicated_shared_tensor[1] - mask.replicated_shared_tensor[1]

            #
            # print(mz_0)
            #
            # print(mz_1)
            new_rss = ReplicatedSecretSharing([mz_0, mz_1], self.party)
            mz = new_rss.restore()
            # print("*******************************")
            # print("mz:", mz)
            # print("pla mul", mz + mask.restore())
            # print("*******************************")
            return ImprovedSecretSharing(mz, mask, self.party)
        elif isinstance(other, RingTensor):
            new_public = self.public * other
            new_replicated_shared_tensor = self.replicated_shared_tensor * other
            return ImprovedSecretSharing(new_public, new_replicated_shared_tensor, self.party)

        elif isinstance(other, int):
            o = torch.tensor([other], dtype=torch.int64)
            r_int = RingTensor.load_from_value(o)
            return self * r_int


    @staticmethod
    def share(tensor: RingTensor):
        """
        对输入的RingTensor进行三方改进加法秘密分享。

        :param tensor: 进行秘密分享的输入数据张量，类型为RingTensor。
        :return: masked_tensor: 经掩码修饰后的数据张量，类型为RingTensor。
        :return: shares: 掩码经三方复制秘密分享后的分享份额列表，包含三个二元RingTensor列表。
        """
        # mask origin tensor with random tensor
        mask = RingTensor.random(tensor.shape, tensor.dtype, tensor.scale)

        # mask = torch.randint_like(tensor.tensor, 0, 10)
        # mask = RingTensor.convert_to_ring(mask)
        # mask = RingTensor.random(tensor.shape, tensor.dtype, tensor.scale)
        masked_tensor = tensor - mask
        # share mask to replicate secret shares
        shares = ReplicatedSecretSharing.share(mask)
        return masked_tensor, shares

    def restore(self):
        """
        基于三方改进加法秘密分享的数据张量的明文值恢复。

        :return: 恢复后的数据张量，类型为RingTensor。
        """
        # 发送部分
        self.party.send_ring_tensor_to((self.party.party_id + 1) % 3,
                                       self.replicated_shared_tensor.replicated_shared_tensor[0])
        # 接收部分
        other = self.party.receive_ring_tensor_from((self.party.party_id + 2) % 3)
        return self.replicated_shared_tensor.replicated_shared_tensor[0] + \
            self.replicated_shared_tensor.replicated_shared_tensor[1] + other + self.public

    def __getitem__(self, item):
        new_public = self.public[item]
        new_rss = self.replicated_shared_tensor[item]
        return ImprovedSecretSharing(new_public, new_rss, self.party).clone()

    def __setitem__(self, key, value):
        self.public[key] = value.public.clone()
        # print("pla value", value.restore())
        # print("public", self.public)
        self.replicated_shared_tensor[key] = value.replicated_shared_tensor.clone()

    def sum(self, dim):
        """
        沿着指定的维度对ImprovedSecretSharing中的public与replicated_shared_tensor分别求和。

        :param dim: 要进行求和操作的维度。
        :return: 一个新的ImprovedSecretSharing实例，其中包含两个求和后的数据和当前的party信息。
        """
        new_public = self.public.sum(dim)
        new_rss = self.replicated_shared_tensor.sum(dim)
        return ImprovedSecretSharing(new_public, new_rss, self.party)

    def __len__(self):
        return self.public.tensor.numel()

    def repeate_interleave(self, repeate_times, dim):
        new_public = self.public.repeat_interleave(repeate_times, dim=dim)
        new_rss = self.replicated_shared_tensor.repeat_interleave(repeate_times, dim=dim)
        return ImprovedSecretSharing(new_public, new_rss, self.party)

    def squeeze(self, dim):
        new_public = self.public.squeeze(dim=dim)
        new_rss = self.replicated_shared_tensor.squeeze(dim=dim)
        return ImprovedSecretSharing(new_public, new_rss, self.party)

    def unsqueeze(self, dim):
        new_public = self.public.unsqueeze(dim=dim)
        new_rss = self.replicated_shared_tensor.unsqueeze(dim=dim)
        return ImprovedSecretSharing(new_public, new_rss, self.party)

    def clone(self):
        public_clone = self.public.clone()
        rss_clone = self.replicated_shared_tensor.clone()
        return ImprovedSecretSharing(public_clone, rss_clone, self.party)

    def view(self, *args):
        public_view = self.public.view(*args)
        rss_view = self.replicated_shared_tensor.view(*args)
        return ImprovedSecretSharing(public_view, rss_view, self.party)

    @staticmethod
    def gen_and_share(r_tensor, party):
        public, psi = ImprovedSecretSharing.share(r_tensor)
        psi0, psi1, psi2 = psi
        psi0 = ReplicatedSecretSharing(psi0, party)
        ISS = ImprovedSecretSharing(public, psi0, party)

        ISS1 = ImprovedSecretSharing(public, ReplicatedSecretSharing(psi1, party), party)
        ISS2 = ImprovedSecretSharing(public, ReplicatedSecretSharing(psi2, party), party)

        party.send_iss_to((party.party_id + 1) % 3, ISS1)
        party.send_iss_to((party.party_id + 2) % 3, ISS2)

        return ISS
