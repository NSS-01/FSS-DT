"""
参考ABY3的3方2选1OT实现
"""

import torch

from config.base_configs import DEVICE


class OT(object):
    @staticmethod
    def sender(m0, m1, party, receiver_id, helper_id):
        """
        3方OT模型下的sender
        :param m0:sender所持有的信息m0
        :param m1:sender所持有的信息m1
        :param party:当前参与方
        :param receiver_id:接收方的编号
        :param helper_id:帮助方的编号
        :return:
        """

        w0 = torch.randint(0, 2, m0.shape, device=DEVICE)

        party.send_torch_tensor_to(helper_id, w0)
        w1 = party.receive_torch_tensor_from(helper_id)

        m0 = m0 ^ w0
        m1 = m1 ^ w1
        party.send_torch_tensor_to(receiver_id, m0)
        party.send_torch_tensor_to(receiver_id, m1)

    @staticmethod
    def receiver(c, party, sender_id, helper_id):
        """
        三方OT模型下的receiver helper默认位置在party_id + 1
        receiver需要选择wc
        :param c:选择位
        :param party:当前参与方
        :param sender_id:发送方的编号
        :param helper_id:帮助方的编号
        :return:
        """

        m0_masked = party.receive_torch_tensor_from(sender_id)
        m1_masked = party.receive_torch_tensor_from(sender_id)
        wc = party.receive_torch_tensor_from(helper_id)
        cond = (c.tensor > 0) + 0
        mc = m0_masked * (1 - cond) + m1_masked * cond
        mc = mc ^ wc

        return mc

    @staticmethod
    def helper(c, party, sender_id, receiver_id):
        """
        三方OT模型下的helper
        helper知道receiver需要选择wc
        :param party:当前参与方
        :param c:选择位
        :param sender_id:发送方的编号
        :param receiver_id:接收方的编号
        :return:
        """

        w0 = party.receive_torch_tensor_from(sender_id)
        w1 = torch.randint(0, 2, w0.shape, device=DEVICE)
        party.send_torch_tensor_to(sender_id, w1)

        cond = (c.tensor > 0) + 0

        mc = w0 * (1 - cond) + w1 * cond

        party.send_torch_tensor_to(receiver_id, mc)
