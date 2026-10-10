import numpy as np
import torch
from utils import *

"""
MeZO类，提供了零阶优化方法来代替常规的基于随机梯度下降的方法
"""


class MeZO:
    def __init__(self, model, eps, criterion, z, weight_decay=0, **args):
        self.model = model
        self.eps = eps
        self.args = args
        self.criterion = criterion
        self.weight_decay = weight_decay
        self.z = z
        self.value = 0

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        依据方向z生成\theta,客户端用
        """

        for (name, param), z_element in zip(self.named_parameters_to_optim, self.z):
            param.data = param.data + scaling_factor * z_element * self.eps

    def zo_step(self, inputs, labels):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        根据inputs,计算方向z的投影梯度,客户端用
        """
        args = self.args
        model = self.model
        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # First function evaluation
        with torch.no_grad():
            self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(inputs, labels).item()

        # Second function evaluation
        with torch.no_grad():
            self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(inputs, labels).item()

        self.projected_grad = ((loss1 - loss2) / (2 * self.eps))
        # self.projected_grads.append(self.projected_grad)
        self.value += self.projected_grad

        # Reset model back to its parameters at start of step
        with torch.no_grad():
            self.zo_perturb_parameters(scaling_factor=1)


        # 手动清除临时变量
        del loss1, loss2
        torch.cuda.empty_cache()

        return loss1

    def zo_update(self, learning_rate):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args

        # Reset the random seed for sampling zs

        for (name, param), z_element in zip(self.named_parameters_to_optim, self.z):
            # Resample z
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - learning_rate * (
                            self.projected_grad * z_element + self.weight_decay * param.data)
            else:
                param.data = param.data - learning_rate * (self.projected_grad * z_element)

        # lr_scheduler.step()

    def zo_forward(self, inputs, labels):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model = self.model
        model.eval()
        criterion = self.criterion
        with torch.inference_mode():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        return loss.detach()


# class MeZO_server(MeZO):
#     def __init__(self, device, z, projected_grads, model, lr_decay, received_vecs, learning_rate, eps, criterion,
#                  weight_decay=0,
#                  **args):
#         super(MeZO_server, self).__init__(model, eps, criterion, z, weight_decay, **args)

#         self.device = device
#         self.received_vecs = received_vecs
#         self.model = set_client_from_params(device=self.device, model=self.model(),
#                                             params=self.received_vecs['Params_list'])
#         self.projected_grads = projected_grads
#         self.received_vecs = received_vecs
#         self.z = z
#         self.lr_decay = lr_decay
#         self.learning_rate = learning_rate
#         self.named_parameters_to_optim = []
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.named_parameters_to_optim.append((name, param))

#         self.comm_vecs = {
#             'local_update_list': None,
#             'local_model_param_list': None,
#         }

#     def zo_update_server(self):
#         """
#         Update the parameters with the estimated gradients.
#         服务器用
#         """
#         args = self.args
#         for projected_grad in self.projected_grads:
#             for (name, param), z_element in zip(self.named_parameters_to_optim, self.z):

#                 if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
#                     param.data = param.data - self.learning_rate * (
#                                 projected_grad * z_element + self.weight_decay * param.data)
#                 else:
#                     param.data = param.data - self.learning_rate * (projected_grad * z_element)

#             self.learning_rate = self.learning_rate * self.lr_decay

#         last_state_params_list = get_mdl_params(self.model)
#         self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
#         self.comm_vecs['local_model_param_list'] = last_state_params_list # type: ignore
#         return self.comm_vecs


# class MeZO_norm:
#     def __init__(self, model, eps, criterion, weight_decay=0, **args):
#         # self.zo_random_seed = random_seed
#         self.model = model
#         self.eps = eps
#         self.args = args
#         self.criterion = criterion
#         self.weight_decay = weight_decay

#     def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
#         """
#         Perturb the parameters with random vector z.
#         Input:
#         - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
#         - scaling_factor: theta = theta + scaling_factor * z * eps
#         依据方向z生成\theta,客户端用
#         """

#         # Set the random seed to ensure that we sample the same z for perturbation/update
#         torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
#         for name, param in self.named_parameters_to_optim:
#             z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
#             param.data = param.data + scaling_factor * z * self.eps

#     def zo_step(self, inputs, labels):
#         """
#         Estimate gradient by MeZO. Return the loss from f(theta + z)
#         根据inputs,计算方向z的投影梯度,客户端用
#         """
#         args = self.args
#         model = self.model
#         # What parameters to optimize
#         self.named_parameters_to_optim = []
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 self.named_parameters_to_optim.append((name, param))

#         # Sample the random seed for sampling z
#         self.zo_random_seed = np.random.randint(1000000000)

#         # 保存random_seeds供server更新参数
#         # self.random_seeds.append(self.zo_random_seed)

#         # First function evaluation
#         self.zo_perturb_parameters(scaling_factor=1)
#         loss1 = self.zo_forward(inputs, labels)

#         # Second function evaluation
#         self.zo_perturb_parameters(scaling_factor=-2)
#         loss2 = self.zo_forward(inputs, labels)

#         self.projected_grad = ((loss1 - loss2) / (2 * self.eps)).item()
#         # self.projected_grads.append(self.projected_grad)

#         # Reset model back to its parameters at start of step
#         self.zo_perturb_parameters(scaling_factor=1)

#         return loss1

#     def zo_update(self, learning_rate):
#         """
#         Update the parameters with the estimated gradients.
#         """
#         args = self.args

#         # Reset the random seed for sampling zs
#         torch.manual_seed(self.zo_random_seed)

#         for name, param in self.named_parameters_to_optim:
#             # Resample z
#             z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
#             if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
#                 param.data = param.data - learning_rate * (self.projected_grad * z + self.weight_decay * param.data)
#             else:
#                 param.data = param.data - learning_rate * (self.projected_grad * z)

#     def zo_forward(self, inputs, labels):
#         """
#         Get (no gradient) loss from the model. Dropout is turned off too.
#         """
#         model = self.model
#         model.eval()
#         criterion = self.criterion
#         with torch.inference_mode():
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#         return loss.detach()
