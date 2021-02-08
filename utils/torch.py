# import torch
# from torch.autograd import grad


# # [Reference] https://github.com/ajlangley/trpo-pytorch


# def get_device():
#     """
#     Return a torch.device object. Returns a CUDA device if it is available and
#     a CPU device otherwise.
#     """
#     if torch.cuda.is_available():
#         return torch.device('cuda')
#     else:
#         return torch.device('cpu')


# save_dir = '../../saved-sessions'


# def apply_update(parameterized_fun, update):
#     """
#     Add update to the weights of parameterized_fun

#     Parameters
#     ----------
#     parameterized_fun : torch.nn.Sequential
#         the function approximator to be updated

#     update : torch.FloatTensor
#         a flattened version of the update to be applied
#     """

#     n = 0

#     for param in parameterized_fun.parameters():
#         numel = param.numel()
#         param_update = update[n:n + numel].view(param.size())
#         param.data += param_update
#         n += numel


# def flatten(vecs):
#     """
#     Return an unrolled, concatenated copy of vecs

#     Parameters
#     ----------
#     vecs : list
#         a list of Pytorch Tensor objects

#     Returns
#     -------
#     flattened : torch.FloatTensor
#         the flattened version of vecs
#     """

#     flattened = torch.cat([v.view(-1) for v in vecs])

#     return flattened


# def flat_grad(functional_output, inputs, retain_graph=False, create_graph=False):
#     """
#     Return a flattened view of the gradients of functional_output w.r.t. inputs

#     Parameters
#     ----------
#     functional_output : torch.FloatTensor
#         The output of the function for which the gradient is to be calculated

#     inputs : torch.FloatTensor (with requires_grad=True)
#         the variables w.r.t. which the gradient will be computed

#     retain_graph : bool
#         whether to keep the computational graph in memory after computing the
#         gradient (not required if create_graph is True)

#     create_graph : bool
#         whether to create a computational graph of the gradient computation
#         itself

#     Return
#     ------
#     flat_grads : torch.FloatTensor
#         a flattened view of the gradients of functional_output w.r.t. inputs
#     """

#     if create_graph:
#         retain_graph = True

#     grads = grad(functional_output, inputs, retain_graph=retain_graph, create_graph=create_graph)
#     flat_grads = flatten(grads)

#     return flat_grads


# def get_flat_params(parameterized_fun):
#     """
#     Get a flattened view of the parameters of a function approximator

#     Parameters
#     ----------
#     parameterized_fun : torch.nn.Sequential
#         the function approximator for which the parameters are to be returned

#     Returns
#     -------
#     flat_params : torch.FloatTensor
#         a flattened view of the parameters of parameterized_fun
#     """
#     parameters = parameterized_fun.parameters()
#     flat_params = flatten([param.view(-1) for param in parameters])

#     return flat_params
