#%%
import torch

# Input tensor
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Compute the output using the existing computational graph
y = x ** 2
z = 2*y + 1

# def f(x):
#     return torch.stack([x[0]**2-x[1], x[1]**2+x[0]])

# y = f(x)

# def unit_vector_at_position(i, size):
#     vector = torch.zeros(size)  # Create a tensor of zeros
#     vector[i] = 1  # Set the i-th position to 1
#     return vector

# def jacobian(_os, _is):
#     jacobian_matrix = torch.zeros((len(_os), len(_is)))
#     for i in range(len(_os)):
#         jacobian_matrix[i], = torch.autograd.grad(_os, _is, unit_vector_at_position(i, 2), retain_graph=True, create_graph=True)
#     return jacobian_matrix

# print(jacobian(y, x))




# jacobian = torch.vmap(get_vjp(y, x))(I_N)

def jacobian(_os, _is):
    I_N = torch.eye(len(_os))
    def get_vjp(v):
        return torch.autograd.grad(_os, _is, v)
    return torch.vmap(get_vjp)(I_N)

# Print the computed Jacobian matrix
print("Jacobian matrix:")
print(jacobian(z, y))
# %%
