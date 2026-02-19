"""
Моделирование области по облаку точек
"""
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

from numpy import loadtxt
import plotly.graph_objects as go
import plotly.io as pio
import torch
from time import time
from scipy import ndimage
from help_functions.fast_optimization import solve_least_squares_subdivision_CG
from help_functions.Sub_A_torch import Sub_A_torch as Sub_A_my
from help_functions.make_conv_3d_torch import make_conv_3d_torch
pio.renderers.default='browser'

# def Sub_A(p, x_0):
#     # Схема подразделений через преобразование Фурье.
#     # Здесь предполагается, что последовательность x_0
#     # периодическая и задана на периоде [0; n_1 - 1] x ...
#     # p -- маска -- массив numpy размера q < min n_i
#     s = len(x_0.shape)
#     if (s == 1):
#         a = p.copy()
#     elif (s == 2):
#         a = np.tensordot(p, p, axes = 0)
#     else:
#         a = np.tensordot(p, np.tensordot(p, p, axes = 0), axes = 0)
#     x = x_0.copy() 
    
#     Up_x = np.zeros(tuple(2 * np.array(x.shape)))
#     DFT_a = np.zeros(Up_x.shape)
#     if (s == 1):
#         Up_x[::2] = x[:]
#         DFT_a[:a.shape[0]] = a[:]
#         DFT_a = np.fft.fftn(DFT_a)
#     elif (s == 2):
#         Up_x[::2, ::2] = x[:, :]
#         DFT_a[:a.shape[0], :a.shape[1]] = a[:, :]
#         DFT_a = np.fft.fftn(DFT_a)
#     else:
#         Up_x[::2, ::2, ::2] = x[:, :, :]
#         DFT_a[:a.shape[0], :a.shape[1], :a.shape[2]] = a[:, :, :]
#         DFT_a = np.fft.fftn(DFT_a)
#     x = np.fft.ifftn(DFT_a * np.fft.fftn(Up_x)).real
#     return x

# def ReversKerDFT(H):
#     '''
#     Переворот ядра H, supp H \subset [0; q_1 - 1] x ... x [0; q_s - 1]
#     Получаем ядро H^- = H_{q - ., q - .}, с носителем [0; q_1 - 1] x ... x [0; q_s - 1]
#     '''
#     s = len(H.shape)
#     arrays_i = []
#     for i in range(s):
#         T = np.arange(H.shape[i])
#         T[1:] = H.shape[i] - T[1:]
#         arrays_i.append(list(T))
#     I_i = np.array(list(product(*arrays_i))) # всевозможные наборы индексов [i_1,...,i_m]
#     U = []
#     for i in range(s):
#         U.append(I_i[:, i])
#     U = tuple(U)
#     ker = H[U].reshape(H.shape)
#     return ker

# def Sub_AT(p, x):
#     # Преобразование z^0 = \downarrow ((a^-) * x)
#     # Ему должно соответствовать преобразование A^T x
#     s = len(x.shape)
#     if (s == 1):
#         a = p.copy()
#     elif (s == 2):
#         a = np.tensordot(p, p, axes = 0)
#     else:
#         a = np.tensordot(p, np.tensordot(p, p, axes = 0), axes = 0)
    
#     x_0 = x.copy()
#     DFT_a = np.zeros(x_0.shape)
#     if (s == 1):
#         DFT_a[:a.shape[0]] = a[:]
#         DFT_a = np.fft.fftn(ReversKerDFT(DFT_a)) # Соответствует ядру a^-
#     elif (s == 2):
#         DFT_a[:a.shape[0], :a.shape[1]] = a[:, :]
#         DFT_a = np.fft.fftn(ReversKerDFT(DFT_a))  # Соответствует ядру a^-
#     else:
#         DFT_a[:a.shape[0], :a.shape[1], :a.shape[2]] = a[:, :, :]
#         DFT_a = np.fft.fftn(ReversKerDFT(DFT_a))  # Соответствует ядру a^-
#     x_0 = np.fft.ifftn(DFT_a * np.fft.fftn(x_0)).real
    
#     # Преобразование \downarrow
#     if (s == 1):
#         x_0 = x_0[::2]
#     elif (s == 2):
#         x_0 = x_0[::2, ::2]
#     else:
#         x_0 = x_0[::2, ::2, ::2]
            
#     return x_0

# def ATA_j(x, p, j):
# # Нахождение преобразования B x = A^T A x, где матрица преобразования x^{i} = p * (\uparrow x^{i-1}), i = 1,2,...,j
#     ATA_x = x.copy()
#     for k in range(j):
#         ATA_x = Sub_A(p, ATA_x)
#     for k in range(j):
#         ATA_x = Sub_AT(p, ATA_x)
#     return ATA_x

# def AT_j(x, p, j):
# # Нахождение преобразования B x = A^T A x, где матрица преобразования x^{i} = p * (\uparrow x^{i-1}), i = 1,2,...,j
#     AT_x = x.copy()
#     for k in range(j):
#         AT_x = Sub_AT(p, AT_x)
#     return AT_x

# def Grad(x_0, y, j, p, N):
# # Реализация гадиентного спуска. N итераций
    
#     b = AT_j(y, p, j) - ATA_j(x_0, p, j)
#     r_k = b.copy()
#     x_k = np.zeros(x_0.shape)
#     er = []
#     for k in range(1, N, 1):
#         print('Итерация ', k)
#         alpha_k = np.sum(r_k * r_k) / np.sum(r_k * ATA_j(r_k, p, j))
#         x_k += alpha_k * r_k
#         r_k = b - ATA_j(x_k, p, j)
#         er.append(0.5 * np.sum(r_k ** 2))
#     return [x_k + x_0, er]


# def Int(I):
#     ''' Морфологическое заполнение области '''
#     T_2 = np.nonzero(I == 1)
#     x = [0, 0]
#     stack = [x]
#     while (len(stack) > 0):
#         x = stack[0]
#         if ((x[0] >= 0) and (x[0] <= I.shape[0] - 1) and (x[1] >= 0) and (x[1] <= I.shape[1] - 1)):
#             if (I[x[0], x[1]] == 0):
#                 I[x[0], x[1]] = 1
#                 stack.remove(x)
#                 stack.append([x[0] - 1, x[1]])
#                 stack.append([x[0] + 1, x[1]])
#                 stack.append([x[0], x[1] - 1])
#                 stack.append([x[0], x[1] + 1])
#             else:
#                 stack.remove(x)
#         else:
#             stack.remove(x)
#     T_0 = np.nonzero(I == 0)
#     T_1 = np.nonzero(I == 1)
#     I[T_0] = 1
#     I[T_1] = 0
#     I[T_2] = 1
#     return I
    
# def ShowI(J):
#     plt.figure(figsize=(5, 5))
#     plt.imshow(J, cmap = 'gray')
    
def Border(r, j):
    d = r // (2 ** j)
    l = r % (2 ** j)
    if (l == 0):
        m = d
    else:
        m = d + 1
    return m


u = np.array([1/4, 3/4, 3/4, 1/4, 0])
#u = np.array([0.5, 1, 0.5])
#u = np.array([1/8, 4/8, 6/8, 4/8, 1/8])
a = np.tensordot(u, np.tensordot(u, u, axes = 0), axes = 0)
r = np.int64((a.shape[0] - 1) / 2)    
    
#data = loadtxt('dragon.txt')
data = loadtxt('./models/BuddaAll.txt')
#data = np.int64(data * 1500)
data = np.int64(data * 3500)

E = 2
# fig = go.Figure(data=[go.Scatter3d(x = data[::E, 0], y = data[::E, 1], z = data[::E, 2], mode = 'markers',  
#                                    marker = dict(size = 1, colorscale = 'Earth', opacity = 0.8))])

# fig.update_layout(scene = dict(xaxis = dict(visible=False), yaxis = dict(visible=False), zaxis =dict(visible=False)))
# fig.show()

delta = 3 # Размер ядра (2\delta + 1) x (2\delta + 1)

# Размер промежутка в R^3
# На всякий случай доплонительный отступ от границы для того чтобы убрать всякие граничные эффекты
delta_dop = 4
k_10 = data[:, 0].min() - delta - delta_dop
k_11 = data[:, 0].max() + delta + delta_dop
k_20 = data[:, 1].min() - delta - delta_dop
k_21 = data[:, 1].max() + delta + delta_dop
k_30 = data[:, 2].min() - delta - delta_dop
k_31 = data[:, 2].max() + delta + delta_dop

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

X_delta = torch.zeros((k_11 - k_10 + 1, k_21 - k_20 + 1, k_31 - k_30 + 1)).to(device)
X_delta[data[:, 0] - k_10, data[:, 1] - k_20, data[:, 2] - k_30] = 1
del data

# b = torch.zeros((2 * delta + 1, 2 * delta + 1, 2 * delta + 1)).to(device)
# for i_1 in range(-delta, delta + 1, 1):
#     for i_2 in range(-delta, delta + 1, 1):
#         for i_3 in range(-delta, delta + 1, 1):
#             if (i_1 ** 2 + i_2 ** 2 + i_3 ** 2 <= delta ** 2):
#                 b[i_1 + delta, i_2 + delta, i_3 + delta] = 1


start = time()
# ker = torch.zeros(X_delta.shape).to(device)
# ker[:(2 * delta + 1), :(2 * delta + 1), :(2 * delta + 1)] = b[:, :, :]
# X_delta = torch.fft.fftn(ker) * torch.fft.fftn(X_delta)
# X_delta = torch.fft.ifftn(X_delta).cpu().real.numpy()
# X_delta[np.nonzero(X_delta > 0.5)] = 1
# X_delta[np.nonzero(X_delta <= 0.5)] = 0

# del ker
finish = time()

def spherical_kernel_3d(radius: int, dim = 3) -> torch.Tensor:
    size = 2 * radius + 1
    center = radius
    
    if dim == 3:
        z, y, x = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        torch.arange(size),
        indexing='ij'
    )
        dist_sq = (x - center)**2 + (y - center)**2 + (z - center)**2
        kernel = (dist_sq <= radius**2).float()
    elif dim == 2:
        y, x = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        indexing='ij'
    )
        dist_sq = (x - center)**2 + (y - center)**2
        kernel = (dist_sq <= radius**2).float()
        
    return kernel


radius = 3 
kernel_3D = spherical_kernel_3d(radius)
X_delta = make_conv_3d_torch(X_delta, kernel=kernel_3D)

X_delta = X_delta.cpu().numpy()

print(f"Вычисление заняло {finish - start:0.4f} секунд")

print('Морфологическое заполнение ...')

def morph_fill_fast(I: torch.Tensor) -> torch.Tensor:
    """
    Flood fill от (0,0) или (0,0,0)
    """

    I_np = I

    mask = (I_np == 0)

    # Реконструкция от стартовой точки
    seed = np.zeros_like(mask)
    seed[(0,) * I_np.ndim] = mask[(0,) * I_np.ndim]

    filled = ndimage.binary_propagation(seed, mask=mask)

    result = 1- filled 
    return result

# for i in range(X_delta.shape[1]):
#     if (np.sum(X_delta[:, i, :]) > 0):
#         X_delta[:, i, :] = Int(X_delta[:, i, :].copy())
        
X_delta = morph_fill_fast(X_delta)
        
r_1, r_2, r_3 = X_delta.shape
j = 2
m_1 = Border(r_1, j)
m_2 = Border(r_2, j)
m_3 = Border(r_3, j)
n_1 = np.int64((2 ** j) * m_1)
n_2 = np.int64((2 ** j) * m_2)
n_3 = np.int64((2 ** j) * m_3)
Y = np.zeros((n_1, n_2, n_3))
# Вкладываем X_delta в Y по возможности по центру:
l_1 = (n_1 - r_1) // 2
l_2 = (n_2 - r_2) // 2
l_3 = (n_3 - r_3) // 2
Y[l_1:(l_1 + r_1), l_2:(l_2 + r_2), l_3:(l_3 + r_3)] = X_delta[:, :, :]

# import pickle
# def save_obj(obj, name):
#     '''  Запись в файл '''
#     pickle.dump(obj, open(name, 'wb'))

# save_obj({'Y' : Y}, 'Y.data')

# import pickle
# def load_obj(name):
#     ''' Чтение из файла'''
#     obj = pickle.load(open(name, 'rb' ))
#     return obj
# Y = load_obj('Y.data')['Y']

x_1 = np.arange(n_1)
x_2 = np.arange(n_2)
x_3 = np.arange(n_3)
X_1, X_2, X_3 = np.meshgrid(x_1, x_2, x_3, indexing='ij')

E = 2

#colorscale = [[0, 'rgb(255,255,0)'], [1, 'rgb(255,255,0)']] # желтый
colorscale = [[0, 'rgb(0,255,0)'], [1, 'rgb(0,255,0)']] #зеленый

# fig = go.Figure(data = go.Isosurface(x = X_1[::E, ::E, ::E].flatten(),
#                                      y = X_2[::E, ::E, ::E].flatten(),
#                                      z = X_3[::E, ::E, ::E].flatten(),
#                                      value = Y[::E, ::E, ::E].flatten(),
#                                      colorscale = colorscale, #colorscale = 'rdbu', #'plotly3', #'matter', #'oranges', #'edge',
#                                      isomin = 0.8,
#                                      isomax = 1,
#                                      showscale = True,
#                                      caps = dict(x_show = False, y_show = False)))

# fig.update_layout(scene = dict(xaxis = dict(visible=False), yaxis = dict(visible=False), zaxis =dict(visible=False)))
# fig.update_layout(coloraxis_showscale=False)
# fig.update_xaxes(showticklabels=False)
# fig.update_yaxes(showticklabels=False)
# fig.show()


# ========== Оптимизационная часть =================
n = np.array([m_1, m_2, m_3], dtype = np.int64)
p = np.array([1 / 4, 3 / 4, 3 / 4, 1 / 4, 0]) # np.array([1 / 8, 4 / 8, 6 / 8, 4 / 8, 1 / 8])
print('Носитель начальной последовательности ', n)

N = 6


# x_0_star, _ = Grad(np.zeros(tuple(n)), Y, j, p, N)
x_0_star = solve_least_squares_subdivision_CG(z = torch.from_numpy(Y),
                                              mask=torch.from_numpy(p),
                                              j = 2,
                                              max_iter=10,
                                              tol=1
                                              )
x_0_star = x_0_star.numpy()

x_j = x_0_star.copy()
print('Находим последовательность x_j')
p_torch = torch.from_numpy(p).to(torch.float32)
x_j_torch = torch.from_numpy(x_j).to(torch.float32)
for k in range(j):
    # x_j = Sub_A(p, x_j)
    x_j = Sub_A_my(p_torch, x_j_torch)
    
x_j = x_j.numpy()
   
import pickle 
with open('UIV_x.pkl', 'wb') as f:
    pickle.dump(x_j, f)
    
''' Ошибка '''
# plt.figure()
# plt.plot(np.arange(len(er)), np.array(er), '--r')

x_1 = np.arange(n_1)
x_2 = np.arange(n_2)
x_3 = np.arange(n_3)
X_1, X_2, X_3 = np.meshgrid(x_1, x_2, x_3, indexing='ij')

E = 2

colorscale = [[0, 'rgb(255,255,0)'], [1, 'rgb(255,255,0)']] # желтый
#colorscale = [[0, 'rgb(0,255,0)'], [1, 'rgb(0,255,0)']] #зеленый

fig = go.Figure(data = go.Isosurface(x = X_1[::E, ::E, ::E].flatten(),
                                     y = X_2[::E, ::E, ::E].flatten(),
                                     z = X_3[::E, ::E, ::E].flatten(),
                                     value = x_j[::E, ::E, ::E].flatten(),
                                     colorscale = colorscale, #colorscale = 'rdbu', #'plotly3', #'matter', #'oranges', #'edge',
                                     isomin = 0.6,
                                     isomax = 1.5,
                                     showscale = True,
                                     caps = dict(x_show = False, y_show = False)))

fig.update_layout(scene = dict(xaxis = dict(visible=False), yaxis = dict(visible=False), zaxis =dict(visible=False)))
fig.update_layout(coloraxis_showscale=False)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()
fig.write_image("my_isosurface.png", width=3000, height=3000, scale=1)
