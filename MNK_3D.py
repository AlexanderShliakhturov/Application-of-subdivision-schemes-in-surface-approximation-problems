"""
Моделирование области по облаку точек
"""
import numpy as np
import torch
import torch.nn.functional as F
from itertools import product
import matplotlib.pyplot as plt
from numpy import loadtxt
import plotly.graph_objects as go
import plotly.io as pio
import torch
from time import time
from scipy import ndimage
from skimage import measure
import numpy as np
import pyvista as pv
import plotly.graph_objects as go
# from help_functions.fast_optimization import solve_least_squares_subdivision_CG
# from help_functions.Sub_A_torch import Sub_A_torch
# from help_functions.make_conv_3d_torch import make_conv_3d_torch
pio.renderers.default='browser'


# Отдельная функция, которое строит n-мерное ядро. 
# Потому что функции операторов А и А_T вызываются часто и на каждой итерации им нужно одно и то же ядро.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_kernel(mask: torch.Tensor, dim: int, device="cpu"):
    mask = mask.to(device)

    if dim == 1:
        kernel = mask
    elif dim == 2:
        kernel = torch.outer(mask, mask)
    elif dim == 3:
        kernel = torch.einsum('i,j,k->ijk', mask, mask, mask)
    else:
        raise ValueError("Only 1D, 2D, 3D supported")

    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel


def Sub_A_fast(x, kernel):
    dim = x.ndim
    padding = kernel.shape[-1] // 2

    x = x.unsqueeze(0).unsqueeze(0)

    if dim == 1:
        out = F.conv_transpose1d(x, kernel, stride=2, padding=padding)
    elif dim == 2:
        out = F.conv_transpose2d(x, kernel, stride=2, padding=padding)
    else:
        out = F.conv_transpose3d(x, kernel, stride=2, padding=padding)

    return out.squeeze(0).squeeze(0)


def Sub_AT_fast(x, kernel):
    dim = x.ndim
    padding = kernel.shape[-1] // 2

    x = x.unsqueeze(0).unsqueeze(0)

    # Оператор [(свертка с перевернутым ядром) + (стрелка вниз)] - это обыкновеный conv со Stride = 2
    if dim == 1:
        out = F.conv1d(x, kernel, stride=2, padding=padding)
    elif dim == 2:
        out = F.conv2d(x, kernel, stride=2, padding=padding)
    else:
        out = F.conv3d(x, kernel, stride=2, padding=padding)

    return out.squeeze(0).squeeze(0)


def apply_A_j(x, kernel, j):
    for _ in range(j):
        x = Sub_A_fast(x, kernel)
    return x


def apply_AT_j(x, kernel, j):
    for _ in range(j):
        x = Sub_AT_fast(x, kernel)
    return x


def apply_C_j(x, kernel, j):
    # C = (A^T)^j A^j
    return apply_AT_j(apply_A_j(x, kernel, j), kernel, j)

def solve_least_squares_subdivision_CG(
    z: torch.Tensor,
    mask: torch.Tensor,
    j=1,
    max_iter=200,
    tol=1e-6,
    device="cpu"
):
    device = torch.device(device)

    z = z.float().to(device)
    mask = mask.float().to(device)

    dim = z.ndim
    kernel = build_kernel(mask, dim, device)

    # b = (A^T)^j z
    b = apply_AT_j(z, kernel, j) 
    x = torch.zeros_like(b)

    # r0 = b - Cx0
    r = b.clone()  # потому что x0 = 0
    p = r.clone()

    rs_old = torch.sum(r * r)

    for k in range(max_iter):

        Cp = apply_C_j(p, kernel, j)

        denom = torch.sum(p * Cp)
        alpha = rs_old / denom

        x = x + alpha * p
        r = r - alpha * Cp

        rs_new = torch.sum(r * r)

        grad_norm = torch.sqrt(rs_new)
        print(f"CG ITER {k}: residual = {grad_norm:.6e}")

        if grad_norm < tol:
            print(f"CG Converged at iter {k}")
            break

        beta = rs_new / rs_old
        p = r + beta * p

        rs_old = rs_new

    return x

def make_conv_3d_torch(source: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    source: torch.Tensor, shape (D, H, W)
    kernel: torch.Tensor, shape (kD, kH, kW)

    return:
        result: torch.Tensor, shape (D, H, W)
    """

    source = source.float()
    kernel = kernel.float()
    
    device = source.device
    
    kernel = kernel.to(device)

    # conv3d формат
    # source: (N, C, D, H, W)
    x = source.unsqueeze(0).unsqueeze(0)
    w = kernel.unsqueeze(0).unsqueeze(0)

    pad_d = kernel.shape[0] // 2
    pad_h = kernel.shape[1] // 2
    pad_w = kernel.shape[2] // 2

    # порядок паддингов обязательно (W_left, W_right, H_left, H_right, D_left, D_right)
    x_padded = F.pad(
        x,
        (pad_w, pad_w,
         pad_h, pad_h,
         pad_d, pad_d)
    )

    y = F.conv3d(x_padded, w)
    y = torch.clamp(y, max=1.0)

    return y.squeeze(0).squeeze(0)

def volume_to_mesh(volume: torch.Tensor, level=0.5):
    volume_np = volume.detach().cpu().numpy().astype(np.float32)

    verts, faces, normals, values = measure.marching_cubes(
        volume_np,
        level=level
    )

    return verts, faces, normals

def shadows_visual(input_tensor: torch.Tensor, name: str, layers = 'vertical'):

    surface = input_tensor

    if layers == 'horizontal':

        surface_rot = torch.rot90(surface, k=1, dims=(1,2))
        surface_rot = torch.rot90(surface_rot, k=2, dims=(0,1))
        surface_rot = torch.rot90(surface_rot, k=-1, dims=(0,1))

    elif layers == 'vertical':
        surface_rot = torch.rot90(surface, k=1, dims=(0,1))
    
    else:
        surface_rot = surface


    verts, faces, normals = volume_to_mesh(surface_rot)

    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3), faces]
    ).astype(np.int64)

    mesh = pv.PolyData(verts, faces_pv)
    mesh.point_data["Normals"] = normals

    plotter = pv.Plotter(off_screen=True,
                        window_size=(3000, 3000))
    plotter.add_mesh(
        mesh,
        color="gold",
        smooth_shading=False,
        specular=0.5,
        specular_power=30
    )
    # plotter.view_yz()



    # plotter.show(screenshot="figure_x0.png",
    #              window_size=(3000, 3000))
    # plotter.save_graphic(f"{name}.pdf")
    plotter.screenshot(f"{name}.png")
    plotter.close()
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
print(f"Размерность характеристической функции Z: {Y.shape}")

# x_0_star, _ = Grad(np.zeros(tuple(n)), Y, j, p, N)
x_0_star = solve_least_squares_subdivision_CG(z = torch.from_numpy(Y),
                                              mask=torch.from_numpy(p),
                                              j = j,
                                              max_iter=10,
                                              tol=1
                                              )
x_0_star = x_0_star.numpy()

print(f"Размерность начальной последовательности x_0: {x_0_star.shape}")

x_j = x_0_star.copy()
print('Находим последовательность x_j')
p_torch = build_kernel(torch.from_numpy(p).to(torch.float32), dim = 3)
# p_torch = torch.from_numpy(p).to(torch.float32)

x_j_torch = torch.from_numpy(x_j).to(torch.float32)
for k in range(j):
    # x_j = Sub_A(p, x_j)
    x_j_torch = Sub_A_fast(x_j_torch, p_torch)
    
x_j = x_j_torch.numpy()

print(f"Размерность x_j: {x_j.shape}")

   
# import pickle 
# with open('UIV_x.pkl', 'wb') as f:
#     pickle.dump(x_j, f)

shadows_visual(torch.from_numpy(x_j), name = 'model_screenshot')
    
''' Ошибка '''
# plt.figure()
# plt.plot(np.arange(len(er)), np.array(er), '--r')

x_1 = np.arange(n_1)
x_2 = np.arange(n_2)
x_3 = np.arange(n_3)
X_1, X_2, X_3 = np.meshgrid(x_1, x_2, x_3, indexing='ij')

E = 2

colorscale = [[0, 'rgb(255,255,0)'], [1, 'rgb(255,255,0)']] # желтый
colorscale = [[0, 'rgb(0,255,0)'], [1, 'rgb(0,255,0)']] #зеленый

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
