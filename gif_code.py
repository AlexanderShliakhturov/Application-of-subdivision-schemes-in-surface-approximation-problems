# level = 400
# for level in tqdm(range(12)):
#     make_scatter_with_plane(budda_tensor_3D, f"./gifs/budda/3d_layer_plane/src/plane_{level:04d}", level, dpi=120)
# make_scatter_with_plane(budda_tensor_3D, f"./gifs/budda/3d_layer_plane/plane_{level}", level)


# save_frames_folder = Path("./gifs/budda/comparison/src")
# save_frames_folder.mkdir(parents=True, exist_ok=True)

# folder_layer = Path("./gifs/budda/3d_layer_plane/src")
# folder_border = Path("./gifs/budda/border_conv/src")
# folder_filled = Path("./gifs/budda/filled/src")


# layer_files = sorted(folder_layer.glob("*"))
# border_files = sorted(folder_border.glob("*"))
# filled_files = sorted(folder_filled.glob("*"))

# if not (
#     len(layer_files) == len(border_files) == len(filled_files)
# ):
#     raise ValueError(
#         "Количество файлов в папках отличается:\n"
#         f"layer_plane: {len(layer_files)}\n"
#         f"border_conv: {len(border_files)}\n"
#         f"filled: {len(filled_files)}"
#     )

# print(f"Найдено кадров: {len(layer_files)}")

# frames = []

# for idx, (img1_path, img2_path, img3_path) in tqdm(enumerate(
#     zip(layer_files, border_files, filled_files)
# )):
#     img1 = Image.open(img1_path).convert("RGB")
#     img2 = Image.open(img2_path).convert("RGB")
#     img3 = Image.open(img3_path).convert("RGB")

#     # Приводим к одинаковой высоте
#     target_height = min(
#         img1.height,
#         img2.height,
#         img3.height
#     )

#     def resize_keep_ratio(img):
#         new_width = int(img.width * target_height / img.height)
#         return img.resize((new_width, target_height))

#     img1 = resize_keep_ratio(img1)
#     img2 = resize_keep_ratio(img2)
#     img3 = resize_keep_ratio(img3)

#     total_width = img1.width + img2.width + img3.width

#     combined = Image.new(
#         "RGB",
#         (total_width, target_height),
#         color="white"
#     )

#     x_offset = 0
#     for img in [img1, img2, img3]:
#         combined.paste(img, (x_offset, 0))
#         x_offset += img.width

#     frames.append(np.array(combined))
#     combined.save(
#     save_frames_folder / f"frame_{idx:04d}.png"
#     )
    
#     # if idx == 100:
#     #     break

#     # if idx % 10 == 0:
#     #     print(f"Обработано: {idx}/{len(layer_files)}")

# output_path = "./gifs/budda/comparison/final_comparison_speed_x6.gif"
# fps = 100
# duration = 1 / fps

# # Загружаем кадры из папки save_frames_folder
# frame_files = sorted(save_frames_folder.glob("*.png"))

# frames = []

# for frame_path in frame_files[::6]:
#     frame = imageio.imread(frame_path)
#     frames.append(frame)

# imageio.mimsave(
#     output_path,
#     frames,
#     duration=duration,
#     loop=0
# )

# print(f"\nGIF сохранён: {output_path}")



# import os
# from pathlib import Path

# import matplotlib.pyplot as plt
# import imageio.v2 as imageio

# tensor_3d = budda_conv_result_3D
# output_frames_dir = Path("./gifs/border_conv/src")
# output_gif_path = Path("./gifs/border_conv/result_speed.gif")

# cmap = "viridis"
# fps = 100
# duration = 1 / fps
# dpi = 120


# output_frames_dir.mkdir(parents=True, exist_ok=True)
# output_gif_path.parent.mkdir(parents=True, exist_ok=True)

# vmin = tensor_3d.min()
# vmax = tensor_3d.max()

# num_slices = tensor_3d.shape[2]
# saved_frames = []

# for i in range(num_slices):
#     fig = plt.figure(figsize=(6, 5))

#     plt.imshow(
#         tensor_3d[:, :, i],
#         cmap=cmap,
#         aspect="auto",
#         origin="lower",
#         vmin=vmin,
#         vmax=vmax
#     )

#     plt.colorbar(label="Значение")
#     plt.xlabel("X координата")
#     plt.ylabel("Y координата")
#     plt.title(f"Сечение Z = {i}")

#     plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
#     plt.tight_layout()

#     frame_path = output_frames_dir / f"slice_{i:04d}.png"
#     plt.savefig(frame_path, dpi=dpi)
#     plt.close(fig)

#     saved_frames.append(frame_path)

# print(f"Сохранено {len(saved_frames)} сечений в: {output_frames_dir}")

# # Собираем GIF
# images = [imageio.imread(frame_path) for frame_path in saved_frames]
# imageio.mimsave(output_gif_path, images, duration=duration, loop=0)

# print(f"GIF сохранена в: {output_gif_path}")

# output_gif_path = Path("./gifs/border_conv/result_speed_x4.gif")
# images = [imageio.imread(frame_path) for frame_path in saved_frames[::4]]
# imageio.mimsave(output_gif_path, images, duration=duration/16, loop=0)

# print(f"GIF сохранена в: {output_gif_path}")



# import os
# from pathlib import Path

# import matplotlib.pyplot as plt
# import imageio.v2 as imageio


# tensor_3d = budda_conv_result_3D_filled
# output_frames_dir = Path("./gifs/budda/filled/src")

# cmap = "viridis"
# fps = 10
# duration = 1 / fps
# dpi = 120


# output_frames_dir.mkdir(parents=True, exist_ok=True)
# output_gif_path.parent.mkdir(parents=True, exist_ok=True)

# vmin = tensor_3d.min()
# vmax = tensor_3d.max()

# num_slices = tensor_3d.shape[2]
# saved_frames = []

# for i in range(num_slices):
#     fig = plt.figure(figsize=(5, 4))

#     plt.imshow(
#         tensor_3d[:, :, i],
#         cmap=cmap,
#         aspect="auto",
#         origin="lower",
#         vmin=vmin,
#         vmax=vmax
#     )

#     plt.colorbar(label="Значение")
#     plt.xlabel("X координата")
#     plt.ylabel("Y координата")
#     plt.title(f"Сечение Z = {i}")

#     plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
#     plt.tight_layout()

#     frame_path = output_frames_dir / f"slice_{i:04d}.png"
#     plt.savefig(frame_path, dpi=dpi)
#     plt.close(fig)

#     saved_frames.append(frame_path)

# print(f"Сохранено {len(saved_frames)} сечений в: {output_frames_dir}")