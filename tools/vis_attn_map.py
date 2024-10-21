import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
import torch

attn_map_path = '/home/Antares.Guo/tmp_folder_sd2_layout'
save_fig_path = '/home/Antares.Guo/tmp_folder_sd2_layout_fig'

prompt = "a dog and a cat sitting on the grass, sunshine"
key_idx = [2, 5]

if not os.path.exists(save_fig_path):
    os.mkdir(save_fig_path)

file_list = os.listdir(attn_map_path)

# vis cross attn only
for it_file in file_list:
    if it_file.find("npy") == -1:
        continue
    if it_file.find("self") != -1:
        continue

    file_name = os.path.join(attn_map_path, it_file)
    np_array = np.load(file_name)

    head_num, wh, token = np_array.shape

    side_len = int(np.sqrt(wh))

    np_array = np_array.reshape((head_num, side_len, side_len, token))
    # need to show, uncon, 77tokens
    # vis uncon sum
    uncon_mean_array = np.mean(np_array[:int(head_num / 2)], axis=0)  # w,h,77
    con_mean_array = np.mean(np_array[int(head_num / 2):], axis=0)  # w,h,77

    # norm each token
    uncon_list = []
    con_list = []
    for i in range(token):
        uncon_max = np.max(uncon_mean_array[:, :, i])+1e-10
        uncon_list.append(torch.from_numpy(uncon_mean_array[:, :, i] / uncon_max).unsqueeze(0))

        con_max = np.max(con_mean_array[:, :, i])+1e-10
        con_list.append(torch.from_numpy(con_mean_array[:, :, i] / con_max).unsqueeze(0))

    # uncon_mean_tensor = torch.from_numpy(uncon_mean_array)
    # con_mean_tensor = torch.from_numpy(con_mean_array)

    # grid_uncon = make_grid(uncon_mean_tensor.permute((2,0,1)).unsqueeze(1),nrow=10)
    # grid_con = make_grid(con_mean_tensor.permute((2, 0, 1)).unsqueeze(1), nrow=10)

    grid_uncon = make_grid(uncon_list, nrow=10)
    grid_con = make_grid(con_list, nrow=10)

    # name
    s_n, s_e = os.path.splitext(it_file)
    save_name = os.path.join(save_fig_path, s_n + '_uncond.png')
    plt.figure(1)
    plt.imshow(grid_uncon[0])
    plt.savefig(save_name)
    plt.close(1)

    save_name = os.path.join(save_fig_path, s_n + '_cond.png')
    plt.figure(1)
    plt.imshow(grid_con[0])
    plt.savefig(save_name)
    plt.close(1)
