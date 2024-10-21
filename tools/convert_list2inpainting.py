import os

src_file = "/home/Antares.Guo/data/shuffle_caption_15_list.txt"
dst_file = "/home/Antares.Guo/data/shuffle_caption_15_list_Finpainting.txt"

with open(src_file, 'r') as f_r, open(dst_file, 'w') as f_w:
    for line in f_r.readlines():
        tokens = line.strip().split('\t')

        img_name = tokens[0]
        prompt = tokens[1]

        s_n, s_e = os.path.splitext(img_name)

        mask_name = s_n + '.jpg'

        f_w.write(img_name + '\t' + prompt + '\t' + mask_name + '\n')
