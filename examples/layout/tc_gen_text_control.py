from PIL import Image, ImageFont, ImageDraw
import os

save_path = "/home/Antares.Guo/tmp_img/layout/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# pre define
image_size = (512, 768)  # w,h
bg_color = tuple([255, 255, 255])
text_pos_bbox = (40, 120, 432, 768 - 40)  # x,y,w,h
main_text_font_size = 25
title_text_font_size = 60

font_color = (0, 0, 0)

line_space = 0  # 行间距
main_text_font_file = "./HYQiHei-55S.ttf"
title_text_font_file = "./YeZiGongChangAoYeHei-2.ttf"

input_str = "初秋的清晨，微风轻拂，带来一丝凉意。太阳初升，橙色的光辉穿透薄薄的云层，映照在草地和树叶上，如金色的绒毯覆盖大地。晨露还未消散，微微的湿润感在空气中弥漫。鸟儿啁啾，夹杂着清澈的溪水声，唤醒了大地的宁静。\n中午时分，初秋的阳光逐渐升高，洒在大地上，温暖而明媚。树影婆娑，微风吹拂，午后的光线透过树叶投下斑驳的斑影。田野中，金黄的麦浪随风摇曳，宛如一片金色的海洋。空气中荡漾着暖意，午间的宁静仿佛时间悠缓流淌。\n傍晚时分，初秋的天空被染上了温柔的橙红色。夕阳西下，余晖映照在云彩和树梢上，创造出一幅如诗如画的画面。微风渐凉，带走了一天的暖意，大地在夜幕降临前沐浴在宁静而祥和的氛围中。这时分，一种淡淡的禅意弥漫在初秋的傍晚里。"

title_input_str = "初秋"

# bg_img
bg_img = Image.new('RGB', image_size, bg_color)
myfont = ImageFont.truetype(main_text_font_file, size=main_text_font_size)
title_font = ImageFont.truetype(title_text_font_file, size=title_text_font_size)

# get cha size
# _,_,font_w, font_h = myfont.getsize(input_str[0])  # font_w,font_h = font_size


def split_line_tokens(text, text_bbox_w, font_w):
    cha_num_per_line = text_bbox_w // font_w

    start_idx = 0
    end_idx = cha_num_per_line
    token_list = []
    while start_idx < len(text):
        token_list.append(text[start_idx:end_idx])
        start_idx += cha_num_per_line
        end_idx += cha_num_per_line
        if end_idx >= len(text):
            end_idx = len(text)
    return token_list


tlist = input_str.split("\n")
draw = ImageDraw.Draw(bg_img)
b_Incomplete = False
start_x = text_pos_bbox[0]  # x
start_y = text_pos_bbox[1]  # y
for t_str in tlist:
    if b_Incomplete:
        break
    tokens = split_line_tokens("    " + t_str, text_pos_bbox[2], main_text_font_size)
    for i, it_token in enumerate(tokens):
        tmp_x, tmp_y, tmp_w, tmp_h = myfont.getbbox(it_token)
        print(it_token, "str_w,h:", tmp_x, tmp_y, tmp_w, tmp_h)
        # start_y = text_pos_bbox[1] + i * tmp_h + line_space  # y
        end_y = start_y + tmp_h
        if end_y > text_pos_bbox[3]:
            print("Incomplete writing")
            b_Incomplete = True
            break
        draw.text((start_x, start_y), it_token, font_color, font=myfont, align='center')
        start_y = end_y + line_space

# draw title
_, _, title_w, title_h = title_font.getbbox(title_input_str)
print(title_input_str, "str_w,h:", title_w, title_h)
draw.text((int(image_size[0] / 2 - title_w / 2), 40), title_input_str, font_color, font=title_font, align='center')

bg_img.save(os.path.join(save_path, "tmp.jpg"))
