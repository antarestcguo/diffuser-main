from gradio_demo_modules.base_pipeline import basepipe
import cv2
from gradio_demo_modules.prompt_config import default_positive_prompt, default_negative_prompt
from gradio_demo_modules.prompt_config import action_prompt_dict, place_prompt_dict, fantasy_prompt_dict, Radio_options
from gradio_demo_modules.iot_process import gen_iot_prompt


def manual_gen_image(it_prompt, positive_prompt_str, negative_prompt_str, num_inference_steps=50, guidance_scale=7.5,
                     gen_num=12, bz=6):
    output_image = []
    inference_time = int(gen_num / bz)

    input_prompt = it_prompt.replace("unique_identifier",
                                     basepipe.unique_identifier) + ',' + positive_prompt_str

    for tmp_i in range(inference_time):
        image = basepipe.pipe(input_prompt, num_inference_steps=num_inference_steps,
                              guidance_scale=guidance_scale,
                              negative_prompt=negative_prompt_str, num_images_per_prompt=bz
                              ).images

        for idx, it_img in enumerate(image):
            output_image.append((it_img, "相片_{}.png".format(tmp_i * bz + idx)))

    return output_image, input_prompt


def select_gen_image_iot(month, hour, Weather, travel_mode, dropdown_all):
    gen_num = 12
    bz = 6
    num_inference_steps = 50
    guidance_scale = 7.5
    output_image = []
    inference_time = int(gen_num / bz)
    base_prompt = gen_select_prompt(travel_mode, dropdown_all)
    iot_prompt = gen_iot_prompt(hour, month, Weather, travel_mode)
    # it_prompt = "A unique_identifier in the Egyptian Pyramids in Egypt,"
    input_prompt = base_prompt.replace("unique_identifier",
                                       basepipe.unique_identifier) + ',' + default_positive_prompt
    if iot_prompt != "":
        input_prompt = input_prompt + "," + iot_prompt

    for tmp_i in range(inference_time):
        image = basepipe.pipe(input_prompt, num_inference_steps=num_inference_steps,
                              guidance_scale=guidance_scale,
                              negative_prompt=default_negative_prompt, num_images_per_prompt=bz
                              ).images

        for idx, it_img in enumerate(image):
            output_image.append((it_img, "相片_{}.png".format(tmp_i * bz + idx)))

    return output_image, input_prompt


def gen_select_prompt(travel_mode, dropdown_all):
    base_prompt = ""
    if travel_mode == Radio_options[0]:
        base_prompt = place_prompt_dict[dropdown_all]
    elif travel_mode == Radio_options[1]:
        base_prompt = action_prompt_dict[dropdown_all]
    elif travel_mode == Radio_options[2]:
        base_prompt = fantasy_prompt_dict[dropdown_all]
    return base_prompt


def tmp_read_img(img_path):
    img = cv2.imread(img_path)
    return [(img, "img.png")]
