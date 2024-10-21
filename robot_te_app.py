import gradio as gr
from gradio_demo_modules.text2img_pipeline import manual_gen_image, select_gen_image_iot
from gradio_demo_modules.prompt_config import action_prompt_dict, place_prompt_dict, fantasy_prompt_dict, Radio_options
import gradio_demo_modules.iot_process as iot_process
from PIL import Image

TE_img = Image.open("/home/Antares.Guo/code/diffusers/gradio_demo_modules/resource/TE.png")


def change_choice(choice):
    if choice == Radio_options[0]:
        return gr.update(choices=[*list(place_prompt_dict.keys())],
                         value=list(place_prompt_dict.keys())[0], visible=True)
    elif choice == Radio_options[1]:
        return gr.update(choices=[*list(action_prompt_dict.keys())],
                         value=list(action_prompt_dict.keys())[0], visible=True)
    elif choice == Radio_options[2]:
        return gr.update(choices=[*list(fantasy_prompt_dict.keys())],
                         value=list(fantasy_prompt_dict.keys())[0], visible=True)
    else:
        return gr.update(visible=False)


with gr.Blocks() as demo:
    # 用markdown语法编辑输出一段话
    # gr.Markdown("# 小特的奇幻之旅")
    gr.Label("小特的奇幻之旅")
    # 设置tab选项卡
    with gr.Tab("模版输入"):
        with gr.Row():
            with gr.Column():
                now = iot_process.time_iot()
                auto_month_select_slider = gr.Slider(1, 12, label="月份", step=1, value=now[0])
                auto_hour_select_slider = gr.Slider(1, 24, label="时间", step=1, value=now[1])
                auto_weather_select_radio = gr.Radio(list(iot_process.weather_dict.keys()), label="天气",
                                                     value=list(iot_process.weather_dict.keys())[0])

            with gr.Column():
                travel_gradio = gr.Radio(Radio_options, label="选择奇幻模式", value=Radio_options[0])
                dropdown_all = gr.Dropdown([*list(place_prompt_dict.keys())], label="选择奇幻内容",
                                           value=list(place_prompt_dict.keys())[0], interactive=True)

                travel_gradio.change(fn=change_choice, inputs=travel_gradio, outputs=dropdown_all)

            with gr.Column():
                gr.Image(value=TE_img, label="我是小特")
                select_act_button = gr.ClearButton(value="小特踏上旅途")

    with gr.Tab("手动输入"):
        # Blocks特有组件，设置所有子组件按垂直排列
        with gr.Column():
            manual_prompt_input = gr.Textbox(label="manual prompt input")
            manual_positive_prompt_input = gr.Textbox(label="manual positive prompt input")
            manual_negative_prompt_input = gr.Textbox(label="manual negative prompt input")

        manual_act_button = gr.ClearButton(value="Generation")

    image_gallery_output = gr.Gallery(
        label="小特奇幻之旅相册", show_label=False, elem_id="gallery"
    ).style(columns=[2], rows=[2], object_fit="contain", height="auto")
    prompt_output = gr.Textbox(label="小特的旅行日记")

    gr.Markdown("特斯联未来城市实验室")

    manual_act_button.add([image_gallery_output])  # it_prompt, positive_prompt_str, negative_prompt_str
    manual_act_button.click(manual_gen_image,
                            inputs=[manual_prompt_input, manual_positive_prompt_input, manual_negative_prompt_input,
                                    ],
                            outputs=[image_gallery_output, prompt_output])

    select_act_button.add([image_gallery_output])
    select_act_button.click(select_gen_image_iot,
                            inputs=[auto_month_select_slider, auto_hour_select_slider, auto_weather_select_radio,
                                    travel_gradio,
                                    dropdown_all, ],
                            outputs=[image_gallery_output, prompt_output])

demo.queue()
demo.launch(share=True, server_name="0.0.0.0", server_port=7878)
