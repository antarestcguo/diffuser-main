import gradio as gr
from gradio_demo_modules.text2img_pipeline import manual_gen_image, select_gen_image_iot
from gradio_demo_modules.prompt_config import action_prompt_dict, place_prompt_dict, fantasy_prompt_dict, radio_options
import gradio_demo_modules.iot_process as iot_process
from IPython.display import display

background_image = "/home/Antares.Guo/code/diffusers/background.png"
style = f"""
    .gradio-block {{
        background-image: url({background_image});
        background-size: cover;
    }}
"""
background_color = ".gradio-container {background-color: red}"
background_style = """
.gradio-container {background: url('file=/home/Antares.Guo/code/diffusers/background.png')}

.gradio-container .markdown-container h1 {
    color: white;
}
"""

def change_choice(choice):
    if choice == radio_options[0]:
        return gr.update(choices=[*list(place_prompt_dict.keys())],
                                                 value=list(place_prompt_dict.keys())[0],visible=True)
    elif choice == radio_options[1]:
        return gr.update(choices=[*list(action_prompt_dict.keys())],
                                                  value=list(action_prompt_dict.keys())[0],visible=True)
    elif choice == radio_options[2]:
        return gr.update(choices=[*list(fantasy_prompt_dict.keys())],
                                                    value=list(fantasy_prompt_dict.keys())[0],visible=True)
    else:
        return gr.update(visible=False)
with gr.Blocks() as demo:
    # 用markdown语法编辑输出一段话
    gr.Markdown("# 小特的奇幻之旅")

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
                travel_gradio = gr.Radio(radio_options, label="选择模式", value=radio_options[0])
                with gr.Row():
                    dropdown_place = gr.Dropdown([*list(place_prompt_dict.keys())], label="选择游玩地点",
                                                 value=list(place_prompt_dict.keys())[0])
                    dropdown_action = gr.Dropdown([*list(action_prompt_dict.keys())], label="选择运动项目",
                                                  value=list(action_prompt_dict.keys())[0])
                    dropdown_fantancy = gr.Dropdown([*list(fantasy_prompt_dict.keys())], label="选择探索方式",
                                                    value=list(fantasy_prompt_dict.keys())[0])

                    dropdown_all = gr.Dropdown([], label="选择",
                                                    value="",interactive=True)

                travel_gradio.change(fn=change_choice, inputs=travel_gradio, outputs=dropdown_all)

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
    prompt_output = gr.Textbox(label="小特的履行日记")

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
                                    dropdown_place,
                                    dropdown_action,
                                    dropdown_fantancy, ],
                            outputs=[image_gallery_output, prompt_output])


# demo.description_style = background_style
demo.css = background_style
# select_act_button.css = background_style
# display(demo)
demo.queue()
demo.launch(share=True, server_name="0.0.0.0", server_port=7878)
