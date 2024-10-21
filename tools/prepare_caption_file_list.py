output_filename = "/home/Antares.Guo/data/caption_total_list.txt"

TE_token = "unique_identifier"
caption_dict = {
    "Image_20230613172521.png": f"a {TE_token} with yellow eyes and a smile, front view",
    "Image_20230613172615.png": f"a {TE_token} with yellow eyes and a smile, 30 degree angle view",
    "Image_20230613172707.png": f"two women are standing next to {TE_token}, the {TE_token} raise left hand",
    "Image_20230613172935.png": f"a {TE_token} wink with smile, with yellow eyes, the {TE_token} stands on the flag paving",
    "Image_20230613173004.png": f"a group of {TE_token} with yellow eyes and a smile, they are on display at an exhibition with red carpet",
    "Image_20230613173046.png": f"a group of {TE_token} with yellow eyes and a smile, are on the floor",
    "Image_20230613173130.png": f"a woman in a pink dress standing next to the {TE_token}, the {TE_token} is front view with yellow eyes",
    "Image_20230613173319.png": f"three {TE_token} are sitting in front of a large screen, the {TE_token} with yellow eyes are in 45 degree angle view",
    "Image_20230613173335.png": f"a {TE_token} with yellow eyes and a smile, 45 degree angle view",
    "Image_20230619141353.png": f"a girl is shaking hands with a {TE_token}, the {TE_token} with a smile and raise hands, 45 degree angle view",

    "WechatIMG10.jpeg": f"a {TE_token} sitting on a white floor,hands down,front view",
    "WechatIMG11.jpeg": f"a {TE_token} that is sitting on a white floor,hands down,front view",
    "WechatIMG12.jpeg": f"a {TE_token} that is sitting on a floor in a room, top to down view",
    "WechatIMG13.jpeg": f"a {TE_token} in a room with a couch and a chair,Medium focal Angle",
    "WechatIMG14.jpeg": f"a {TE_token} is sitting on the floor next to a couch,top to down view",
    "WechatIMG15.jpeg": f"a {TE_token} is in 60 degree angle view",
    "WechatIMG16.jpeg": f"a {TE_token} is in 45 degree angle view, hands down",
    "WechatIMG17.jpeg": f"a {TE_token} sitting on a white floor,hands down, top to down view",
    "WechatIMG18.jpeg": f"a close-up view {TE_token} sitting on a white floor,hands down",
    "WechatIMG1.jpeg": f"a {TE_token} sitting on a white floor,hands down,front view",
    "WechatIMG2.jpeg": f"a {TE_token} that is sitting on the floor in a room, 30 degree angle view",
    "WechatIMG3.jpeg": f"a {TE_token} stand on the floor, side view",
    "WechatIMG4.jpeg": f"a {TE_token} in a room with a couch",
    "WechatIMG5.jpeg": f"a {TE_token} stand with table, side view",
    "WechatIMG6.jpeg": f"a {TE_token} is sitting on a floor beside the green plants",
    "WechatIMG8.jpeg": f"a {TE_token} is sitting on a floor beside the green plants and a couch, 60 degree angle view",
    "WechatIMG9.jpeg": f"a {TE_token} stands with couch and table, side view"
}

with open(output_filename, 'w') as f:
    for k, v in caption_dict.items():
        f.write(k + '\t' + v + '\n')
