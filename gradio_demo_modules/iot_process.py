import datetime
from random import choice
from gradio_demo_modules.prompt_config import Radio_options

month = [3, 6, 8, 12]

month_dict = {
    3: ["early spring", "Spring blooms"],
    4: ["spring in full bloom"],
    5: ["late spring"],
    6: ["early summer"],
    7: ["peak summer"],
    8: ["sweltering summer"],
    9: ["early autumn"],
    10: ["mid autumn"],
    11: ["deep autumn"],
    12: ["mid winter"],
    1: ["freezing winter"],
    2: ["chilly winter"]
}

feelings = {
    1: "happy new year atmosphere",
    11: "thinksgiving day atmosphere",
    12: "christmas atmosphere",
    "evening, night scene": "moon, dark",
    "dusk, sunset": "sunset glow",
    "morning, sunrise": "advective fog",
    "afternoon": "warm sun"
}

time_dict = {
    5: ["morning", "sunrise", "sunup", "advective fog"],
    6: ["morning", "sunrise", "sunup", "advective fog"],
    7: ["morning", "sunrise", "sunup", "advective fog"],
    8: ["morning", "sunrise", "sunup", "advective fog"],
    9: ["morning", "sunrise", "sunup", "advective fog"],
    10: ["morning", "sunrise", "sunup", "advective fog"],
    11: ["afternoon", ],  # "Scorching sun",  "Sunlight beaming"
    12: ["afternoon", ],
    13: ["afternoon", ],
    14: ["afternoon", ],
    15: ["afternoon", ],
    16: ["afternoon", ],
    17: ["dusk", "sunset,", "Sunset in the west", "Golden rays of the setting sun", "Ethereal glow", "Colorful clouds",
         "Mist lingering", "Gradual darkening", "Comforting atmosphere of dusk", "Tranquil silence"],
    18: ["dusk", "sunset,", "Sunset in the west", "Golden rays of the setting sun", "Ethereal glow", "Colorful clouds",
         "Mist lingering", "Gradual darkening", "Comforting atmosphere of dusk", "Tranquil silence"],
    19: ["dusk", "sunset,", "Sunset in the west", "Golden rays of the setting sun", "Ethereal glow", "Colorful clouds",
         "Mist lingering", "Gradual darkening", "Comforting atmosphere of dusk", "Tranquil silence"],
    20: ["evening", "night scene", "moon", "dark", "Twinkling stars", "Moonlight silver", "Late-night tranquility",
         "Nightfall", "Quiet darkness", "Starry sky", "Nighttime serenity", "Glistening nocturnal scenery",
         "Nighttime mystery"],
    21: ["evening", "night scene", "moon", "dark", "Twinkling stars", "Moonlight silver", "Late-night tranquility",
         "Nightfall", "Quiet darkness", "Starry sky", "Nighttime serenity", "Glistening nocturnal scenery",
         "Nighttime mystery"],
    22: ["evening", "night scene", "moon", "dark", "Twinkling stars", "Moonlight silver", "Late-night tranquility",
         "Nightfall", "Quiet darkness", "Starry sky", "Nighttime serenity", "Glistening nocturnal scenery",
         "Nighttime mystery"],
    23: ["evening", "night scene", "moon", "dark", "Twinkling stars", "Moonlight silver", "Late-night tranquility",
         "Nightfall", "Quiet darkness", "Starry sky", "Nighttime serenity", "Glistening nocturnal scenery",
         "Nighttime mystery"],
    24: ["evening", "night scene", "moon", "dark", "Twinkling stars", "Moonlight silver", "Late-night tranquility",
         "Nightfall", "Quiet darkness", "Starry sky", "Nighttime serenity", "Glistening nocturnal scenery",
         "Nighttime mystery"],
    1: ["evening", "night scene", "moon", "dark", "Twinkling stars", "Moonlight silver", "Late-night tranquility",
        "Nightfall", "Quiet darkness", "Starry sky", "Nighttime serenity", "Glistening nocturnal scenery",
        "Nighttime mystery"],
    2: ["evening", "night scene", "moon", "dark", "Twinkling stars", "Moonlight silver", "Late-night tranquility",
        "Nightfall", "Quiet darkness", "Starry sky", "Nighttime serenity", "Glistening nocturnal scenery",
        "Nighttime mystery"],
    3: ["evening", "night scene", "moon", "dark", "Twinkling stars", "Moonlight silver", "Late-night tranquility",
        "Nightfall", "Quiet darkness", "Starry sky", "Nighttime serenity", "Glistening nocturnal scenery",
        "Nighttime mystery"],
    4: ["evening", "night scene", "moon", "dark", "Twinkling stars", "Moonlight silver", "Late-night tranquility",
        "Nightfall", "Quiet darkness", "Starry sky", "Nighttime serenity", "Glistening nocturnal scenery",
        "Nighttime mystery"],
}

# weather = [
#     "自然风光", "阳光明媚", "微风拂面", "浓雾弥漫", "大雪纷飞", "烟雨蒙蒙","电闪雷鸣"
# ]

weather_dict = {
    "自然风光": ["hdr"],
    "阳光明媚": ["Blue sky and white clouds", "Bright sunshine", "Clear blue sky", "sunny weather", "Cloudless sky",
                 "Radiant sunlight", "Warm and pleasant", "Endless clear sky", "A clear day"],
    "微风拂面": ["Soft rustling of the wind", "Cool and pleasant", "Gentle breeze caressing",
                 "Gentle breeze blowing slowly", "Wind whispering by the ears", "Gentle breeze rippling",
                 "Gentle and light breeze", "Breeze sweeping across the land"],
    "浓雾弥漫": ["Thick fog enveloping", "Dense fog swirling", "Hazy mist", "Vast expanse of fog", "Fog condensing",
                 "Fog shrouding"],
    "大雪纷飞": ["Snowflakes swirling", "Heavy snowfall", "A vast expanse of white snow", "Snowflakes dancing",
                 "Snowflakes fluttering", "Big snowflakes falling", "Pure white snow", ],
    "烟雨蒙蒙": ["Drizzling rain", "Fine drizzle", "Light misty rain", "Pouring rain", "Curtain of rain swaying", ],
    # "电闪雷鸣": ["Thunder rumbling",
    #              "Flashing lightning and rumbling thunder", "Lightning streaking across the sky",
    #              "Thunderclaps and lightning bolts"]
}


def gen_iot_prompt(time, month, weather, Radio, iot_strength=3):
    # "环游世界", "极限运动", "探索未来"
    final_prompt = ""
    if Radio == Radio_options[0]:
        if weather != "自然风光":
            final_prompt = ",".join([choice(weather_dict[weather]) for i in range(iot_strength)])
        else:
            time_prompt = ",".join([choice(time_dict[time]) for i in range(iot_strength)])
            month_prompt = ",".join([choice(month_dict[month]) for i in range(iot_strength)])

            final_prompt = ",".join([time_prompt, month_prompt])

    return final_prompt


def time_iot(m=None, h=None):
    now = datetime.datetime.now()
    month, hour = now.month, now.hour
    if m is not None:
        month = m
    if h is not None:
        hour = h

    return [month, hour]
