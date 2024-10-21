unique_identifier = "sks orange round robot"
positive_promot_str = " 8k, high quality, hdr"

negative_prompt_str = "obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands, extra fingers, disconnected limbs, mutation, mutated, ugly, disgusting, low quality, long neck, frame, text, cut"

base_prompt_dict_list = {
    # "Christmas": "A %s is in Christmas" % unique_identifier,
    "superman": "A %s is wearing a Superman costume" % unique_identifier,
    "hat": "A %s is wearing a Christmas hat" % unique_identifier,
    # "skateboarding": "A %s is playing skateboard on the street," % unique_identifier,
    # "skiing": "A %s is skiing on the snow," % unique_identifier,
    # "dollar": "A %s is lying in a pile of dollar" % unique_identifier,
    # "Times_Square": "A %s is wearing sunglasses in Times Square," % unique_identifier,
    # "Great_Wall": "A %s is standing on the Great Wall," % unique_identifier,
    # "mountain": "A %s is climbing mountain," % unique_identifier,
    # "bike": "A %s is riding a bike on the street," % unique_identifier,
    # "beach": "A %s is lying and basking on the beach, under the coconut tree," % unique_identifier,
    # "moon": "A %s is navigating on the moon in space station, technology style, " % unique_identifier,
    # "flowers": "A %s is dancing and smiling among the flowers," % unique_identifier,
    # "diving": "A %s is underwater diving in the sea surrounded by a lot of fish" % unique_identifier,
    # "clock": "A %s in front of the Big Ben," % unique_identifier,

}

# A and B take a group photo together
group_prompt_dict_list = {
    "girl": "%s and a lovely girl take a group photo together, outside, sunshine," % unique_identifier,
    "dog": "%s and a dog wearing a sunglasses take a group photo together," % unique_identifier,
    "snowman": "%s and a snowman take a group photo together on the snow mountain," % unique_identifier,
    "car": "%s and sports car take a group photo together in front of a building," % unique_identifier,
    "unicorn": "%s and a white unicorn take a group photo together in a castle" % unique_identifier,
}

enhance_prompt_dict_list = {
    "football": "A %s is shotting on the soccer field, wide shot view,realistic,sports photography," % unique_identifier,
    "keeper": "A %s is a goalkeeper saving a soccer ball in front of goal gate, realistic,sports photography," % unique_identifier,
    "skydiving": "A %s is skydiving in the blue sky wearing a parachute, fisheye,selfiemirror," % unique_identifier,
    "parachute": "A %s is wearing a parachute in the blue sky, 3D cartoon movie style," % unique_identifier,
    "rain": "A %s holds an balck umbrella in the rain, the background is on a street, it is peace and silent, retro artstyle, cinematiclighting, " % unique_identifier,
    "restaurant": "A %s is eating steak and spaghetti in a restaurant, the food is tasty, cinematic angle, close-up," % unique_identifier,
    "butterflies": "A %s is chasing butterflies in the garden, the flower is beautiful, chibi and cartoon movie style," % unique_identifier,
    "Olympic": "A %s is standing on the podium of the Beijing Olympic Games, wide shot view, realistic sytle" % unique_identifier,
    "shopping": "A %s is walking on the street with a shopping bag in his hand, realistic sytle, cinematic angle, " % unique_identifier,
    "coffee": "A %s is in coffee shop, drinking coffee, at dusk, lens 135mm,f1.8, " % unique_identifier,
    "Santa": "A %s is wearing in a Santa costume in the dinner hall,realistic,warmly, happness" % unique_identifier,
}

other_prompt_dict_list = {
    "kitten": "A white kitten sit on a table, lovely, cute, smile, big eyes,",
    "vampire": "vivid color Portrait of a pale female vampire, Red Hair, Her Face covered with Black Foliage and white rosses,",
    "dog": "a dog with sunglass,",
}

img_base_path = "/home/Antares.Guo/data/webimg"
img_base_dict = {
    "beach": "beach.jpg",
    "bike": "bike.jpg",
    "clock": "clock.jpeg",
    "diving": ["diving.jpg", "diving2.jpg"],
    "dollar": "dollar.jpg",
    "flowers": "flowers.jpg",
    "moon": "moon.jpg",
    "mountain": "mountain.jpg",
    "skateboarding": "skateboarding.jpg",
    "skiing": "skiing.jpg",
    "skydiving": "skydiving.jpeg",
    "Times_Square": "Times_Square.jpg"
}

TE_img_dict = {"TE": "/home/Antares.Guo/data/TE/WechatIMG10.jpeg"}
