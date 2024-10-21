# unique_identifier = "sks orange round robot"
positive_promot_str = " 8k, high quality, hdr"

negative_prompt_str = "obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands, extra fingers, disconnected limbs, mutation, mutated, ugly, disgusting, low quality, long neck, frame, text, cut"
test_dict_list={}
base_prompt_dict_list = {
    "Christmas": "A unique_identifier is in Christmas,",
    "superman": "A unique_identifier is wearing a Superman suit,",
    "hat": "A unique_identifier is wearing a Christmas hat,",
    "skateboarding": "A unique_identifier is playing skateboard on the street,",
    "skiing": "A unique_identifier is skiing on the snow,",
    "dollar": "A unique_identifier is lying in a pile of dollar",
    "Times_Square": "A unique_identifier is wearing sunglasses in Times Square,",
    "Great_Wall": "A unique_identifier is standing on the Great Wall,",
    "mountain": "A unique_identifier is climbing on a cliff,",
    "bike": "A unique_identifier is riding a bike on the street,",
    "beach": "A unique_identifier is lying and basking on the beach, under the coconut tree,",
    "moon": "A unique_identifier is navigating on the moon in space station, technology style, ",
    "flowers": "A unique_identifier is dancing and smiling among the flowers,",
    "diving": "A unique_identifier is underwater diving in the sea surrounded by a lot of fish,",
    "clock": "A unique_identifier in front of the Big Ben,",
    "cloth": "A unique_identifier is wearing a red T-shirt"
}

# A and B take a group photo together
group_prompt_dict_list = {
    "girl": "a unique_identifier and a lovely girl take a group photo together, outside, sunshine,",
    "dog": "a unique_identifier and a dog wearing a sunglasses take a group photo together,",
    "snowman": "a unique_identifier and a snowman take a group photo together on the snow mountain,",
    "car": "a unique_identifier and a sports car take a group photo together in front of a building,",
    "unicorn": "a unique_identifier and a white unicorn take a group photo together in a castle",
}

enhance_prompt_dict_list = {
    "football": "A unique_identifier is shooting at a football field, wide shot view,realistic,sports photography,",
    "keeper": "A unique_identifier is a goalkeeper saving a football that is shot into the goal in front of the gate, realistic,sports photography,",
    "skydiving": "A unique_identifier is skydiving in the blue sky wearing a parachute, fisheye,selfiemirror,",
    "parachute": "A unique_identifier parachuting with a  parachute in the air, blue sky, 3D cartoon movie style, full view,",
    "rain": "A unique_identifier holds an black umbrella with its hand,  rainy, on a street, it is peace and silent, retro art style, cinematic lighting,",
    "restaurant": "A unique_identifier is eating steak and spaghetti in a restaurant, the food is tasty, cinematic angle, close-up,",
    "butterflies": "A unique_identifier is chasing butterflies in the garden, the flower is beautiful, chibi and cartoon movie style,",
    "Olympic": "A unique_identifier is standing on the podium of the Beijing Olympic Games, wide shot view, realistic sytle",
    "shopping": "A unique_identifier is walking on the street with a shopping bag in his hand, realistic sytle, cinematic angle, ",
    "coffee": "A unique_identifier is in coffee shop, drinking coffee, at dusk, aestheticism, ",
    "Santa": "A unique_identifier is wearing in a Santa costume in the dinner hall,realistic,cozy, happness, ",
    "Hallowmas": "A unique_identifier is wearing a witch hat, playing with Halloween pumpkin doll, cozy, happness,"
}

other_prompt_dict_list = {
    "kitten": "A white kitten sit on a table, lovely, cute, smile, big eyes,",
    "vampire": "vivid color Portrait of a pale female vampire, Red Hair, Her Face covered with Black Foliage and white rosses,",
    "dog": "a dog with sunglass,",
}

img_base_path = "/home/Antares.Guo/data/webimg"
img_base_edit_path = "/home/Antares.Guo/data/webimg_mask"
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
