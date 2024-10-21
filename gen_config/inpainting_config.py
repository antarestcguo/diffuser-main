positive_promot_str = " 8k, high quality, hdr"

negative_prompt_str = "obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands, extra fingers, disconnected limbs, mutation, mutated, ugly, disgusting, low quality, long neck, frame, text, cut, human, person, man, girl, woman, lady, gentalman, Multiple buildings"

unique_identifier = "unique_identifier"
# format:
# key: save name str, value: ["img_name","mask_name","prompt"]
# need check using os.path.existing(file_name) in the code
demo_dict_list = {
    # "dog": ["tmp_dog.png", "tmp_dog.png", "a unique_identifier sitting on a park bench,"],
    "dog1": ["tmp_dog.png", "tmp_dog.png", "a unique_identifier,"],
    "dog2": ["tmp_dog.png", "tmp_dog.png",
             "A unique_identifier in the beach, sea, blue sky,"],
}

place_dict_list = {
    "beach": ["beach.jpg", "beach.jpg", "a unique_identifier lying on the beach"],
    "GreatWall": ["Great_Wall.jpg", "Great_Wall.jpg", "A unique_identifier is standing on the Great Wall,"],
    "GuiLin": ["guilin.jpeg", "guilin.jpg", "A unique_identifier,realistic style,"],
    "hongyadong": ["hongyadong.jpeg", "hongyadong.jpg", "A unique_identifier,realistic style,night sence,"],
    "mogaoku": ["mogaoku.jpeg", "mogaoku.jpg", "A unique_identifier,realistic style,"],
    "mountain": ["mountain.jpg", "mountain.jpg", "A unique_identifier on snow mountain,realistic style,"],
    "TimesSquare": ["Times_Square.jpg", "Times_Square.jpg", "A unique_identifier on Times Square"],
    "dinnerroom": ["dinnerroom.png", "dinnerroom.jpg", "A unique_identifier stand in the dinner room"],
}

action_dict_list = {
    "bike": ["bike.jpg", "bike.jpg", "A unique_identifier is riding a bike on the street, view to unique_identifier, "],
    "diving": ["diving.jpg", "diving.jpg",
               "A unique_identifier is underwater diving in the sea, Great Barrier Reef in Australia,realistic style,"],
    "diving2": ["diving2.jpg", "diving2.jpg",
                "A unique_identifier is underwater diving in the sea, Great Barrier Reef in Australia,realistic style,"],
    "skateboarding": ["skateboarding.jpg", "skateboarding.jpg", "A unique_identifier on skateboarding,"],
    "skiing": ["skiing.jpg", "skiing.jpg", "A unique_identifier is skiing,"],
    "skydiving": ["skydiving.jpeg", "skydiving.jpg", "A unique_identifier is skydiving,"],
    "swing": ["swing.jpeg", "swing.jpg", "A unique_identifier on swing,"],
}

status_dict_list = {
    "flowers": ["flowers.jpg", "flowers.jpg", "A unique_identifier is in the garden among the flowers"],
    "dollar": ["dollar.jpg", "dollar.jpg",
               "A unique_identifier is lying in a pile of dollar,"],
    "lion": ["lion.jpeg", "lion.jpg", "A unique_identifier"],
    "lion2": ["lion2.jpeg", "lion2.jpg", "A unique_identifier"],
    "moon": ["moon.jpg", "moon.jpg", "A unique_identifier on the moon, technology style, realistic style,"],
    "beer": ["beer.jpg", "beer.jpg", "A unique_identifier is landing,"],
    "boat": ["boat.jpg", "boat.jpg", "A unique_identifier on the boat,"],
    "crood": ["crood.jpeg", "crood.jpg", "A unique_identifier stand on the wood in the Croods,"],
    "dragon1": ["dragon.jpeg", "dragon.jpg", "A unique_identifier ride the toothless black dragon,"],
    "dragon2": ["dragon2.jpg", "dragon2.jpg", "A boy with a unique_identifier,"],
    "dragon3": ["dragon3.jpg", "dragon3.jpg", "A unique_identifier stand in the middle of the picture,"],
    "dragon4": ["dragon4.jpeg", "dragon4.jpg", "A unique_identifier stand beside the toothless black dragon,"],
    "dragon5": ["dragon5.jpeg", "dragon5.jpg", "A unique_identifier ride the toothless black dragon,"],
    "fall": ["fall.jpg", "fall.jpg", "A unique_identifier is landing,"],
    "human": ["human.jpeg", "human.jpg", "A unique_identifier stand on a human's hand,"],
    "madajiasijia": ["madajiasijia.jpeg", "madajiasijia.jpg", "A unique_identifier sit in a box on the sea,"],
    "madajiasijia2": ["madajiasijia2.jpeg", "madajiasijia2.jpg", "A unique_identifier stand beside animals,"],
    "minions": ["minions.jpeg", "minions.jpg", "A unique_identifier stand beside a guitar,"],
    "minions2": ["minions2.jpeg", "minions2.jpg", "A unique_identifier,"],
    "minions3": ["minions3.jpeg", "minions3.jpg", "A unique_identifier sit on the step,"],
    "minions4": ["minions4.jpg", "minions4.jpg", "A unique_identifier,"],
    "minions5": ["minions5.jpeg", "minions5.jpg", "A unique_identifier,"],
    "minions6": ["minions6.jpeg", "minions6.jpg", "A unique_identifier sit on the step,"],
    "mooncake": ["mooncake.jpg", "mooncake.jpg", "A unique_identifier stand beside the mooncake,"],
    "narnia": ["narnia.jpeg", "narnia.jpg", "A unique_identifier stand on the stone,"],
    "narnia2": ["narnia2.jpg", "narnia2.jpg", "A unique_identifier stand on the stone,"],
    "panda": ["panda.joeg", "panda.jpg", "A unique_identifier,"],
    "panda2": ["panda2.jpg", "panda2.jpg", "A unique_identifier stand beside the KungFu Panda,"],
    "springfestival": ["springfestival.jpeg", "springfestival.jpg", "A unique_identifier in red lanterns,"],
    "Thor": ["Thor.png", "Thor.jpg", "A unique_identifier stand on the stone,"],
    "yangjian": ["yangjian.jpg", "yangjian.jpg", "A unique_identifier in the middle of the picture,"],
    "zongzi": ["zongzi.jpg", "zongzi.jpg", "A unique_identifier,"],
    "zootopia": ["zootopia.jpeg", "zootopia.jpg", "A unique_identifier among the animal in ZooTopia,"],
    "zootopia2": ["zootopia2.jpg", "zootopia2.jpg", "A unique_identifier stand beside the Nike fox"],

    "Avatar": ["Avatar.jpeg", "Avatar.jpg", "A unique_identifier,"],
    "robot": ["robot.jpeg", "robot.jpg", "A unique_identifier is communicating with a human,"],
    "robot2": ["robot2.jpeg", "robot2.jpg", "A boy fondle the unique_identifier's head,"],
    "robot3": ["robot3.jpeg", "robot3.jpg", "A unique_identifier stand on the desk beside a boy,"],
    "robot4": ["robot4.jpeg", "robot4.jpg", "A unique_identifier in the middle of the image,"],
    "robot5": ["robot5.jpeg", "robot5.jpg", "a human with arm around the unique_identifier,"],
    "robot6": ["robot6.jpeg", "robot6.jpg", "A unique_identifier in the middle of the image,"],
    "robot8": ["robot8.png", "robot8.jpg", "A unique_identifier in the middle of the image,"],

}
