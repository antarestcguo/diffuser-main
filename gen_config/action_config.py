positive_promot_str = " 8k, high quality, hdr"

negative_prompt_str = "obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands, extra fingers, disconnected limbs, mutation, mutated, ugly, disgusting, low quality, long neck, frame, text, cut, human, person, man, girl, woman, lady, gentalman, Multiple buildings"
building_negative_prompt_str = "Mosaic figure, incomplete figure, misshapen figure,fragmentary image"

test_dict_list = {
    "tiger": "a super cute unique_identifier sitting on an oversized tiger with a pair of wings all the time, flying in the sky, flash, large, scenes, full body, sky city, pastel color, C4D octane rendering, ray tracing, bright light, perspective, depth of field, more intricate details, 8k",
    "swim": "a unique_identifier swims underwater, happy, fantasy, in a realistic hyper detailed render style, glow, yellow, blue,zbrush, hyper-realistic oil, head close-up, exaggerated perspective, Tyndall effect, water drops, mother-of-pearl iridescence, Holographic white, green background, realistic",
    "boating": "this is a sharply focused photo: on a cloudy day, on the lake, a group of unique_identifier in red vests are huddled on a small boat, the unique_identifier in the front opened their mouths, showing their tongues and teeth, with healthy and clean hair. This scene was shot on Canon XDIII, using the classic pet shot composition.",
    "rollercoaster": "a unique_identifier siting on the roller coaster",
    "kenya": "a unique_identifier walking the streets of Kenya, futuristic movies.",
    "wings": "A unique_identifier with a pair of wings all the time, flying in the sky, realistic style,",
}
action_dict_list = {
    # "diving": "A unique_identifier is underwater diving in the sea, Great Barrier Reef in Australia,realistic style,",
    # "bike": "A unique_identifier is riding a bike on the street, view to unique_identifier, ",
    # "skiing": "A unique_identifier Wearing a very big sled, skiing from high with a burst of speed, leaving a trail of flying snowflakes in its wake, ",
    # "skateboarding": "A unique_identifier is playing skate boarding on the street, view to the unique_identifier, ",
    # "football": "A unique_identifier is shootting the goal, the football leaves its foot, soaring towards the goal, realistic,sports photography, side view to unique_identifier,",
    # "keeper": "A unique_identifier  is a goalkeeper, dives towards the top left corner, try to touch the football, saving a football, realistic,sports photography , front view to the football gate,",
    # "skydiving": "A unique_identifier dons its skydiving gear and leaps from the plane.wearing a parachute, parachute is large, blue sky, fisheye,selfiemirror,",
    # "parachute": "A unique_identifier parachuting with a parachute in the air, blue sky, 3D cartoon movie style, full view,",
    # "rain": "A unique_identifier holds an black umbrella with its hand,  rainy, on a street, it is peace and silent, retro art style, cinematic lighting, view to the unique_identifier, ",
    # "eatting": "A unique_identifier in front of the table, eating a delicious steak. Close-up scene to unique_identifier,",
    # "shopping": "A unique_identifier is walking on the street with a shopping bag in his hand, realistic sytle, cinematic angle, view to the unique_identifier,",
    # "drinking": "A unique_identifier sit in chair, drinking coffee. side view to unique_identifier, in the coffee shop, at dusk, aestheticism, ",
    # "basketball": "A unique_identifier plays basketball, shoot the basket, jump highly,  slamdunk, basketball hoop,3D cartoon movie style,view to unique_identifier",
    "tennis": "A unique_identifier is holding a tennis racket on his left hand and playing tennis on Wimbledon tennis tournament, feel the Olympic spirit, A 18mm Wide-angle perspective photo , hyper realistic, summer vibe, sun shine, sunny and bright, 32k, super details, photorealistic, ",
    # "badminton": "A unique_identifier play badminton, with a badminton racket in his hand, view to the unique_identifier,",
    # "swimming": "A unique_identifier is swimming in the river,  head outside the water, body inside the water, sunshine, happy,summer,cool, side view to unique_identifier, ",
    # "cooking": "A unique_identifier is a chef, cooking in the kitchen, with a pot in hand, wearing chef's uniform, food is delicious, Close-up scene to unique_identifier",
    # "climbing": "A unique_identifier is climbing on a cliff,",
    # "dancing": "A unique_identifier is dancing at the ball, view to the unique_identifier,",
    # "driving": "A unique_identifier is a driving in the car,  front view Close-up scene to car,",
    # "broom": "A unique_identifier is flying on a magic broom at Hogwarts,  overcast sky, bottom view to unique_identifier, Magic style, dark style,",
    # "boating": "A unique_identifier is boating with paddle, on the river, side view to unique_identifier, spring, peace, ",
    # "fishing": "A unique_identifier sit, fishing by the river, hold fishing rod, wearing straw hat, side view to unique_identifier, quiet, ",
    # "guitar": "A unique_identifier is playing guitar, front view to unique_identifier,Full-length portrait,",
    # "TV": "A unique_identifier  is watching Television,  hold the remote control, side view to the unique_identifier, wide view, afternoon, warmly, ",
    # "violin": "A unique_identifier is playing violin, front view to unique_identifier,Full-length portrait,",
    # "typing": "A unique_identifier in office in front of the computer, the office has a computer desk and computer, film photos, fashion sense, rich details, technique style, ",
    # "unicorn": "A unique_identifier ride on unicorn, side view to the unique_identifier, grass, Fairy tale style,",
    # "dog": "A unique_identifier is walking dog in the garden, view to dog",
    # "duanwuBoating": "a group of lovely unique_identifier rowing with oars, happy and smile, vivid, orderly movements, on the vast expanse of water, intense and exciting , feel the Olympic spirit, A 35mm photo , hyper realistic, summer vibe, sun shine, sunny and bright, 32k, super details, photorealistic,",
    # "Swing": "a Swing on the tree, A unique_identifier siting on the Big wood swing seat, Swinging on a Swing,  the swing seat is Long and wide,",
    # "surfing": "A unique_identifier gracefully stands on the surfboard while surfing, soaring over the crest of the wave, a massive wave approaches,Splashes of water, blue sea",
}

wearing_dict_list = {
    "Christmas": "A unique_identifier is a welcome doll in the Christmas Day,Full-length portrait,front view to unique_identifier,",
    "santa": "A unique_identifier is a welcome doll,  wearing in Santa hat ,Full-body portrait,front view to unique_identifier,",
    "chef": "A unique_identifier is a welcome doll, wearing chef hat in the dinner room, Full-length portrait,front view to unique_identifier, wide view,",
    "boxing": "A unique_identifier is wearing boxing gloves in front of boxing bag, Full-length portrait, side view to unique_identifier, wide view, ",
    "Hallowmas": "A unique_identifier is wearing in wizard hat , on Hallowmas style,  with pumpkin monster, ghost style, dark style, ",
    "SpringFestival": "A unique_identifier is wearing in red , on Chinese Spring Festival, firecracker,lantern,",
    "Thanksgiving": "A unique_identifier, on Thanksgiving Day, turkey dinner, view to the unique_identifier",
    "DragonBoat": "A unique_identifier is a welcome doll, hold a green zongzi on hand, on Dragon Boat Festival, in the lobby,",
    "MidAutumn": "A unique_identifier is a welcome doll, hold mooncake, on Mid-Autumn Festival, in the lobby,",

}

group_dict_list = {
    "IronMan": "A unique_identifier and a Iron Man standing together,A full-body photo",
    "Spider": "A unique_identifier and a Spider-Man standing together ,A full-body photo",
    "panda": "The unique_identifier stands amidst a group of pandas, capturing a joyful group photo. pandas' soft fur, The pandas playfully gather around the unique_identifier, their adorable faces filled with curiosity and excitement. Sunlight filters through the bamboo forest, casting a warm glow on the scene. realistic style",
    "snowman": "a unique_identifier and a Snowman Olaf standing together,A full-body photo",
    "car": "a unique_identifier and a black car standing together,A full-body photo",
    "Musk": "a unique_identifier and Elon Musk, standing together,A full-body photo",
    "Messi": "The unique_identifier and Messi pose for a photo on the soccer field.unique_identifier's metallic arms wrap around Messi, showcasing their friendly bond.Messi stands beside the unique_identifier, creating a warm ambiance",
}

status_dict_list = {
    "moon": "A unique_identifier is navigating on the moon in space station, technology style, realistic style",
    "dollar": "A unique_identifier is lying in a pile of dollar, dollar Filling the sky, background is dollar",
    "Castle": "A unique_identifier in front of the Castle",
    "Disneyland": "A unique_identifier in front of the Disneyland Resort",
    "Mars": "A unique_identifier is navigating on the Mars in space station, technology style, realistic style",
    "fly": "A unique_identifier is flying in the sky",
    "road": "A unique_identifier is standing on the road",
}

place_dict_list = {
    "TempleofHeaven": "A unique_identifier in front of the Temple of Heaven, A full-body photo,",
    "PotalaPalace": "A unique_identifier in front of the Potala Palace in Tibet, ,A full-body photo,, distant perspective,",
    "Xitang": "A unique_identifier in  Xitang Ancient Town,ancient customs,",
    "Zhangye": "A unique_identifier in  Zhangye Danxia Landform,,A full-body photo,",
    "MoGaoCave": "A unique_identifier in front of the Mogao Caves,A full-body photo in a distant perspective ,",
    "QingHaiLake": "A unique_identifier in the bank of the Qinghai Lake,A full-body photo in a distant perspective,",
    "BigBen": "A unique_identifier in front of the Big Ben,A full-body photo,",
    "LondonEye": "A unique_identifier in front of the London Eye, A full-body photo,",
    "TimesSquare": "A unique_identifier is wearing sunglasses in Times Square,",
    "GreatWall": "A unique_identifier is standing on the Great Wall,",
    "OrientaPearlTower": "A unique_identifier in front of the Oriental Pearl Tower, A full-body photo in a distant perspective ,",
    # 通常合照在黄浦江
    "HuangpuRiver": "A unique_identifier in front of the Huangpu River Bank in Shanghai,",
    # "HuangpuRivernight": "A unique_identifier in front of the Huangpu River Bank in Shanghai, night scene,",# night靠IoT加
    "TheBund": "A unique_identifier is walking on the Bund street, ancient building in the side, movie style,",
    "TheForbiddenCity": "A unique_identifier in the Forbidden City in China,",
    # "TianAnMen": "A unique_identifier in Tiananmen Square, A full-body photo in a medium focal length perspective,",
    "EiffelTower": "A unique_identifier in front of the Eiffel Tower, A full-body photo,",
    # "EiffelTowernight": "A unique_identifier in front of the Eiffel Tower,A full-body photo,night scene,", # night靠IoT加
    "AthenAcropolis": "A unique_identifier in front of the Athen Acropolis,",
    # "BirdNestStadium": "A unique_identifier in front of the  Beijing National Stadium, ",
    # "WaterCube": "A unique_identifier in front of the Water Cube, in Beijing 2008 Olympic, ",
    # "BirdNestStadiumnight": "A unique_identifier in front of the Beijing National Stadium, night scene,", # 都是内部体育场结构，没有鸟巢特点
    # "WaterCubenight": "A unique_identifier in front of the Water Cube, in Beijing 2008 Olympic, night scene,",
    # from GPT
    "EgyptianPyramids": "A unique_identifier in the Egyptian Pyramids in Egypt,",
    "StatueofLiberty": "A unique_identifier in front of Statue of Liberty in United States, A full-body photo in a distant perspective,",
    "Sphinx": "A unique_identifier in front of the Sphinx in Egypt, A full-body photo in a distant perspective,",
    "Colosseum": "A unique_identifier in front of the Colosseum in Italy,",
    "TajMahal": "A unique_identifier in front of the Taj Mahal in India,",
    "SydneyOperaHouse": "A unique_identifier in front of Sydney Opera House in Australia,A full-body photo in a medium focal length perspective,",
    "MountFuji": "A unique_identifier in front of the Mount Fuji in Japan,A full-body photo in a medium focal length perspective, ",
    "GrandCanyonNationalPark": "A unique_identifier in Grand Canyon National Park,A full-body photo in a medium focal length perspective,",
    # "Kremlin": "A unique_identifier in the Kremlin in Russia, A full-body photo,", # 特征不明显
    "LuxorTemple": "A unique_identifier in the Luxor Temple in Egypt, A full-body photo, ",
    "WhiteHouse": "A unique_identifier in front of the White House in United States, A full-body photo, ",
    "IndiaGate": "A unique_identifier in front of the India Gate in India , A full-body photo ,",
    # "ChristtheRedeemer": "A unique_identifier in front of the  Christ the Redeemer in Brazil, A full-body photo, in a distant perspective,Selfie perspective,", # 场景太大，已经不能合照了
    "LeaningTowerofPisa": "A unique_identifier in front of  Leaning Tower of Pisa in Italy, A full-body photo,",
    "ArcdeTriomphe": "A unique_identifier in front of the Arc de Triomphe in France, A full-body photo,",
    # "MaasaiMara": "A unique_identifier in Maasai Mara National Reserve in Kenya, A full-body photo,", # 放到sence
    # "BelémTower": "A unique_identifier in front of the Belém Tower in Portugal, A full-body photo,", # 特征不明显
    "GrandPalace": "A unique_identifier in front of the Grand Palace in Thailand, A full-body photo,",
    "StPaulCathedral": "A unique_identifier in front of the St. Paul's Cathedral,A full-body photo,",
    # "ValleyoftheKings": "A unique_identifier in front of the Valley of the Kings in Egypt, A full-body photo,",# 特征不明显
    "JiuzhaigouValley": "A unique_identifier beside the Jiuzhaigou Valley in China, A full-body photo,",
    "ErhaiLake": "A unique_identifier beside the bank of the Erhai Lake in Dali, A full-body photo,",
    "JadeDragonSnowMountain": "A unique_identifier claim the Jade Dragon Snow Mountain in China, A full-body photo,",
    "Guilin": "A unique_identifier on the bamboo raft of the Li River in Guilin, A full-body photo,",
    "Zhangjiajie": "A unique_identifier in Zhangjiajie National Forest Park, A full-body photo,",
    "CornerTower": "A unique_identifier in the Corner Tower of the Forbidden City,",
}

scene_reference = {
    "beach": ["beach", "Icarai Beach", "Paradise Island", "South Island Beaches", "Lady Elliot Island",
              "Whitehaven Beach"],
    "garden": ["garden", "Sissinghurst Castle Garden", "Naroa-Narbinem Gardens"],
    "desert": ["desert", "Sahara Desert", "Arabian Desert", ],
    "lake": ["lake", "Lake Malawi", "Swan Lake", "Lake Wanaka", "Lake Garda", "Lake Como", "Lake Niagara"],
    "grassland": ["grassland", "Great Plains", "Pampas", "Alpine Meadows", "Greater Khingan Prairie",
                  "Campos de Cima da Serra", "Serengeti", "MaasaiMara"],
    "forest": ["forest", "Amazon Rainforest", "Congo Basin Rainforest", "Indonesian Borneo Rainforest",
               "Redwood National Park, USA", "Malaysian Tropical Rainforest", "Senegal Forest", "Madagascar Rainforest",
               "Rennes River Basin Forest", "Ivan Susanin Forest", "Himalayan Forest", ]
}

scene_dict_list = {
    "beach": [f"A unique_identifier is lying and basking on the {C}, realistic style, " for C in
              scene_reference["beach"]],
    "garden": [f"A unique_identifier is in the {C} among the flowers," for C in scene_reference["garden"]],
    "desert": [f"A unique_identifier in the {C}, sunset, A full-body photo in a distant perspective, " for C in
               scene_reference["desert"]],
    "lake": [f"A unique_identifier in front of the {C}, A full-body photo , " for C in scene_reference["lake"]],
    "grassland": [f"A unique_identifier on the {C}, A full-body photo in a distant perspective," for C in
                  scene_reference["grassland"]],
    "forest": [f"A unique_identifier on the {C}, A full-body photo," for C in scene_reference["forest"]],

}
