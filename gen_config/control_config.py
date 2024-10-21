positive_promot_str = " best quality"

negative_prompt_str = "worst quality, poorly drawn, extra limb, missing limb, floating limbs, mutated hands, extra fingers, disconnected limbs, mutation, mutated, ugly, disgusting, low quality,lowres, bad anatomy, bad hands, cropped, worst quality,"

unique_identifier = "unique_identifier"

# key:[img_name,pose_name,prompt]
action_dict_list = {
    # "basketball": ["basketball.png", 
    #                "A unique_identifier plays basketball,movie style,basketball on unique_identifier's hand,"],
    "basketball2": ["basketball2.jpeg", 
                    "A unique_identifier plays basketball,movie style,basketball on unique_identifier's hand,"],
    # "bike_front": ["bike_front.jpeg",  "A unique_identifier is riding a bike on the street,"],
    # "bike_side": ["bike_side.jpeg",  "A unique_identifier is riding a bike on the street,"],
    "tennis": ["child_tennis2.jpeg", 
               "A unique_identifier is holding a tennis racket on his left hand and playing tennis on Wimbledon tennis tournament, feel the Olympic spirit, A 18mm Wide-angle perspective photo , hyper realistic, summer vibe, sun shine, sunny and bright, 32k, super details, photorealistic,"],
    # "fencing": ["fencing.jpg", 
    #             "A unique_identifier is fencing, ready to embrace the challenge,gracefully swings the sword, performing fluid and precise fencing moves with speed and accuracy, showcasing the perfect fusion of technology and sports"],
    "keeper": ["keeper.jpeg",
               "A unique_identifier is a goalkeeper, dives towards the top left corner, try to touch the football, saving a football, realistic,sports photography , front view to the football gate,"],
    "keeper2": ["keeper2.jpg", 
                "A unique_identifier is a goalkeeper, try to touch the football, saving a football, realistic,sports photography , front view to the football gate,"],
    "keeper3": ["keeper3.jpeg", 
                "A unique_identifier is a goalkeeper, try to touch the football, saving a football, realistic,sports photography , front view to the football gate,"],
    # "keeper4": ["keeper4.jpeg", 
    #             "A unique_identifier is a goalkeeper, try to touch the football, saving a football, realistic,sports photography , front view to the football gate,"],
    # "keeper5": ["keeper5.jpeg", 
    #             "A unique_identifier is a goalkeeper, try to touch the football, saving a football, realistic,sports photography , front view to the football gate,"],
    "left_hand_up": ["left_hand_up.jpeg",  "A unique_identifier raises left hand up,"],
    # "lie": ["lie.jpeg",  "A unique_identifier lying on the sofa,"],
    "right_hand_up": ["right_hand_up.jpeg",  "A unique_identifier raises right hand up,"],
    "shoot": ["shoot.jpeg",
              "A unique_identifier is shootting the goal, the football leaves its foot, soaring towards the goal, realistic,sports photography, side view to unique_identifier,"],
    "shoot2": ["shoot2.jpg",
               "A unique_identifier is shootting the goal, the football leaves its foot, soaring towards the goal, realistic,sports photography, side view to unique_identifier,"],
    "sit": ["sit.jpeg",
            "A unique_identifier sit in chair, drinking coffee. side view to unique_identifier, in the coffee shop, at dusk, aestheticism,"],
    # "slamdunk": ["slamdunk.jpg", 
    #              "A unique_identifier plays basketball, shoot the basket, jump highly,  slamdunk, basketball hoop, movie style,view to unique_identifier,"],
    "slamdunk2": ["slamdunk2.png", 
                  "A unique_identifier plays basketball, shoot the basket, jump highly,  slamdunk, basketball hoop, movie style,view to unique_identifier,"],
    # "slamdunk3": ["slamdunk3.jpeg", 
    #               "A unique_identifier plays basketball, shoot the basket, jump highly,  slamdunk, basketball hoop, movie style,view to unique_identifier,"],
    # "slamdunk4": ["slamdunk4.jpeg", 
    #               "A unique_identifier plays basketball, shoot the basket, jump highly,  slamdunk, basketball hoop, movie style,view to unique_identifier,"],
    # "slamdunk5": ["slamdunk5.jpeg", 
    #               "A unique_identifier plays basketball, shoot the basket, jump highly,  slamdunk, basketball hoop, movie style,view to unique_identifier,"],
    "stand": ["stand.jpeg", 
              "A unique_identifier is wearing in wizard hat , on Hallowmas style,  with pumpkin monster, ghost style, dark style, "],
    "stand2": ["stand2.jpeg", 
               "A unique_identifier is wearing in Santa hat ,Full-body portrait,front view to unique_identifier,"],
    # "swim": ["swim.jpeg", 
    #          "a unique_identifier swims underwater, happy, fantasy, in a realistic hyper detailed render style, glow, yellow, blue,zbrush, hyper-realistic oil, head close-up, exaggerated perspective, Tyndall effect, water drops, mother-of-pearl iridescence, Holographic white, green background, realistic"],
    "tennis2": ["tennis.jpeg", 
                "A unique_identifier play tennis, with a tennis racket in his hand,sunshine, view to the unique_identifier,"],
    "volleyball": ["volleyball.jpg",
                   "A unique_identifier is jumping on the volleyball court, ready to spike the ball, As the volleyball rises, the unique_identifier swiftly adjusts its position and then forcefully swings its arms, launching a powerful spike. "],
    "welcome": ["welcome.jpeg", 
                "A unique_identifier is wearing chef hat in the dinner room, Full-length portrait,front view to unique_identifier, wide view,"],

}

pose_dict_list = {
    "pose1": ["pose1.jpeg",
                       "A unique_identifier plays basketball,movie style,basketball on unique_identifier's hand,"],
}