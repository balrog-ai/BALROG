Use MAX_STATIC_DATA of 500000.
When play begins, seed the random-number generator with 1234.

container is a kind of thing.
door is a kind of thing.
object-like is a kind of thing.
supporter is a kind of thing.
food is a kind of object-like.
key is a kind of object-like.
containers are openable, lockable and fixed in place. containers are usually closed.
door is openable and lockable.
object-like is portable.
supporters are fixed in place.
food is edible.
A room has a text called internal name.


The r_0 and the r_1 and the r_10 and the r_11 and the r_14 and the r_12 and the r_13 and the r_16 and the r_15 and the r_18 and the r_19 and the r_2 and the r_3 and the r_4 and the r_5 and the r_7 and the r_6 and the r_8 and the r_9 and the r_17 are rooms.

Understand "bedroom" as r_0.
The internal name of r_0 is "bedroom".
The printed name of r_0 is "-= Bedroom =-".
The bedroom part 0 is some text that varies. The bedroom part 0 is "This might come as a shock to you, but you've just stumbled into a bedroom.

 You can make out a bed. [if there is something on the s_0]You see [a list of things on the s_0] on the bed. Something scurries by right in the corner of your eye. Probably nothing.[end if]".
The bedroom part 1 is some text that varies. The bedroom part 1 is "[if there is nothing on the s_0]The bed appears to be empty.[end if]".
The bedroom part 2 is some text that varies. The bedroom part 2 is "

 There is [if d_3 is open]an open[otherwise]a closed[end if]".
The bedroom part 3 is some text that varies. The bedroom part 3 is " door leading east. There is an exit to the west. Don't worry, it is unguarded.".
The description of r_0 is "[bedroom part 0][bedroom part 1][bedroom part 2][bedroom part 3]".

The r_1 is mapped west of r_0.
east of r_0 and west of r_2 is a door called d_3.
Understand "steam room" as r_1.
The internal name of r_1 is "steam room".
The printed name of r_1 is "-= Steam Room =-".
The steam room part 0 is some text that varies. The steam room part 0 is "You've just sauntered into a steam room. You can barely contain your excitement.

 You see [if c_0 is locked]a locked[else if c_0 is open]an opened[otherwise]a closed[end if]".
The steam room part 1 is some text that varies. The steam room part 1 is " trunk close by.[if c_0 is open and there is something in the c_0] The trunk contains [a list of things in the c_0]. You shudder, but continue examining the room.[end if]".
The steam room part 2 is some text that varies. The steam room part 2 is "[if c_0 is open and the c_0 contains nothing] What a letdown! The trunk is empty![end if]".
The steam room part 3 is some text that varies. The steam room part 3 is "

There is an exit to the east. Don't worry, it is unguarded.".
The description of r_1 is "[steam room part 0][steam room part 1][steam room part 2][steam room part 3]".

The r_0 is mapped east of r_1.
Understand "restroom" as r_10.
The internal name of r_10 is "restroom".
The printed name of r_10 is "-= Restroom =-".
The restroom part 0 is some text that varies. The restroom part 0 is "You've just sauntered into a restroom. I guess you better just go and list everything you see here.

 You make out [if c_1 is locked]a locked[else if c_1 is open]an opened[otherwise]a closed[end if]".
The restroom part 1 is some text that varies. The restroom part 1 is " case, which looks standard, in the corner.[if c_1 is open and there is something in the c_1] The case contains [a list of things in the c_1].[end if]".
The restroom part 2 is some text that varies. The restroom part 2 is "[if c_1 is open and the c_1 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The restroom part 3 is some text that varies. The restroom part 3 is " You see a board. [if there is something on the s_1]You see [a list of things on the s_1] on the board. You idly wonder how they came up with the name TextWorld for this place. It's pretty fitting.[end if]".
The restroom part 4 is some text that varies. The restroom part 4 is "[if there is nothing on the s_1]But oh no! there's nothing on this piece of garbage. Oh! Why couldn't there just be stuff on it?[end if]".
The restroom part 5 is some text that varies. The restroom part 5 is "

There is an exit to the east. Don't worry, it is unblocked. You don't like doors? Why not try going north, that entranceway is unblocked. There is an exit to the south. Don't worry, it is unblocked. There is an exit to the west. Don't worry, it is unguarded.".
The description of r_10 is "[restroom part 0][restroom part 1][restroom part 2][restroom part 3][restroom part 4][restroom part 5]".

The r_11 is mapped west of r_10.
The r_9 is mapped south of r_10.
The r_16 is mapped north of r_10.
The r_7 is mapped east of r_10.
Understand "workshop" as r_11.
The internal name of r_11 is "workshop".
The printed name of r_11 is "-= Workshop =-".
The workshop part 0 is some text that varies. The workshop part 0 is "You find yourself in a workshop. An usual kind of place.

 You can make out a suitcase.[if c_2 is open and there is something in the c_2] The suitcase contains [a list of things in the c_2].[end if]".
The workshop part 1 is some text that varies. The workshop part 1 is "[if c_2 is open and the c_2 contains nothing] The suitcase is empty! What a waste of a day![end if]".
The workshop part 2 is some text that varies. The workshop part 2 is " You smell an intriguing smell, and follow it to an armchair. [if there is something on the s_2]You see [a list of things on the s_2] on the armchair. There's something strange about this thing being here, but you don't have time to worry about that now.[end if]".
The workshop part 3 is some text that varies. The workshop part 3 is "[if there is nothing on the s_2]But there isn't a thing on it.[end if]".
The workshop part 4 is some text that varies. The workshop part 4 is "

There is an unguarded exit to the east. You need an unblocked exit? You should try going north. There is an unblocked exit to the south. You don't like doors? Why not try going west, that entranceway is unblocked.".
The description of r_11 is "[workshop part 0][workshop part 1][workshop part 2][workshop part 3][workshop part 4]".

The r_14 is mapped west of r_11.
The r_12 is mapped south of r_11.
The r_15 is mapped north of r_11.
The r_10 is mapped east of r_11.
Understand "office" as r_14.
The internal name of r_14 is "office".
The printed name of r_14 is "-= Office =-".
The office part 0 is some text that varies. The office part 0 is "You're now in the office.

 You can make out a shelf. The shelf is ordinary.[if there is something on the s_3] On the shelf you make out [a list of things on the s_3]. Something scurries by right in the corner of your eye. Probably nothing.[end if]".
The office part 1 is some text that varies. The office part 1 is "[if there is nothing on the s_3] But the thing is empty, unfortunately.[end if]".
The office part 2 is some text that varies. The office part 2 is "

There is an unblocked exit to the east. There is an exit to the south. Don't worry, it is unblocked.".
The description of r_14 is "[office part 0][office part 1][office part 2]".

The r_13 is mapped south of r_14.
The r_11 is mapped east of r_14.
Understand "study" as r_12.
The internal name of r_12 is "study".
The printed name of r_12 is "-= Study =-".
The study part 0 is some text that varies. The study part 0 is "You arrive in a study. A standard kind of place. You decide to just list off a complete list of everything you see in the room, because hey, why not?

 You make out [if c_3 is locked]a locked[else if c_3 is open]an opened[otherwise]a closed[end if]".
The study part 1 is some text that varies. The study part 1 is " portmanteau right there by you.[if c_3 is open and there is something in the c_3] The portmanteau contains [a list of things in the c_3].[end if]".
The study part 2 is some text that varies. The study part 2 is "[if c_3 is open and the c_3 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The study part 3 is some text that varies. The study part 3 is " You make out a toolbox.[if c_4 is open and there is something in the c_4] The toolbox contains [a list of things in the c_4].[end if]".
The study part 4 is some text that varies. The study part 4 is "[if c_4 is open and the c_4 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The study part 5 is some text that varies. The study part 5 is " You see [if c_5 is locked]a locked[else if c_5 is open]an opened[otherwise]a closed[end if]".
The study part 6 is some text that varies. The study part 6 is " standard looking cabinet nearby.[if c_5 is open and there is something in the c_5] The cabinet contains [a list of things in the c_5].[end if]".
The study part 7 is some text that varies. The study part 7 is "[if c_5 is open and the c_5 contains nothing] The cabinet is empty! This is the worst thing that could possibly happen, ever![end if]".
The study part 8 is some text that varies. The study part 8 is "

There is an unblocked exit to the east. You need an unblocked exit? You should try going north. You need an unblocked exit? You should try going west.".
The description of r_12 is "[study part 0][study part 1][study part 2][study part 3][study part 4][study part 5][study part 6][study part 7][study part 8]".

The r_13 is mapped west of r_12.
The r_11 is mapped north of r_12.
The r_9 is mapped east of r_12.
Understand "bar" as r_13.
The internal name of r_13 is "bar".
The printed name of r_13 is "-= Bar =-".
The bar part 0 is some text that varies. The bar part 0 is "You arrive in a bar. A standard kind of place.



There is an exit to the east. Don't worry, it is unblocked. There is an unguarded exit to the north.".
The description of r_13 is "[bar part 0]".

The r_14 is mapped north of r_13.
The r_12 is mapped east of r_13.
Understand "pantry" as r_16.
The internal name of r_16 is "pantry".
The printed name of r_16 is "-= Pantry =-".
The pantry part 0 is some text that varies. The pantry part 0 is "You are in a pantry. An usual one. You can barely contain your excitement.



There is an exit to the south. Don't worry, it is unguarded. There is an exit to the west. Don't worry, it is unguarded.".
The description of r_16 is "[pantry part 0]".

The r_15 is mapped west of r_16.
The r_10 is mapped south of r_16.
Understand "chamber" as r_15.
The internal name of r_15 is "chamber".
The printed name of r_15 is "-= Chamber =-".
The chamber part 0 is some text that varies. The chamber part 0 is "Well, here we are in a chamber.



 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The chamber part 1 is some text that varies. The chamber part 1 is " hatch leading north. There is an exit to the east. Don't worry, it is unguarded. You don't like doors? Why not try going south, that entranceway is unguarded.".
The description of r_15 is "[chamber part 0][chamber part 1]".

The r_11 is mapped south of r_15.
north of r_15 and south of r_17 is a door called d_1.
The r_16 is mapped east of r_15.
Understand "attic" as r_18.
The internal name of r_18 is "attic".
The printed name of r_18 is "-= Attic =-".
The attic part 0 is some text that varies. The attic part 0 is "Well, here we are in an attic.

 You can make out a counter. Why don't you take a picture of it, it'll last longer! [if there is something on the s_4]On the counter you see [a list of things on the s_4].[end if]".
The attic part 1 is some text that varies. The attic part 1 is "[if there is nothing on the s_4]Unfortunately, there isn't a thing on it. This always happens![end if]".
The attic part 2 is some text that varies. The attic part 2 is " You hear a noise behind you and spin around, but you can't see anything other than a stand. The stand is normal.[if there is something on the s_5] On the stand you see [a list of things on the s_5].[end if]".
The attic part 3 is some text that varies. The attic part 3 is "[if there is nothing on the s_5] However, the stand, like an empty stand, has nothing on it.[end if]".
The attic part 4 is some text that varies. The attic part 4 is "

 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The attic part 5 is some text that varies. The attic part 5 is " Microsoft limited edition hatch leading south. You need an unblocked exit? You should try going west.".
The description of r_18 is "[attic part 0][attic part 1][attic part 2][attic part 3][attic part 4][attic part 5]".

The r_19 is mapped west of r_18.
south of r_18 and north of r_17 is a door called d_0.
Understand "bedchamber" as r_19.
The internal name of r_19 is "bedchamber".
The printed name of r_19 is "-= Bedchamber =-".
The bedchamber part 0 is some text that varies. The bedchamber part 0 is "You've just shown up in a bedchamber.

 You can make out a dresser. Wow, isn't TextWorld just the best?[if c_6 is open and there is something in the c_6] The dresser contains [a list of things in the c_6]. Wow, isn't TextWorld just the best?[end if]".
The bedchamber part 1 is some text that varies. The bedchamber part 1 is "[if c_6 is open and the c_6 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The bedchamber part 2 is some text that varies. The bedchamber part 2 is " You see a mantle. You can't really describe the mantle besides that it's typical. [if there is something on the s_6]You see [a list of things on the s_6] on the mantle.[end if]".
The bedchamber part 3 is some text that varies. The bedchamber part 3 is "[if there is nothing on the s_6]However, the mantle, like an empty mantle, has nothing on it.[end if]".
The bedchamber part 4 is some text that varies. The bedchamber part 4 is "

You need an unguarded exit? You should try going east.".
The description of r_19 is "[bedchamber part 0][bedchamber part 1][bedchamber part 2][bedchamber part 3][bedchamber part 4]".

The r_18 is mapped east of r_19.
Understand "parlor" as r_2.
The internal name of r_2 is "parlor".
The printed name of r_2 is "-= Parlor =-".
The parlor part 0 is some text that varies. The parlor part 0 is "You've entered a parlor.

 You can make out a box.[if c_7 is open and there is something in the c_7] The box contains [a list of things in the c_7].[end if]".
The parlor part 1 is some text that varies. The parlor part 1 is "[if c_7 is open and the c_7 contains nothing] The box is empty! What a waste of a day![end if]".
The parlor part 2 is some text that varies. The parlor part 2 is "

 There is [if d_2 is open]an open[otherwise]a closed[end if]".
The parlor part 3 is some text that varies. The parlor part 3 is " gateway leading south. There is [if d_3 is open]an open[otherwise]a closed[end if]".
The parlor part 4 is some text that varies. The parlor part 4 is " door leading west.".
The description of r_2 is "[parlor part 0][parlor part 1][parlor part 2][parlor part 3][parlor part 4]".

west of r_2 and east of r_0 is a door called d_3.
south of r_2 and north of r_3 is a door called d_2.
Understand "playroom" as r_3.
The internal name of r_3 is "playroom".
The printed name of r_3 is "-= Playroom =-".
The playroom part 0 is some text that varies. The playroom part 0 is "You find yourself in a playroom. An ordinary one.

 You lean against the wall, inadvertently pressing a secret button. The wall opens up to reveal a desk. The desk is normal.[if there is something on the s_7] On the desk you can see [a list of things on the s_7].[end if]".
The playroom part 1 is some text that varies. The playroom part 1 is "[if there is nothing on the s_7] But the thing is empty, unfortunately. You think about smashing the desk to bits, throwing the bits into the garbage, etc, until you get bored.[end if]".
The playroom part 2 is some text that varies. The playroom part 2 is "

 There is [if d_2 is open]an open[otherwise]a closed[end if]".
The playroom part 3 is some text that varies. The playroom part 3 is " gateway leading north. There is an unguarded exit to the south. There is an exit to the west. Don't worry, it is unguarded.".
The description of r_3 is "[playroom part 0][playroom part 1][playroom part 2][playroom part 3]".

The r_4 is mapped west of r_3.
The r_5 is mapped south of r_3.
north of r_3 and south of r_2 is a door called d_2.
Understand "studio" as r_4.
The internal name of r_4 is "studio".
The printed name of r_4 is "-= Studio =-".
The studio part 0 is some text that varies. The studio part 0 is "You've just sauntered into a studio.



You don't like doors? Why not try going east, that entranceway is unblocked. There is an unblocked exit to the south.".
The description of r_4 is "[studio part 0]".

The r_7 is mapped south of r_4.
The r_3 is mapped east of r_4.
Understand "washroom" as r_5.
The internal name of r_5 is "washroom".
The printed name of r_5 is "-= Washroom =-".
The washroom part 0 is some text that varies. The washroom part 0 is "You've just shown up in a washroom.

 You make out a table. The table is standard.[if there is something on the s_8] On the table you see [a list of things on the s_8].[end if]".
The washroom part 1 is some text that varies. The washroom part 1 is "[if there is nothing on the s_8] But oh no! there's nothing on this piece of garbage. This always happens![end if]".
The washroom part 2 is some text that varies. The washroom part 2 is "

You need an unguarded exit? You should try going north. You don't like doors? Why not try going south, that entranceway is unguarded. There is an exit to the west. Don't worry, it is unblocked.".
The description of r_5 is "[washroom part 0][washroom part 1][washroom part 2]".

The r_7 is mapped west of r_5.
The r_6 is mapped south of r_5.
The r_3 is mapped north of r_5.
Understand "shower" as r_7.
The internal name of r_7 is "shower".
The printed name of r_7 is "-= Shower =-".
The shower part 0 is some text that varies. The shower part 0 is "You are in a shower. A normal kind of place. You begin to take stock of what's here.



There is an unblocked exit to the east. There is an unguarded exit to the north. There is an unguarded exit to the south. There is an exit to the west. Don't worry, it is unblocked.".
The description of r_7 is "[shower part 0]".

The r_10 is mapped west of r_7.
The r_8 is mapped south of r_7.
The r_4 is mapped north of r_7.
The r_5 is mapped east of r_7.
Understand "salon" as r_6.
The internal name of r_6 is "salon".
The printed name of r_6 is "-= Salon =-".
The salon part 0 is some text that varies. The salon part 0 is "You have fallen into a salon. Not the salon you'd expect. No, this is a salon.

 You make out a chest. You idly wonder how they came up with the name TextWorld for this place. It's pretty fitting.[if c_8 is open and there is something in the c_8] The chest contains [a list of things in the c_8].[end if]".
The salon part 1 is some text that varies. The salon part 1 is "[if c_8 is open and the c_8 contains nothing] The chest is empty! What a waste of a day![end if]".
The salon part 2 is some text that varies. The salon part 2 is " You scan the room for a recliner, and you find a recliner. [if there is something on the s_9]On the recliner you see [a list of things on the s_9].[end if]".
The salon part 3 is some text that varies. The salon part 3 is "[if there is nothing on the s_9]But the thing is empty, unfortunately.[end if]".
The salon part 4 is some text that varies. The salon part 4 is "

You need an unguarded exit? You should try going north. There is an unblocked exit to the west.".
The description of r_6 is "[salon part 0][salon part 1][salon part 2][salon part 3][salon part 4]".

The r_8 is mapped west of r_6.
The r_5 is mapped north of r_6.
Understand "kitchenette" as r_8.
The internal name of r_8 is "kitchenette".
The printed name of r_8 is "-= Kitchenette =-".
The kitchenette part 0 is some text that varies. The kitchenette part 0 is "You are in a kitchenette. An ordinary kind of place. You decide to just list off a complete list of everything you see in the room, because hey, why not?



There is an exit to the east. Don't worry, it is unguarded. There is an unblocked exit to the north. There is an exit to the west. Don't worry, it is unguarded.".
The description of r_8 is "[kitchenette part 0]".

The r_9 is mapped west of r_8.
The r_7 is mapped north of r_8.
The r_6 is mapped east of r_8.
Understand "recreation zone" as r_9.
The internal name of r_9 is "recreation zone".
The printed name of r_9 is "-= Recreation Zone =-".
The recreation zone part 0 is some text that varies. The recreation zone part 0 is "You arrive in a recreation zone. A standard kind of place.

 You smell an awful smell, and follow it to a bed stand. [if there is something on the s_10]You see [a list of things on the s_10] on the bed stand.[end if]".
The recreation zone part 1 is some text that varies. The recreation zone part 1 is "[if there is nothing on the s_10]But there isn't a thing on it.[end if]".
The recreation zone part 2 is some text that varies. The recreation zone part 2 is "

There is an unguarded exit to the east. You need an unguarded exit? You should try going north. There is an exit to the west. Don't worry, it is unblocked.".
The description of r_9 is "[recreation zone part 0][recreation zone part 1][recreation zone part 2]".

The r_12 is mapped west of r_9.
The r_10 is mapped north of r_9.
The r_8 is mapped east of r_9.
Understand "cubicle" as r_17.
The internal name of r_17 is "cubicle".
The printed name of r_17 is "-= Cubicle =-".
The cubicle part 0 is some text that varies. The cubicle part 0 is "You've entered a cubicle.



 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The cubicle part 1 is some text that varies. The cubicle part 1 is " Microsoft limited edition hatch leading north. There is [if d_1 is open]an open[otherwise]a closed[end if]".
The cubicle part 2 is some text that varies. The cubicle part 2 is " hatch leading south.".
The description of r_17 is "[cubicle part 0][cubicle part 1][cubicle part 2]".

south of r_17 and north of r_15 is a door called d_1.
north of r_17 and south of r_18 is a door called d_0.

The c_0 and the c_1 and the c_2 and the c_3 and the c_4 and the c_5 and the c_6 and the c_7 and the c_8 are containers.
The c_0 and the c_1 and the c_2 and the c_3 and the c_4 and the c_5 and the c_6 and the c_7 and the c_8 are privately-named.
The d_3 and the d_1 and the d_0 and the d_2 are doors.
The d_3 and the d_1 and the d_0 and the d_2 are privately-named.
The k_0 and the k_1 are keys.
The k_0 and the k_1 are privately-named.
The o_0 and the o_1 are object-likes.
The o_0 and the o_1 are privately-named.
The r_0 and the r_1 and the r_10 and the r_11 and the r_14 and the r_12 and the r_13 and the r_16 and the r_15 and the r_18 and the r_19 and the r_2 and the r_3 and the r_4 and the r_5 and the r_7 and the r_6 and the r_8 and the r_9 and the r_17 are rooms.
The r_0 and the r_1 and the r_10 and the r_11 and the r_14 and the r_12 and the r_13 and the r_16 and the r_15 and the r_18 and the r_19 and the r_2 and the r_3 and the r_4 and the r_5 and the r_7 and the r_6 and the r_8 and the r_9 and the r_17 are privately-named.
The s_0 and the s_1 and the s_10 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 and the s_9 are supporters.
The s_0 and the s_1 and the s_10 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 and the s_9 are privately-named.

The description of d_3 is "it's a grand door [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_3 is "door".
Understand "door" as d_3.
The d_3 is open.
The description of d_1 is "The hatch looks commanding. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_1 is "hatch".
Understand "hatch" as d_1.
The d_1 is locked.
The description of d_0 is "The Microsoft limited edition hatch looks hefty. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_0 is "Microsoft limited edition hatch".
Understand "Microsoft limited edition hatch" as d_0.
Understand "Microsoft" as d_0.
Understand "limited" as d_0.
Understand "edition" as d_0.
Understand "hatch" as d_0.
The d_0 is locked.
The description of d_2 is "it's an imposing gateway [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_2 is "gateway".
Understand "gateway" as d_2.
The d_2 is open.
The description of c_0 is "The trunk looks strong, and impossible to break. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_0 is "trunk".
Understand "trunk" as c_0.
The c_0 is in r_1.
The c_0 is closed.
The description of c_1 is "The case looks strong, and impossible to destroy. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_1 is "case".
Understand "case" as c_1.
The c_1 is in r_10.
The c_1 is open.
The description of c_2 is "The suitcase looks strong, and impossible to crack. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_2 is "suitcase".
Understand "suitcase" as c_2.
The c_2 is in r_11.
The c_2 is locked.
The description of c_3 is "The portmanteau looks strong, and impossible to crack. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_3 is "portmanteau".
Understand "portmanteau" as c_3.
The c_3 is in r_12.
The c_3 is closed.
The description of c_4 is "The toolbox looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_4 is "toolbox".
Understand "toolbox" as c_4.
The c_4 is in r_12.
The c_4 is open.
The description of c_5 is "The cabinet looks strong, and impossible to destroy. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_5 is "cabinet".
Understand "cabinet" as c_5.
The c_5 is in r_12.
The c_5 is locked.
The description of c_6 is "The dresser looks strong, and impossible to crack. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_6 is "dresser".
Understand "dresser" as c_6.
The c_6 is in r_19.
The c_6 is locked.
The description of c_7 is "The box looks strong, and impossible to break. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_7 is "box".
Understand "box" as c_7.
The c_7 is in r_2.
The c_7 is open.
The description of c_8 is "The chest looks strong, and impossible to destroy. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_8 is "chest".
Understand "chest" as c_8.
The c_8 is in r_6.
The c_8 is open.
The description of k_0 is "The passkey is surprisingly heavy.".
The printed name of k_0 is "passkey".
Understand "passkey" as k_0.
The k_0 is in r_19.
The matching key of the d_1 is the k_0.
The description of k_1 is "The Microsoft limited edition latchkey is cold to the touch".
The printed name of k_1 is "Microsoft limited edition latchkey".
Understand "Microsoft limited edition latchkey" as k_1.
Understand "Microsoft" as k_1.
Understand "limited" as k_1.
Understand "edition" as k_1.
Understand "latchkey" as k_1.
The k_1 is in r_17.
The matching key of the d_0 is the k_1.
The description of o_0 is "The plant appears to be to fit in here".
The printed name of o_0 is "plant".
Understand "plant" as o_0.
The o_0 is in r_2.
The description of o_1 is "The cd is dirty.".
The printed name of o_1 is "cd".
Understand "cd" as o_1.
The o_1 is in r_0.
The description of s_0 is "The bed is an unstable piece of junk.".
The printed name of s_0 is "bed".
Understand "bed" as s_0.
The s_0 is in r_0.
The description of s_1 is "The board is undependable.".
The printed name of s_1 is "board".
Understand "board" as s_1.
The s_1 is in r_10.
The description of s_10 is "The bed stand is durable.".
The printed name of s_10 is "bed stand".
Understand "bed stand" as s_10.
Understand "bed" as s_10.
Understand "stand" as s_10.
The s_10 is in r_9.
The description of s_2 is "The armchair is an unstable piece of garbage.".
The printed name of s_2 is "armchair".
Understand "armchair" as s_2.
The s_2 is in r_11.
The description of s_3 is "The shelf is durable.".
The printed name of s_3 is "shelf".
Understand "shelf" as s_3.
The s_3 is in r_14.
The description of s_4 is "The counter is reliable.".
The printed name of s_4 is "counter".
Understand "counter" as s_4.
The s_4 is in r_18.
The description of s_5 is "The stand is stable.".
The printed name of s_5 is "stand".
Understand "stand" as s_5.
The s_5 is in r_18.
The description of s_6 is "The mantle is reliable.".
The printed name of s_6 is "mantle".
Understand "mantle" as s_6.
The s_6 is in r_19.
The description of s_7 is "The desk is unstable.".
The printed name of s_7 is "desk".
Understand "desk" as s_7.
The s_7 is in r_3.
The description of s_8 is "The table is balanced.".
The printed name of s_8 is "table".
Understand "table" as s_8.
The s_8 is in r_5.
The description of s_9 is "The recliner is solidly built.".
The printed name of s_9 is "recliner".
Understand "recliner" as s_9.
The s_9 is in r_6.


The player is in r_17.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with "take Microsoft limited edition latchkey / unlock Microsoft limited edition hatch with Microsoft limited edition latchkey / open Microsoft limited edition hatch / go north / go west / take passkey / go east / go south / unlock hatch with passkey / open hatch / go south / go south / go south / go east / go north / go east / go north / go east / go north / take plant"

Every turn:
	if quest0 completed is true:
		do nothing;
	else if The player carries the o_1:
		end the story; [Lost]
	else if The player is in r_2 and The player carries the o_0:
		increase the score by 1; [Quest completed]
		if 1 is 1 [always true]:
			Now the quest0 completed is true;

Use scoring. The maximum score is 1.
This is the simpler notify score changes rule:
	If the score is not the last notified score:
		let V be the score - the last notified score;
		if V > 0:
			say "Your score has just gone up by [V in words] ";
		else:
			say "Your score changed by [V in words] ";
		if V >= -1 and V <= 1:
			say "point.";
		else:
			say "points.";
		Now the last notified score is the score;
	if quest0 completed is true:
		end the story finally; [Win]

The simpler notify score changes rule substitutes for the notify score changes rule.

Rule for listing nondescript items:
	stop.

Rule for printing the banner text:
	say "[fixed letter spacing]";
	say "                    ________  ________  __    __  ________        [line break]";
	say "                   |        \|        \|  \  |  \|        \       [line break]";
	say "                    \$$$$$$$$| $$$$$$$$| $$  | $$ \$$$$$$$$       [line break]";
	say "                      | $$   | $$__     \$$\/  $$   | $$          [line break]";
	say "                      | $$   | $$  \     >$$  $$    | $$          [line break]";
	say "                      | $$   | $$$$$    /  $$$$\    | $$          [line break]";
	say "                      | $$   | $$_____ |  $$ \$$\   | $$          [line break]";
	say "                      | $$   | $$     \| $$  | $$   | $$          [line break]";
	say "                       \$$    \$$$$$$$$ \$$   \$$    \$$          [line break]";
	say "              __       __   ______   _______   __        _______  [line break]";
	say "             |  \  _  |  \ /      \ |       \ |  \      |       \ [line break]";
	say "             | $$ / \ | $$|  $$$$$$\| $$$$$$$\| $$      | $$$$$$$\[line break]";
	say "             | $$/  $\| $$| $$  | $$| $$__| $$| $$      | $$  | $$[line break]";
	say "             | $$  $$$\ $$| $$  | $$| $$    $$| $$      | $$  | $$[line break]";
	say "             | $$ $$\$$\$$| $$  | $$| $$$$$$$\| $$      | $$  | $$[line break]";
	say "             | $$$$  \$$$$| $$__/ $$| $$  | $$| $$_____ | $$__/ $$[line break]";
	say "             | $$$    \$$$ \$$    $$| $$  | $$| $$     \| $$    $$[line break]";
	say "              \$$      \$$  \$$$$$$  \$$   \$$ \$$$$$$$$ \$$$$$$$ [line break]";
	say "[variable letter spacing][line break]";
	say "[objective][line break]".

Include Basic Screen Effects by Emily Short.

Rule for printing the player's obituary:
	if story has ended finally:
		center "*** The End ***";
	else:
		center "*** You lost! ***";
	say paragraph break;
	if maximum score is -32768:
		say "You scored a total of [score] point[s], in [turn count] turn[s].";
	else:
		say "You scored [score] out of a possible [maximum score], in [turn count] turn[s].";
	[wait for any key;
	stop game abruptly;]
	rule succeeds.

Carry out requesting the score:
	if maximum score is -32768:
		say "You have so far scored [score] point[s], in [turn count] turn[s].";
	else:
		say "You have so far scored [score] out of a possible [maximum score], in [turn count] turn[s].";
	rule succeeds.

Rule for implicitly taking something (called target):
	if target is fixed in place:
		say "The [target] is fixed in place.";
	otherwise:
		say "You need to take the [target] first.";
		set pronouns from target;
	stop.

Does the player mean doing something:
	if the noun is not nothing and the second noun is nothing and the player's command matches the text printed name of the noun:
		it is likely;
	if the noun is nothing and the second noun is not nothing and the player's command matches the text printed name of the second noun:
		it is likely;
	if the noun is not nothing and the second noun is not nothing and the player's command matches the text printed name of the noun and the player's command matches the text printed name of the second noun:
		it is very likely.  [Handle action with two arguments.]

Printing the content of the room is an activity.
Rule for printing the content of the room:
	let R be the location of the player;
	say "Room contents:[line break]";
	list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the world is an activity.
Rule for printing the content of the world:
	let L be the list of the rooms;
	say "World: [line break]";
	repeat with R running through L:
		say "  [the internal name of R][line break]";
	repeat with R running through L:
		say "[the internal name of R]:[line break]";
		if the list of things in R is empty:
			say "  nothing[line break]";
		otherwise:
			list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the inventory is an activity.
Rule for printing the content of the inventory:
	say "You are carrying: ";
	list the contents of the player, as a sentence, giving inventory information, including all contents;
	say ".".

The print standard inventory rule is not listed in any rulebook.
Carry out taking inventory (this is the new print inventory rule):
	say "You are carrying: ";
	list the contents of the player, as a sentence, giving inventory information, including all contents;
	say ".".

Printing the content of nowhere is an activity.
Rule for printing the content of nowhere:
	say "Nowhere:[line break]";
	let L be the list of the off-stage things;
	repeat with thing running through L:
		say "  [thing][line break]";

Printing the things on the floor is an activity.
Rule for printing the things on the floor:
	let R be the location of the player;
	let L be the list of things in R;
	remove yourself from L;
	remove the list of containers from L;
	remove the list of supporters from L;
	remove the list of doors from L;
	if the number of entries in L is greater than 0:
		say "There is [L with indefinite articles] on the floor.";

After printing the name of something (called target) while
printing the content of the room
or printing the content of the world
or printing the content of the inventory
or printing the content of nowhere:
	follow the property-aggregation rules for the target.

The property-aggregation rules are an object-based rulebook.
The property-aggregation rulebook has a list of text called the tagline.

[At the moment, we only support "open/unlocked", "closed/unlocked" and "closed/locked" for doors and containers.]
[A first property-aggregation rule for an openable open thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable closed thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an lockable unlocked thing (this is the mention unlocked lockable rule):
	add "unlocked" to the tagline.

A property-aggregation rule for an lockable locked thing (this is the mention locked lockable rule):
	add "locked" to the tagline.]

A first property-aggregation rule for an openable lockable open unlocked thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable lockable closed unlocked thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an openable lockable closed locked thing (this is the mention locked openables rule):
	add "locked" to the tagline.

A property-aggregation rule for a lockable thing (called the lockable thing) (this is the mention matching key of lockable rule):
	let X be the matching key of the lockable thing;
	if X is not nothing:
		add "match [X]" to the tagline.

A property-aggregation rule for an edible off-stage thing (this is the mention eaten edible rule):
	add "eaten" to the tagline.

The last property-aggregation rule (this is the print aggregated properties rule):
	if the number of entries in the tagline is greater than 0:
		say " ([tagline])";
		rule succeeds;
	rule fails;

The objective part 0 is some text that varies. The objective part 0 is "Hey, thanks for coming over to TextWorld! Please pick up the plant from the floor of the parlor.".

An objective is some text that varies. The objective is "[objective part 0]".
Printing the objective is an action applying to nothing.
Carry out printing the objective:
	say "[objective]".

Understand "goal" as printing the objective.

The taking action has an object called previous locale (matched as "from").

Setting action variables for taking:
	now previous locale is the holder of the noun.

Report taking something from the location:
	say "You pick up [the noun] from the ground." instead.

Report taking something:
	say "You take [the noun] from [the previous locale]." instead.

Report dropping something:
	say "You drop [the noun] on the ground." instead.

The print state option is a truth state that varies.
The print state option is usually false.

Turning on the print state option is an action applying to nothing.
Carry out turning on the print state option:
	Now the print state option is true.

Turning off the print state option is an action applying to nothing.
Carry out turning off the print state option:
	Now the print state option is false.

Printing the state is an activity.
Rule for printing the state:
	let R be the location of the player;
	say "Room: [line break] [the internal name of R][line break]";
	[say "[line break]";
	carry out the printing the content of the room activity;]
	say "[line break]";
	carry out the printing the content of the world activity;
	say "[line break]";
	carry out the printing the content of the inventory activity;
	say "[line break]";
	carry out the printing the content of nowhere activity;
	say "[line break]".

Printing the entire state is an action applying to nothing.
Carry out printing the entire state:
	say "-=STATE START=-[line break]";
	carry out the printing the state activity;
	say "[line break]Score:[line break] [score]/[maximum score][line break]";
	say "[line break]Objective:[line break] [objective][line break]";
	say "[line break]Inventory description:[line break]";
	say "  You are carrying: [a list of things carried by the player].[line break]";
	say "[line break]Room description:[line break]";
	try looking;
	say "[line break]-=STATE STOP=-";

Every turn:
	if extra description command option is true:
		say "<description>";
		try looking;
		say "</description>";
	if extra inventory command option is true:
		say "<inventory>";
		try taking inventory;
		say "</inventory>";
	if extra score command option is true:
		say "<score>[line break][score][line break]</score>";
	if extra score command option is true:
		say "<moves>[line break][turn count][line break]</moves>";
	if print state option is true:
		try printing the entire state;

When play ends:
	if print state option is true:
		try printing the entire state;

After looking:
	carry out the printing the things on the floor activity.

Understand "print_state" as printing the entire state.
Understand "enable print state option" as turning on the print state option.
Understand "disable print state option" as turning off the print state option.

Before going through a closed door (called the blocking door):
	say "You have to open the [blocking door] first.";
	stop.

Before opening a locked door (called the locked door):
	let X be the matching key of the locked door;
	if X is nothing:
		say "The [locked door] is welded shut.";
	otherwise:
		say "You have to unlock the [locked door] with the [X] first.";
	stop.

Before opening a locked container (called the locked container):
	let X be the matching key of the locked container;
	if X is nothing:
		say "The [locked container] is welded shut.";
	otherwise:
		say "You have to unlock the [locked container] with the [X] first.";
	stop.

Displaying help message is an action applying to nothing.
Carry out displaying help message:
	say "[fixed letter spacing]Available commands:[line break]";
	say "  look:                describe the current room[line break]";
	say "  goal:                print the goal of this game[line break]";
	say "  inventory:           print player's inventory[line break]";
	say "  go <dir>:            move the player north, east, south or west[line break]";
	say "  examine ...:         examine something more closely[line break]";
	say "  eat ...:             eat edible food[line break]";
	say "  open ...:            open a door or a container[line break]";
	say "  close ...:           close a door or a container[line break]";
	say "  drop ...:            drop an object on the floor[line break]";
	say "  take ...:            take an object that is on the floor[line break]";
	say "  put ... on ...:      place an object on a supporter[line break]";
	say "  take ... from ...:   take an object from a container or a supporter[line break]";
	say "  insert ... into ...: place an object into a container[line break]";
	say "  lock ... with ...:   lock a door or a container with a key[line break]";
	say "  unlock ... with ...: unlock a door or a container with a key[line break]";

Understand "help" as displaying help message.

Taking all is an action applying to nothing.
Check taking all:
	say "You have to be more specific!";
	rule fails.

Understand "take all" as taking all.
Understand "get all" as taking all.
Understand "pick up all" as taking all.

Understand "take each" as taking all.
Understand "get each" as taking all.
Understand "pick up each" as taking all.

Understand "take everything" as taking all.
Understand "get everything" as taking all.
Understand "pick up everything" as taking all.

The extra description command option is a truth state that varies.
The extra description command option is usually false.

Turning on the extra description command option is an action applying to nothing.
Carry out turning on the extra description command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra description command option is true.

Understand "tw-extra-infos description" as turning on the extra description command option.

The extra inventory command option is a truth state that varies.
The extra inventory command option is usually false.

Turning on the extra inventory command option is an action applying to nothing.
Carry out turning on the extra inventory command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra inventory command option is true.

Understand "tw-extra-infos inventory" as turning on the extra inventory command option.

The extra score command option is a truth state that varies.
The extra score command option is usually false.

Turning on the extra score command option is an action applying to nothing.
Carry out turning on the extra score command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra score command option is true.

Understand "tw-extra-infos score" as turning on the extra score command option.

The extra moves command option is a truth state that varies.
The extra moves command option is usually false.

Turning on the extra moves command option is an action applying to nothing.
Carry out turning on the extra moves command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra moves command option is true.

Understand "tw-extra-infos moves" as turning on the extra moves command option.

To trace the actions:
	(- trace_actions = 1; -).

Tracing the actions is an action applying to nothing.
Carry out tracing the actions:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	trace the actions;

Understand "tw-trace-actions" as tracing the actions.

The restrict commands option is a truth state that varies.
The restrict commands option is usually false.

Turning on the restrict commands option is an action applying to nothing.
Carry out turning on the restrict commands option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the restrict commands option is true.

Understand "restrict commands" as turning on the restrict commands option.

The taking allowed flag is a truth state that varies.
The taking allowed flag is usually false.

Before removing something from something:
	now the taking allowed flag is true.

After removing something from something:
	now the taking allowed flag is false.

Before taking a thing (called the object) when the object is on a supporter (called the supporter):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [supporter] instead.";
		rule fails.

Before of taking a thing (called the object) when the object is in a container (called the container):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [container] instead.";
		rule fails.

Understand "take [something]" as removing it from.

Rule for supplying a missing second noun while removing:
	if restrict commands option is false and noun is on a supporter (called the supporter):
		now the second noun is the supporter;
	else if restrict commands option is false and noun is in a container (called the container):
		now the second noun is the container;
	else:
		try taking the noun;
		say ""; [Needed to avoid printing a default message.]

The version number is always 1.

Reporting the version number is an action applying to nothing.
Carry out reporting the version number:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	say "[version number]".

Understand "tw-print version" as reporting the version number.

Reporting max score is an action applying to nothing.
Carry out reporting max score:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	if maximum score is -32768:
		say "infinity";
	else:
		say "[maximum score]".

Understand "tw-print max_score" as reporting max score.

To print id of (something - thing):
	(- print {something}, "^"; -).

Printing the id of player is an action applying to nothing.
Carry out printing the id of player:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	print id of player.

Printing the id of EndOfObject is an action applying to nothing.
Carry out printing the id of EndOfObject:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	print id of EndOfObject.

Understand "tw-print player id" as printing the id of player.
Understand "tw-print EndOfObject id" as printing the id of EndOfObject.

There is a EndOfObject.

