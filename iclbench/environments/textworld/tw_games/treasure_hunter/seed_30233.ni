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


The r_0 and the r_16 and the r_1 and the r_2 and the r_10 and the r_5 and the r_11 and the r_12 and the r_9 and the r_13 and the r_14 and the r_15 and the r_18 and the r_17 and the r_19 and the r_4 and the r_3 and the r_7 and the r_6 and the r_8 are rooms.

Understand "vault" as r_0.
The internal name of r_0 is "vault".
The printed name of r_0 is "-= Vault =-".
The vault part 0 is some text that varies. The vault part 0 is "Well, here we are in a vault. You begin to take stock of what's in the room.

 You can make out a counter. Make a note of this, you might have to put stuff on or in it later on. The counter is normal.[if there is something on the s_0] On the counter you make out [a list of things on the s_0], so there's that.[end if]".
The vault part 1 is some text that varies. The vault part 1 is "[if there is nothing on the s_0] The counter appears to be empty.[end if]".
The vault part 2 is some text that varies. The vault part 2 is "

 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The vault part 3 is some text that varies. The vault part 3 is " spherical door leading south. There is [if d_3 is open]an open[otherwise]a closed[end if]".
The vault part 4 is some text that varies. The vault part 4 is " rectangular portal leading west.".
The description of r_0 is "[vault part 0][vault part 1][vault part 2][vault part 3][vault part 4]".

west of r_0 and east of r_16 is a door called d_3.
south of r_0 and north of r_1 is a door called d_0.
Understand "cookery" as r_16.
The internal name of r_16 is "cookery".
The printed name of r_16 is "-= Cookery =-".
The cookery part 0 is some text that varies. The cookery part 0 is "Well, here we are in a cookery. You begin looking for stuff.

 You can see a pan. [if there is something on the s_1]On the pan you see [a list of things on the s_1].[end if]".
The cookery part 1 is some text that varies. The cookery part 1 is "[if there is nothing on the s_1]Looks like someone's already been here and taken everything off it, though.[end if]".
The cookery part 2 is some text that varies. The cookery part 2 is "

 There is [if d_3 is open]an open[otherwise]a closed[end if]".
The cookery part 3 is some text that varies. The cookery part 3 is " rectangular portal leading east. There is [if d_2 is open]an open[otherwise]a closed[end if]".
The cookery part 4 is some text that varies. The cookery part 4 is " gateway leading north.".
The description of r_16 is "[cookery part 0][cookery part 1][cookery part 2][cookery part 3][cookery part 4]".

north of r_16 and south of r_17 is a door called d_2.
east of r_16 and west of r_0 is a door called d_3.
Understand "kitchen" as r_1.
The internal name of r_1 is "kitchen".
The printed name of r_1 is "-= Kitchen =-".
The kitchen part 0 is some text that varies. The kitchen part 0 is "You're now in a kitchen.

 You can see a saucepan. The saucepan is typical.[if there is something on the s_2] On the saucepan you make out [a list of things on the s_2].[end if]".
The kitchen part 1 is some text that varies. The kitchen part 1 is "[if there is nothing on the s_2] But the thing hasn't got anything on it.[end if]".
The kitchen part 2 is some text that varies. The kitchen part 2 is "

 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The kitchen part 3 is some text that varies. The kitchen part 3 is " spherical door leading north. You need an unguarded exit? You should try going east. You don't like doors? Why not try going south, that entranceway is unguarded. There is an exit to the west. Don't worry, it is unblocked.".
The description of r_1 is "[kitchen part 0][kitchen part 1][kitchen part 2][kitchen part 3]".

The r_2 is mapped west of r_1.
The r_4 is mapped south of r_1.
north of r_1 and south of r_0 is a door called d_0.
The r_13 is mapped east of r_1.
Understand "dish-pit" as r_2.
The internal name of r_2 is "dish-pit".
The printed name of r_2 is "-= Dish-Pit =-".
The dish-pit part 0 is some text that varies. The dish-pit part 0 is "You find yourself in a dish-pit. A normal one.

 You see a chair. The chair is usual.[if there is something on the s_3] On the chair you see [a list of things on the s_3].[end if]".
The dish-pit part 1 is some text that varies. The dish-pit part 1 is "[if there is nothing on the s_3] But the thing hasn't got anything on it. Sometimes, just sometimes, TextWorld can just be the worst.[end if]".
The dish-pit part 2 is some text that varies. The dish-pit part 2 is " You can see a bowl. The bowl is ordinary.[if there is something on the s_4] On the bowl you make out [a list of things on the s_4].[end if]".
The dish-pit part 3 is some text that varies. The dish-pit part 3 is "[if there is nothing on the s_4] But the thing hasn't got anything on it.[end if]".
The dish-pit part 4 is some text that varies. The dish-pit part 4 is "

There is an exit to the east. Don't worry, it is unguarded. There is an exit to the south. Don't worry, it is unblocked.".
The description of r_2 is "[dish-pit part 0][dish-pit part 1][dish-pit part 2][dish-pit part 3][dish-pit part 4]".

The r_3 is mapped south of r_2.
The r_1 is mapped east of r_2.
Understand "basement" as r_10.
The internal name of r_10 is "basement".
The printed name of r_10 is "-= Basement =-".
The basement part 0 is some text that varies. The basement part 0 is "You arrive in a basement. An usual kind of place.

 [if c_0 is locked]A locked[else if c_0 is open]An open[otherwise]A closed[end if]".
The basement part 1 is some text that varies. The basement part 1 is " safe is close by.[if c_0 is open and there is something in the c_0] The safe contains [a list of things in the c_0].[end if]".
The basement part 2 is some text that varies. The basement part 2 is "[if c_0 is open and the c_0 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The basement part 3 is some text that varies. The basement part 3 is " If you haven't noticed it already, there seems to be something there by the wall, it's a workbench. The workbench is ordinary.[if there is something on the s_5] On the workbench you can make out [a list of things on the s_5]. Hmmm... what else, what else?[end if]".
The basement part 4 is some text that varies. The basement part 4 is "[if there is nothing on the s_5] However, the workbench, like an empty workbench, has nothing on it.[end if]".
The basement part 5 is some text that varies. The basement part 5 is "

There is an exit to the east. Don't worry, it is unblocked. You don't like doors? Why not try going south, that entranceway is unblocked. There is an unguarded exit to the west.".
The description of r_10 is "[basement part 0][basement part 1][basement part 2][basement part 3][basement part 4][basement part 5]".

The r_5 is mapped west of r_10.
The r_9 is mapped south of r_10.
The r_11 is mapped east of r_10.
Understand "office" as r_5.
The internal name of r_5 is "office".
The printed name of r_5 is "-= Office =-".
The office part 0 is some text that varies. The office part 0 is "You find yourself in an office. An usual kind of place.



There is an exit to the east. Don't worry, it is unblocked. There is an unblocked exit to the north. There is an unblocked exit to the west.".
The description of r_5 is "[office part 0]".

The r_4 is mapped west of r_5.
The r_13 is mapped north of r_5.
The r_10 is mapped east of r_5.
Understand "chamber" as r_11.
The internal name of r_11 is "chamber".
The printed name of r_11 is "-= Chamber =-".
The chamber part 0 is some text that varies. The chamber part 0 is "You arrive in a chamber. A normal kind of place. Let's see what's in here.

 Look over there! a desk. [if there is something on the s_6]You see [a list of things on the s_6] on the desk.[end if]".
The chamber part 1 is some text that varies. The chamber part 1 is "[if there is nothing on the s_6]But oh no! there's nothing on this piece of trash.[end if]".
The chamber part 2 is some text that varies. The chamber part 2 is " Oh wow! Is that what I think it is? It is! It's a bed. The bed is standard.[if there is something on the s_7] On the bed you make out [a list of things on the s_7].[end if]".
The chamber part 3 is some text that varies. The chamber part 3 is "[if there is nothing on the s_7] But the thing hasn't got anything on it. What, you think everything in TextWorld should have stuff on it?[end if]".
The chamber part 4 is some text that varies. The chamber part 4 is "

You don't like doors? Why not try going north, that entranceway is unguarded. You don't like doors? Why not try going south, that entranceway is unguarded. There is an unblocked exit to the west.".
The description of r_11 is "[chamber part 0][chamber part 1][chamber part 2][chamber part 3][chamber part 4]".

The r_10 is mapped west of r_11.
The r_12 is mapped south of r_11.
The r_14 is mapped north of r_11.
Understand "bedroom" as r_12.
The internal name of r_12 is "bedroom".
The printed name of r_12 is "-= Bedroom =-".
The bedroom part 0 is some text that varies. The bedroom part 0 is "You arrive in a bedroom. An ordinary one. You try to gain information on your surroundings by using a technique you call 'looking.'

 You make out [if c_1 is locked]a locked[else if c_1 is open]an opened[otherwise]a closed[end if]".
The bedroom part 1 is some text that varies. The bedroom part 1 is " portmanteau.[if c_1 is open and there is something in the c_1] The portmanteau contains [a list of things in the c_1].[end if]".
The bedroom part 2 is some text that varies. The bedroom part 2 is "[if c_1 is open and the c_1 contains nothing] The portmanteau is empty! What a waste of a day![end if]".
The bedroom part 3 is some text that varies. The bedroom part 3 is "

There is an unblocked exit to the north. You need an unguarded exit? You should try going west.".
The description of r_12 is "[bedroom part 0][bedroom part 1][bedroom part 2][bedroom part 3]".

The r_9 is mapped west of r_12.
The r_11 is mapped north of r_12.
Understand "cubicle" as r_9.
The internal name of r_9 is "cubicle".
The printed name of r_9 is "-= Cubicle =-".
The cubicle part 0 is some text that varies. The cubicle part 0 is "You're now in the cubicle. You begin to take stock of what's in the room.



There is an exit to the east. Don't worry, it is unguarded. There is an exit to the north. Don't worry, it is unguarded. There is an unguarded exit to the west.".
The description of r_9 is "[cubicle part 0]".

The r_8 is mapped west of r_9.
The r_10 is mapped north of r_9.
The r_12 is mapped east of r_9.
Understand "bedchamber" as r_13.
The internal name of r_13 is "bedchamber".
The printed name of r_13 is "-= Bedchamber =-".
The bedchamber part 0 is some text that varies. The bedchamber part 0 is "Well, here we are in the bedchamber.

 Were you looking for a bookshelf? Because look over there, it's a bookshelf. The bookshelf is typical.[if there is something on the s_8] On the bookshelf you see [a list of things on the s_8].[end if]".
The bedchamber part 1 is some text that varies. The bedchamber part 1 is "[if there is nothing on the s_8] But there isn't a thing on it. Oh! Why couldn't there just be stuff on it?[end if]".
The bedchamber part 2 is some text that varies. The bedchamber part 2 is "

There is an unguarded exit to the east. You need an unguarded exit? You should try going south. You don't like doors? Why not try going west, that entranceway is unblocked.".
The description of r_13 is "[bedchamber part 0][bedchamber part 1][bedchamber part 2]".

The r_1 is mapped west of r_13.
The r_5 is mapped south of r_13.
The r_15 is mapped east of r_13.
Understand "studio" as r_14.
The internal name of r_14 is "studio".
The printed name of r_14 is "-= Studio =-".
The studio part 0 is some text that varies. The studio part 0 is "You are in a studio. A normal kind of place.



You don't like doors? Why not try going south, that entranceway is unblocked. There is an unguarded exit to the west.".
The description of r_14 is "[studio part 0]".

The r_15 is mapped west of r_14.
The r_11 is mapped south of r_14.
Understand "study" as r_15.
The internal name of r_15 is "study".
The printed name of r_15 is "-= Study =-".
The study part 0 is some text that varies. The study part 0 is "You've entered a study.

 Look out! It's a- oh, never mind, it's just a box. You shudder, but continue examining the room.[if c_2 is open and there is something in the c_2] The box contains [a list of things in the c_2]. Make a note of this, you might have to put stuff on or in it later on.[end if]".
The study part 1 is some text that varies. The study part 1 is "[if c_2 is open and the c_2 contains nothing] The box is empty! This is the worst thing that could possibly happen, ever![end if]".
The study part 2 is some text that varies. The study part 2 is " You see a mantle. [if there is something on the s_9]On the mantle you see [a list of things on the s_9].[end if]".
The study part 3 is some text that varies. The study part 3 is "[if there is nothing on the s_9]But the thing hasn't got anything on it.[end if]".
The study part 4 is some text that varies. The study part 4 is "

You don't like doors? Why not try going east, that entranceway is unguarded. There is an unblocked exit to the west.".
The description of r_15 is "[study part 0][study part 1][study part 2][study part 3][study part 4]".

The r_13 is mapped west of r_15.
The r_14 is mapped east of r_15.
Understand "bar" as r_18.
The internal name of r_18 is "bar".
The printed name of r_18 is "-= Bar =-".
The bar part 0 is some text that varies. The bar part 0 is "Well, here we are in a bar. I guess you better just go and list everything you see here.



 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The bar part 1 is some text that varies. The bar part 1 is " hatch leading west. There is an unguarded exit to the east.".
The description of r_18 is "[bar part 0][bar part 1]".

west of r_18 and east of r_17 is a door called d_1.
The r_19 is mapped east of r_18.
Understand "sauna" as r_17.
The internal name of r_17 is "sauna".
The printed name of r_17 is "-= Sauna =-".
The sauna part 0 is some text that varies. The sauna part 0 is "You arrive in a sauna. A typical kind of place. You begin looking for stuff.

 You see [if c_3 is locked]a locked[else if c_3 is open]an opened[otherwise]a closed[end if]".
The sauna part 1 is some text that varies. The sauna part 1 is " normal looking trunk right there by you.[if c_3 is open and there is something in the c_3] The trunk contains [a list of things in the c_3]. Hmmm... what else, what else?[end if]".
The sauna part 2 is some text that varies. The sauna part 2 is "[if c_3 is open and the c_3 contains nothing] The trunk is empty, what a horrible day![end if]".
The sauna part 3 is some text that varies. The sauna part 3 is "

 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The sauna part 4 is some text that varies. The sauna part 4 is " hatch leading east. There is [if d_2 is open]an open[otherwise]a closed[end if]".
The sauna part 5 is some text that varies. The sauna part 5 is " gateway leading south.".
The description of r_17 is "[sauna part 0][sauna part 1][sauna part 2][sauna part 3][sauna part 4][sauna part 5]".

south of r_17 and north of r_16 is a door called d_2.
east of r_17 and west of r_18 is a door called d_1.
Understand "laundromat" as r_19.
The internal name of r_19 is "laundromat".
The printed name of r_19 is "-= Laundromat =-".
The laundromat part 0 is some text that varies. The laundromat part 0 is "You find yourself in a laundromat. A standard kind of place. Let's see what's in here.

 You rest your hand against a wall, but you miss the wall and fall onto a shelf. What a coincidence, weren't you just thinking about a shelf? The shelf is usual.[if there is something on the s_10] On the shelf you make out [a list of things on the s_10].[end if]".
The laundromat part 1 is some text that varies. The laundromat part 1 is "[if there is nothing on the s_10] Looks like someone's already been here and taken everything off it, though. What's the point of an empty shelf?[end if]".
The laundromat part 2 is some text that varies. The laundromat part 2 is "

You don't like doors? Why not try going west, that entranceway is unblocked.".
The description of r_19 is "[laundromat part 0][laundromat part 1][laundromat part 2]".

The r_18 is mapped west of r_19.
Understand "attic" as r_4.
The internal name of r_4 is "attic".
The printed name of r_4 is "-= Attic =-".
The attic part 0 is some text that varies. The attic part 0 is "You are in an attic. A normal kind of place. You can barely contain your excitement.

 Look out! It's a- oh, never mind, it's just a crate.[if c_4 is open and there is something in the c_4] The crate contains [a list of things in the c_4]. The light flickers for a second, but nothing else happens.[end if]".
The attic part 1 is some text that varies. The attic part 1 is "[if c_4 is open and the c_4 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The attic part 2 is some text that varies. The attic part 2 is " You make out a rack. The rack is typical.[if there is something on the s_11] On the rack you can make out [a list of things on the s_11].[end if]".
The attic part 3 is some text that varies. The attic part 3 is "[if there is nothing on the s_11] But oh no! there's nothing on this piece of trash.[end if]".
The attic part 4 is some text that varies. The attic part 4 is " Look out! It's a- oh, never mind, it's just a stand. [if there is something on the s_12]You see [a list of things on the s_12] on the stand. Now that's what I call TextWorld![end if]".
The attic part 5 is some text that varies. The attic part 5 is "[if there is nothing on the s_12]But the thing is empty, unfortunately.[end if]".
The attic part 6 is some text that varies. The attic part 6 is "

You need an unblocked exit? You should try going east. You don't like doors? Why not try going north, that entranceway is unblocked. There is an unblocked exit to the south. There is an unblocked exit to the west.".
The description of r_4 is "[attic part 0][attic part 1][attic part 2][attic part 3][attic part 4][attic part 5][attic part 6]".

The r_3 is mapped west of r_4.
The r_7 is mapped south of r_4.
The r_1 is mapped north of r_4.
The r_5 is mapped east of r_4.
Understand "workshop" as r_3.
The internal name of r_3 is "workshop".
The printed name of r_3 is "-= Workshop =-".
The workshop part 0 is some text that varies. The workshop part 0 is "This just in- You, in the workshop. You decide to just list off a complete list of everything you see in the room, because hey, why not?



There is an exit to the east. Don't worry, it is unblocked. You need an unblocked exit? You should try going north. You need an unblocked exit? You should try going south.".
The description of r_3 is "[workshop part 0]".

The r_6 is mapped south of r_3.
The r_2 is mapped north of r_3.
The r_4 is mapped east of r_3.
Understand "garage" as r_7.
The internal name of r_7 is "garage".
The printed name of r_7 is "-= Garage =-".
The garage part 0 is some text that varies. The garage part 0 is "Well, here we are in the garage. You begin to take stock of what's in the room.



There is an unblocked exit to the east. You don't like doors? Why not try going north, that entranceway is unguarded. There is an unguarded exit to the west.".
The description of r_7 is "[garage part 0]".

The r_6 is mapped west of r_7.
The r_4 is mapped north of r_7.
The r_8 is mapped east of r_7.
Understand "spare room" as r_6.
The internal name of r_6 is "spare room".
The printed name of r_6 is "-= Spare Room =-".
The spare room part 0 is some text that varies. The spare room part 0 is "You make a grand eccentric entrance into a spare room.

 If you haven't noticed it already, there seems to be something there by the wall, it's a table. [if there is something on the s_13]You see [a list of things on the s_13] on the table.[end if]".
The spare room part 1 is some text that varies. The spare room part 1 is "[if there is nothing on the s_13]But oh no! there's nothing on this piece of trash. Oh! Why couldn't there just be stuff on it?[end if]".
The spare room part 2 is some text that varies. The spare room part 2 is "

You don't like doors? Why not try going east, that entranceway is unblocked. There is an exit to the north. Don't worry, it is unblocked.".
The description of r_6 is "[spare room part 0][spare room part 1][spare room part 2]".

The r_3 is mapped north of r_6.
The r_7 is mapped east of r_6.
Understand "austere cubicle" as r_8.
The internal name of r_8 is "austere cubicle".
The printed name of r_8 is "-= Austere Cubicle =-".
The austere cubicle part 0 is some text that varies. The austere cubicle part 0 is "You've moved into an austere room. Your mind races to think of what kind of room would be austere. And then it hits you. Of course. You're in the austere cubicle. You begin to take stock of what's here.

 Hey, want to see a mantelpiece? Look over there, a mantelpiece. You wonder idly who left that here. The mantelpiece is normal.[if there is something on the s_14] On the mantelpiece you can make out [a list of things on the s_14].[end if]".
The austere cubicle part 1 is some text that varies. The austere cubicle part 1 is "[if there is nothing on the s_14] But the thing hasn't got anything on it.[end if]".
The austere cubicle part 2 is some text that varies. The austere cubicle part 2 is "

You don't like doors? Why not try going east, that entranceway is unguarded. There is an unguarded exit to the west.".
The description of r_8 is "[austere cubicle part 0][austere cubicle part 1][austere cubicle part 2]".

The r_7 is mapped west of r_8.
The r_9 is mapped east of r_8.

The c_0 and the c_1 and the c_2 and the c_3 and the c_4 are containers.
The c_0 and the c_1 and the c_2 and the c_3 and the c_4 are privately-named.
The d_0 and the d_3 and the d_2 and the d_1 are doors.
The d_0 and the d_3 and the d_2 and the d_1 are privately-named.
The f_0 are foods.
The f_0 are privately-named.
The k_2 and the k_3 and the k_0 and the k_4 and the k_1 are keys.
The k_2 and the k_3 and the k_0 and the k_4 and the k_1 are privately-named.
The r_0 and the r_16 and the r_1 and the r_2 and the r_10 and the r_5 and the r_11 and the r_12 and the r_9 and the r_13 and the r_14 and the r_15 and the r_18 and the r_17 and the r_19 and the r_4 and the r_3 and the r_7 and the r_6 and the r_8 are rooms.
The r_0 and the r_16 and the r_1 and the r_2 and the r_10 and the r_5 and the r_11 and the r_12 and the r_9 and the r_13 and the r_14 and the r_15 and the r_18 and the r_17 and the r_19 and the r_4 and the r_3 and the r_7 and the r_6 and the r_8 are privately-named.
The s_0 and the s_1 and the s_10 and the s_11 and the s_12 and the s_13 and the s_14 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 and the s_9 are supporters.
The s_0 and the s_1 and the s_10 and the s_11 and the s_12 and the s_13 and the s_14 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 and the s_9 are privately-named.

The description of d_0 is "it's a robust spherical door [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_0 is "spherical door".
Understand "spherical door" as d_0.
Understand "spherical" as d_0.
Understand "door" as d_0.
The d_0 is locked.
The description of d_3 is "it's a stuffy rectangular portal [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_3 is "rectangular portal".
Understand "rectangular portal" as d_3.
Understand "rectangular" as d_3.
Understand "portal" as d_3.
The d_3 is locked.
The description of d_2 is "The gateway looks towering. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_2 is "gateway".
Understand "gateway" as d_2.
The d_2 is locked.
The description of d_1 is "it is what it is, a hatch [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_1 is "hatch".
Understand "hatch" as d_1.
The d_1 is locked.
The description of c_0 is "The safe looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_0 is "safe".
Understand "safe" as c_0.
The c_0 is in r_10.
The c_0 is locked.
The description of c_1 is "The portmanteau looks strong, and impossible to crack. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_1 is "portmanteau".
Understand "portmanteau" as c_1.
The c_1 is in r_12.
The c_1 is locked.
The description of c_2 is "The box looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_2 is "box".
Understand "box" as c_2.
The c_2 is in r_15.
The c_2 is locked.
The description of c_3 is "The trunk looks strong, and impossible to crack. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_3 is "trunk".
Understand "trunk" as c_3.
The c_3 is in r_17.
The c_3 is closed.
The description of c_4 is "The crate looks strong, and impossible to destroy. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_4 is "crate".
Understand "crate" as c_4.
The c_4 is in r_4.
The c_4 is open.
The description of k_2 is "The rectangular latchkey is cold to the touch".
The printed name of k_2 is "rectangular latchkey".
Understand "rectangular latchkey" as k_2.
Understand "rectangular" as k_2.
Understand "latchkey" as k_2.
The k_2 is in r_18.
The matching key of the d_3 is the k_2.
The description of k_3 is "The passkey looks useful".
The printed name of k_3 is "passkey".
Understand "passkey" as k_3.
The k_3 is in r_17.
The matching key of the d_1 is the k_3.
The description of s_0 is "The counter is an unstable piece of trash.".
The printed name of s_0 is "counter".
Understand "counter" as s_0.
The s_0 is in r_0.
The description of s_1 is "The pan is balanced.".
The printed name of s_1 is "pan".
Understand "pan" as s_1.
The s_1 is in r_16.
The description of s_10 is "The shelf is undependable.".
The printed name of s_10 is "shelf".
Understand "shelf" as s_10.
The s_10 is in r_19.
The description of s_11 is "The rack is stable.".
The printed name of s_11 is "rack".
Understand "rack" as s_11.
The s_11 is in r_4.
The description of s_12 is "The stand is stable.".
The printed name of s_12 is "stand".
Understand "stand" as s_12.
The s_12 is in r_4.
The description of s_13 is "The table is durable.".
The printed name of s_13 is "table".
Understand "table" as s_13.
The s_13 is in r_6.
The description of s_14 is "The mantelpiece is unstable.".
The printed name of s_14 is "mantelpiece".
Understand "mantelpiece" as s_14.
The s_14 is in r_8.
The description of s_2 is "The saucepan is durable.".
The printed name of s_2 is "saucepan".
Understand "saucepan" as s_2.
The s_2 is in r_1.
The description of s_3 is "The chair is unstable.".
The printed name of s_3 is "chair".
Understand "chair" as s_3.
The s_3 is in r_2.
The description of s_4 is "The bowl is balanced.".
The printed name of s_4 is "bowl".
Understand "bowl" as s_4.
The s_4 is in r_2.
The description of s_5 is "The workbench is shaky.".
The printed name of s_5 is "workbench".
Understand "workbench" as s_5.
The s_5 is in r_10.
The description of s_6 is "The desk is solidly built.".
The printed name of s_6 is "desk".
Understand "desk" as s_6.
The s_6 is in r_11.
The description of s_7 is "The bed is stable.".
The printed name of s_7 is "bed".
Understand "bed" as s_7.
The s_7 is in r_11.
The description of s_8 is "The bookshelf is balanced.".
The printed name of s_8 is "bookshelf".
Understand "bookshelf" as s_8.
The s_8 is in r_13.
The description of s_9 is "The mantle is shaky.".
The printed name of s_9 is "mantle".
Understand "mantle" as s_9.
The s_9 is in r_15.
The description of k_0 is "The latchkey is weighty.".
The printed name of k_0 is "latchkey".
Understand "latchkey" as k_0.
The k_0 is in the c_1.
The description of k_4 is "The key is heavier than it looks.".
The printed name of k_4 is "key".
Understand "key" as k_4.
The player carries the k_4.
The matching key of the d_2 is the k_4.
The description of k_1 is "The spherical latchkey looks useful".
The printed name of k_1 is "spherical latchkey".
Understand "spherical latchkey" as k_1.
Understand "spherical" as k_1.
Understand "latchkey" as k_1.
The matching key of the d_0 is the k_1.
The k_1 is on the s_0.
The description of f_0 is "You couldn't pay me to eat that standard thing.".
The printed name of f_0 is "burger".
Understand "burger" as f_0.
The f_0 is on the s_9.


The player is in r_16.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with "unlock gateway with key / open gateway / go north / take passkey / unlock hatch with passkey / open hatch / go east / take rectangular latchkey / go west / go south / unlock rectangular portal with rectangular latchkey / open rectangular portal / go east / take spherical latchkey from counter / unlock spherical door with spherical latchkey / open spherical door / go south / go east / go east / take burger from mantle"

Every turn:
	if quest0 completed is true:
		do nothing;
	else if The player carries the k_0:
		end the story; [Lost]
	else if The player is in r_15 and The s_9 is in r_15 and The player carries the f_0:
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

The objective part 0 is some text that varies. The objective part 0 is "I hope you're ready to go into rooms and interact with objects, because you've just entered TextWorld! Your task for today is to take the burger from the mantle.".

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

