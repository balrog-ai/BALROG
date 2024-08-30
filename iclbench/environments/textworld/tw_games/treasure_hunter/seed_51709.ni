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


The r_1 and the r_2 and the r_10 and the r_9 and the r_11 and the r_12 and the r_4 and the r_13 and the r_15 and the r_8 and the r_19 and the r_18 and the r_3 and the r_7 and the r_5 and the r_6 and the r_0 and the r_14 and the r_16 and the r_17 are rooms.

Understand "attic" as r_1.
The internal name of r_1 is "attic".
The printed name of r_1 is "-= Attic =-".
The attic part 0 is some text that varies. The attic part 0 is "This is going to sound unbelievable, but you've just entered an attic.

 You see [if c_0 is locked]a locked[else if c_0 is open]an opened[otherwise]a closed[end if]".
The attic part 1 is some text that varies. The attic part 1 is " normal looking crate nearby.[if c_0 is open and there is something in the c_0] The crate contains [a list of things in the c_0]. You shudder, but continue examining the room.[end if]".
The attic part 2 is some text that varies. The attic part 2 is "[if c_0 is open and the c_0 contains nothing] The crate is empty! What a waste of a day![end if]".
The attic part 3 is some text that varies. The attic part 3 is " You see a counter. Now why would someone leave that there? [if there is something on the s_0]You see [a list of things on the s_0] on the counter.[end if]".
The attic part 4 is some text that varies. The attic part 4 is "[if there is nothing on the s_0]However, the counter, like an empty counter, has nothing on it. Aw, here you were, all excited for there to be things on it![end if]".
The attic part 5 is some text that varies. The attic part 5 is "

 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The attic part 6 is some text that varies. The attic part 6 is " passageway leading south. There is an exit to the west. Don't worry, it is unblocked.".
The description of r_1 is "[attic part 0][attic part 1][attic part 2][attic part 3][attic part 4][attic part 5][attic part 6]".

The r_2 is mapped west of r_1.
south of r_1 and north of r_0 is a door called d_0.
Understand "kitchen" as r_2.
The internal name of r_2 is "kitchen".
The printed name of r_2 is "-= Kitchen =-".
The kitchen part 0 is some text that varies. The kitchen part 0 is "A well framed signboard tells you that you are now in the kitchen.

 You can make out [if c_1 is locked]a locked[else if c_1 is open]an opened[otherwise]a closed[end if]".
The kitchen part 1 is some text that varies. The kitchen part 1 is " case.[if c_1 is open and there is something in the c_1] The case contains [a list of things in the c_1].[end if]".
The kitchen part 2 is some text that varies. The kitchen part 2 is "[if c_1 is open and the c_1 contains nothing] The case is empty! This is the worst thing that could possibly happen, ever![end if]".
The kitchen part 3 is some text that varies. The kitchen part 3 is " You make out [if c_2 is locked]a locked[else if c_2 is open]an opened[otherwise]a closed[end if]".
The kitchen part 4 is some text that varies. The kitchen part 4 is " cabinet.[if c_2 is open and there is something in the c_2] The cabinet contains [a list of things in the c_2].[end if]".
The kitchen part 5 is some text that varies. The kitchen part 5 is "[if c_2 is open and the c_2 contains nothing] The cabinet is empty! This is the worst thing that could possibly happen, ever![end if]".
The kitchen part 6 is some text that varies. The kitchen part 6 is "

There is an exit to the east. Don't worry, it is unblocked.".
The description of r_2 is "[kitchen part 0][kitchen part 1][kitchen part 2][kitchen part 3][kitchen part 4][kitchen part 5][kitchen part 6]".

The r_1 is mapped east of r_2.
Understand "shower" as r_10.
The internal name of r_10 is "shower".
The printed name of r_10 is "-= Shower =-".
The shower part 0 is some text that varies. The shower part 0 is "You have entered a shower. Not the shower you'd expect. No, this is a shower. You decide to just list off a complete list of everything you see in the room, because hey, why not?



You don't like doors? Why not try going east, that entranceway is unblocked. There is an unguarded exit to the north. You don't like doors? Why not try going west, that entranceway is unguarded.".
The description of r_10 is "[shower part 0]".

The r_9 is mapped west of r_10.
The r_12 is mapped north of r_10.
The r_11 is mapped east of r_10.
Understand "study" as r_9.
The internal name of r_9 is "study".
The printed name of r_9 is "-= Study =-".
The study part 0 is some text that varies. The study part 0 is "This might come as a shock to you, but you've just entered a study.

 You can make out [if c_3 is locked]a locked[else if c_3 is open]an opened[otherwise]a closed[end if]".
The study part 1 is some text that varies. The study part 1 is " coffer.[if c_3 is open and there is something in the c_3] The coffer contains [a list of things in the c_3].[end if]".
The study part 2 is some text that varies. The study part 2 is "[if c_3 is open and the c_3 contains nothing] What a letdown! The coffer is empty![end if]".
The study part 3 is some text that varies. The study part 3 is " If you haven't noticed it already, there seems to be something there by the wall, it's a suitcase. There's something strange about this being here, but you can't put your finger on it.[if c_4 is open and there is something in the c_4] The suitcase contains [a list of things in the c_4].[end if]".
The study part 4 is some text that varies. The study part 4 is "[if c_4 is open and the c_4 contains nothing] The suitcase is empty! This is the worst thing that could possibly happen, ever![end if]".
The study part 5 is some text that varies. The study part 5 is " You lean against the wall, inadvertently pressing a secret button. The wall opens up to reveal a mantelpiece. [if there is something on the s_1]You see [a list of things on the s_1] on the mantelpiece.[end if]".
The study part 6 is some text that varies. The study part 6 is "[if there is nothing on the s_1]The mantelpiece appears to be empty.[end if]".
The study part 7 is some text that varies. The study part 7 is " You can make out a desk. The desk is typical.[if there is something on the s_2] On the desk you make out [a list of things on the s_2].[end if]".
The study part 8 is some text that varies. The study part 8 is "[if there is nothing on the s_2] But the thing is empty, unfortunately.[end if]".
The study part 9 is some text that varies. The study part 9 is "

You need an unguarded exit? You should try going east. There is an exit to the north. Don't worry, it is unblocked. There is an unblocked exit to the west.".
The description of r_9 is "[study part 0][study part 1][study part 2][study part 3][study part 4][study part 5][study part 6][study part 7][study part 8][study part 9]".

The r_5 is mapped west of r_9.
The r_4 is mapped north of r_9.
The r_10 is mapped east of r_9.
Understand "closet" as r_11.
The internal name of r_11 is "closet".
The printed name of r_11 is "-= Closet =-".
The closet part 0 is some text that varies. The closet part 0 is "You are in a closet. An usual kind of place. You try to gain information on your surroundings by using a technique you call 'looking.'



There is an exit to the north. Don't worry, it is unguarded. There is an exit to the west. Don't worry, it is unblocked.".
The description of r_11 is "[closet part 0]".

The r_10 is mapped west of r_11.
The r_13 is mapped north of r_11.
Understand "cubicle" as r_12.
The internal name of r_12 is "cubicle".
The printed name of r_12 is "-= Cubicle =-".
The cubicle part 0 is some text that varies. The cubicle part 0 is "You find yourself in a cubicle. A normal kind of place. You decide to just list off a complete list of everything you see in the room, because hey, why not?



You don't like doors? Why not try going east, that entranceway is unguarded. You need an unguarded exit? You should try going north. There is an exit to the south. Don't worry, it is unguarded. You don't like doors? Why not try going west, that entranceway is unblocked.".
The description of r_12 is "[cubicle part 0]".

The r_4 is mapped west of r_12.
The r_10 is mapped south of r_12.
The r_15 is mapped north of r_12.
The r_13 is mapped east of r_12.
Understand "launderette" as r_4.
The internal name of r_4 is "launderette".
The printed name of r_4 is "-= Launderette =-".
The launderette part 0 is some text that varies. The launderette part 0 is "Well how about that, you are in the place we're calling the launderette.

 You see a table. [if there is something on the s_3]You see [a list of things on the s_3] on the table, so there's that.[end if]".
The launderette part 1 is some text that varies. The launderette part 1 is "[if there is nothing on the s_3]Looks like someone's already been here and taken everything off it, though. Hm. Oh well[end if]".
The launderette part 2 is some text that varies. The launderette part 2 is " You can see a rack. The rack is normal.[if there is something on the s_4] On the rack you see [a list of things on the s_4].[end if]".
The launderette part 3 is some text that varies. The launderette part 3 is "[if there is nothing on the s_4] The rack appears to be empty.[end if]".
The launderette part 4 is some text that varies. The launderette part 4 is "

There is an unguarded exit to the east. You need an unguarded exit? You should try going north. You need an unblocked exit? You should try going south. There is an exit to the west. Don't worry, it is unblocked.".
The description of r_4 is "[launderette part 0][launderette part 1][launderette part 2][launderette part 3][launderette part 4]".

The r_3 is mapped west of r_4.
The r_9 is mapped south of r_4.
The r_8 is mapped north of r_4.
The r_12 is mapped east of r_4.
Understand "lounge" as r_13.
The internal name of r_13 is "lounge".
The printed name of r_13 is "-= Lounge =-".
The lounge part 0 is some text that varies. The lounge part 0 is "You've just walked into a lounge.



There is an unblocked exit to the north. You don't like doors? Why not try going south, that entranceway is unguarded. You need an unguarded exit? You should try going west.".
The description of r_13 is "[lounge part 0]".

The r_12 is mapped west of r_13.
The r_11 is mapped south of r_13.
The r_14 is mapped north of r_13.
Understand "kitchenette" as r_15.
The internal name of r_15 is "kitchenette".
The printed name of r_15 is "-= Kitchenette =-".
The kitchenette part 0 is some text that varies. The kitchenette part 0 is "You've just sauntered into a kitchenette. I guess you better just go and list everything you see here.



 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The kitchenette part 1 is some text that varies. The kitchenette part 1 is " hatch leading west. There is an unblocked exit to the north. You need an unguarded exit? You should try going south.".
The description of r_15 is "[kitchenette part 0][kitchenette part 1]".

west of r_15 and east of r_8 is a door called d_1.
The r_12 is mapped south of r_15.
The r_16 is mapped north of r_15.
Understand "bathroom" as r_8.
The internal name of r_8 is "bathroom".
The printed name of r_8 is "-= Bathroom =-".
The bathroom part 0 is some text that varies. The bathroom part 0 is "Well, here we are in a bathroom.

 You make out a drawer.[if c_5 is open and there is something in the c_5] The drawer contains [a list of things in the c_5].[end if]".
The bathroom part 1 is some text that varies. The bathroom part 1 is "[if c_5 is open and the c_5 contains nothing] The drawer is empty! What a waste of a day![end if]".
The bathroom part 2 is some text that varies. The bathroom part 2 is "

 There is [if d_3 is open]an open[otherwise]a closed[end if]".
The bathroom part 3 is some text that varies. The bathroom part 3 is " type 0 gateway leading north. There is [if d_1 is open]an open[otherwise]a closed[end if]".
The bathroom part 4 is some text that varies. The bathroom part 4 is " hatch leading east. There is an unblocked exit to the south.".
The description of r_8 is "[bathroom part 0][bathroom part 1][bathroom part 2][bathroom part 3][bathroom part 4]".

The r_4 is mapped south of r_8.
north of r_8 and south of r_17 is a door called d_3.
east of r_8 and west of r_15 is a door called d_1.
Understand "pantry" as r_19.
The internal name of r_19 is "pantry".
The printed name of r_19 is "-= Pantry =-".
The pantry part 0 is some text that varies. The pantry part 0 is "Look around you. Take it all in. It's not every day someone gets to be in a pantry.

 You scan the room for a shelf, and you find a shelf. Make a note of this, you might have to put stuff on or in it later on. The shelf is normal.[if there is something on the s_5] On the shelf you see [a list of things on the s_5].[end if]".
The pantry part 1 is some text that varies. The pantry part 1 is "[if there is nothing on the s_5] But oh no! there's nothing on this piece of junk.[end if]".
The pantry part 2 is some text that varies. The pantry part 2 is "

You don't like doors? Why not try going west, that entranceway is unblocked.".
The description of r_19 is "[pantry part 0][pantry part 1][pantry part 2]".

The r_18 is mapped west of r_19.
Understand "canteen" as r_18.
The internal name of r_18 is "canteen".
The printed name of r_18 is "-= Canteen =-".
The canteen part 0 is some text that varies. The canteen part 0 is "This is going to sound unbelievable, but you've just entered a canteen. I guess you better just go and list everything you see here.

 You make out a box.[if c_6 is open and there is something in the c_6] The box contains [a list of things in the c_6].[end if]".
The canteen part 1 is some text that varies. The canteen part 1 is "[if c_6 is open and the c_6 contains nothing] The box is empty! This is the worst thing that could possibly happen, ever![end if]".
The canteen part 2 is some text that varies. The canteen part 2 is "

 There is [if d_2 is open]an open[otherwise]a closed[end if]".
The canteen part 3 is some text that varies. The canteen part 3 is " portal leading south. There is an unblocked exit to the east.".
The description of r_18 is "[canteen part 0][canteen part 1][canteen part 2][canteen part 3]".

south of r_18 and north of r_17 is a door called d_2.
The r_19 is mapped east of r_18.
Understand "dish-pit" as r_3.
The internal name of r_3 is "dish-pit".
The printed name of r_3 is "-= Dish-Pit =-".
The dish-pit part 0 is some text that varies. The dish-pit part 0 is "You're now in the dish-pit.

 You make out a board! [if there is something on the s_6]On the board you make out [a list of things on the s_6].[end if]".
The dish-pit part 1 is some text that varies. The dish-pit part 1 is "[if there is nothing on the s_6]But the thing is empty. Aw, here you were, all excited for there to be things on it![end if]".
The dish-pit part 2 is some text that varies. The dish-pit part 2 is "

 There is [if d_4 is open]an open[otherwise]a closed[end if]".
The dish-pit part 3 is some text that varies. The dish-pit part 3 is " door leading north. You don't like doors? Why not try going east, that entranceway is unblocked. You don't like doors? Why not try going south, that entranceway is unblocked. There is an exit to the west. Don't worry, it is unblocked.".
The description of r_3 is "[dish-pit part 0][dish-pit part 1][dish-pit part 2][dish-pit part 3]".

The r_7 is mapped west of r_3.
The r_5 is mapped south of r_3.
north of r_3 and south of r_0 is a door called d_4.
The r_4 is mapped east of r_3.
Understand "playroom" as r_7.
The internal name of r_7 is "playroom".
The printed name of r_7 is "-= Playroom =-".
The playroom part 0 is some text that varies. The playroom part 0 is "You find yourself in a playroom. An ordinary kind of place.

 You can make out a bench. The bench is usual.[if there is something on the s_7] On the bench you can see [a list of things on the s_7]. It doesn't get more TextWorld than this![end if]".
The playroom part 1 is some text that varies. The playroom part 1 is "[if there is nothing on the s_7] Looks like someone's already been here and taken everything off it, though.[end if]".
The playroom part 2 is some text that varies. The playroom part 2 is "

You don't like doors? Why not try going east, that entranceway is unblocked. You need an unblocked exit? You should try going south.".
The description of r_7 is "[playroom part 0][playroom part 1][playroom part 2]".

The r_6 is mapped south of r_7.
The r_3 is mapped east of r_7.
Understand "workshop" as r_5.
The internal name of r_5 is "workshop".
The printed name of r_5 is "-= Workshop =-".
The workshop part 0 is some text that varies. The workshop part 0 is "Look at that sign! What does it say? It says Welcome to the workshop? Well that's cool.

 You can see [if c_7 is locked]a locked[else if c_7 is open]an opened[otherwise]a closed[end if]".
The workshop part 1 is some text that varies. The workshop part 1 is " trunk.[if c_7 is open and there is something in the c_7] The trunk contains [a list of things in the c_7].[end if]".
The workshop part 2 is some text that varies. The workshop part 2 is "[if c_7 is open and the c_7 contains nothing] What a letdown! The trunk is empty![end if]".
The workshop part 3 is some text that varies. The workshop part 3 is " You scan the room for a bookshelf, and you find a bookshelf. The bookshelf is usual.[if there is something on the s_8] On the bookshelf you see [a list of things on the s_8].[end if]".
The workshop part 4 is some text that varies. The workshop part 4 is "[if there is nothing on the s_8] But the thing is empty.[end if]".
The workshop part 5 is some text that varies. The workshop part 5 is "

You don't like doors? Why not try going east, that entranceway is unblocked. You don't like doors? Why not try going north, that entranceway is unguarded. There is an unguarded exit to the west.".
The description of r_5 is "[workshop part 0][workshop part 1][workshop part 2][workshop part 3][workshop part 4][workshop part 5]".

The r_6 is mapped west of r_5.
The r_3 is mapped north of r_5.
The r_9 is mapped east of r_5.
Understand "bedroom" as r_6.
The internal name of r_6 is "bedroom".
The printed name of r_6 is "-= Bedroom =-".
The bedroom part 0 is some text that varies. The bedroom part 0 is "You are in a bedroom. You decide to just list off a complete list of everything you see in the room, because hey, why not?



You need an unguarded exit? You should try going east. You need an unblocked exit? You should try going north.".
The description of r_6 is "[bedroom part 0]".

The r_7 is mapped north of r_6.
The r_5 is mapped east of r_6.
Understand "basement" as r_0.
The internal name of r_0 is "basement".
The printed name of r_0 is "-= Basement =-".
The basement part 0 is some text that varies. The basement part 0 is "You are in a basement. A typical one. You begin to take stock of what's in the room.



 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The basement part 1 is some text that varies. The basement part 1 is " passageway leading north. There is [if d_4 is open]an open[otherwise]a closed[end if]".
The basement part 2 is some text that varies. The basement part 2 is " door leading south.".
The description of r_0 is "[basement part 0][basement part 1][basement part 2]".

south of r_0 and north of r_3 is a door called d_4.
north of r_0 and south of r_1 is a door called d_0.
Understand "laundromat" as r_14.
The internal name of r_14 is "laundromat".
The printed name of r_14 is "-= Laundromat =-".
The laundromat part 0 is some text that varies. The laundromat part 0 is "Fancy seeing you here. Here, by the way, being the laundromat.

 You see a dusty bench. [if there is something on the s_9]On the dusty bench you can make out [a list of things on the s_9].[end if]".
The laundromat part 1 is some text that varies. The laundromat part 1 is "[if there is nothing on the s_9]But the thing hasn't got anything on it. It would have been so cool if there was stuff on the bench! oh well.[end if]".
The laundromat part 2 is some text that varies. The laundromat part 2 is "

There is an unguarded exit to the south.".
The description of r_14 is "[laundromat part 0][laundromat part 1][laundromat part 2]".

The r_13 is mapped south of r_14.
Understand "parlor" as r_16.
The internal name of r_16 is "parlor".
The printed name of r_16 is "-= Parlor =-".
The parlor part 0 is some text that varies. The parlor part 0 is "You are in a parlor. A standard kind of place.

 You make out a bed stand. [if there is something on the s_10]You see [a list of things on the s_10] on the bed stand. There's something strange about this thing being here, but you don't have time to worry about that now.[end if]".
The parlor part 1 is some text that varies. The parlor part 1 is "[if there is nothing on the s_10]But the thing is empty, unfortunately.[end if]".
The parlor part 2 is some text that varies. The parlor part 2 is "

There is an unblocked exit to the south.".
The description of r_16 is "[parlor part 0][parlor part 1][parlor part 2]".

The r_15 is mapped south of r_16.
Understand "laundry place" as r_17.
The internal name of r_17 is "laundry place".
The printed name of r_17 is "-= Laundry Place =-".
The laundry place part 0 is some text that varies. The laundry place part 0 is "You are in a laundry place. An ordinary one.

 You see a Microsoft box. There's something strange about this being here, but you can't put your finger on it.[if c_8 is open and there is something in the c_8] The Microsoft box contains [a list of things in the c_8]. There's something strange about this being here, but you can't put your finger on it.[end if]".
The laundry place part 1 is some text that varies. The laundry place part 1 is "[if c_8 is open and the c_8 contains nothing] The Microsoft box is empty! What a waste of a day![end if]".
The laundry place part 2 is some text that varies. The laundry place part 2 is "

 There is [if d_2 is open]an open[otherwise]a closed[end if]".
The laundry place part 3 is some text that varies. The laundry place part 3 is " portal leading north. There is [if d_3 is open]an open[otherwise]a closed[end if]".
The laundry place part 4 is some text that varies. The laundry place part 4 is " type 0 gateway leading south.".
The description of r_17 is "[laundry place part 0][laundry place part 1][laundry place part 2][laundry place part 3][laundry place part 4]".

south of r_17 and north of r_8 is a door called d_3.
north of r_17 and south of r_18 is a door called d_2.

The c_0 and the c_1 and the c_2 and the c_3 and the c_4 and the c_5 and the c_6 and the c_7 and the c_8 are containers.
The c_0 and the c_1 and the c_2 and the c_3 and the c_4 and the c_5 and the c_6 and the c_7 and the c_8 are privately-named.
The d_0 and the d_4 and the d_1 and the d_2 and the d_3 are doors.
The d_0 and the d_4 and the d_1 and the d_2 and the d_3 are privately-named.
The f_0 are foods.
The f_0 are privately-named.
The k_2 and the k_0 and the k_1 and the k_3 are keys.
The k_2 and the k_0 and the k_1 and the k_3 are privately-named.
The o_0 are object-likes.
The o_0 are privately-named.
The r_1 and the r_2 and the r_10 and the r_9 and the r_11 and the r_12 and the r_4 and the r_13 and the r_15 and the r_8 and the r_19 and the r_18 and the r_3 and the r_7 and the r_5 and the r_6 and the r_0 and the r_14 and the r_16 and the r_17 are rooms.
The r_1 and the r_2 and the r_10 and the r_9 and the r_11 and the r_12 and the r_4 and the r_13 and the r_15 and the r_8 and the r_19 and the r_18 and the r_3 and the r_7 and the r_5 and the r_6 and the r_0 and the r_14 and the r_16 and the r_17 are privately-named.
The s_0 and the s_1 and the s_10 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 and the s_9 are supporters.
The s_0 and the s_1 and the s_10 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 and the s_9 are privately-named.

The description of d_0 is "The passageway looks sturdy. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_0 is "passageway".
Understand "passageway" as d_0.
The d_0 is open.
The description of d_4 is "The door looks stuffy. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_4 is "door".
Understand "door" as d_4.
The d_4 is open.
The description of d_1 is "The hatch looks commanding. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_1 is "hatch".
Understand "hatch" as d_1.
The d_1 is open.
The description of d_2 is "it is what it is, a portal [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_2 is "portal".
Understand "portal" as d_2.
The d_2 is locked.
The description of d_3 is "it's a solid type 0 gateway [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_3 is "type 0 gateway".
Understand "type 0 gateway" as d_3.
Understand "type" as d_3.
Understand "0" as d_3.
Understand "gateway" as d_3.
The d_3 is locked.
The description of c_0 is "The crate looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_0 is "crate".
Understand "crate" as c_0.
The c_0 is in r_1.
The c_0 is locked.
The description of c_1 is "The case looks strong, and impossible to crack. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_1 is "case".
Understand "case" as c_1.
The c_1 is in r_2.
The c_1 is closed.
The description of c_2 is "The cabinet looks strong, and impossible to crack. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_2 is "cabinet".
Understand "cabinet" as c_2.
The c_2 is in r_2.
The c_2 is open.
The description of c_3 is "The coffer looks strong, and impossible to crack. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_3 is "coffer".
Understand "coffer" as c_3.
The c_3 is in r_9.
The c_3 is locked.
The description of c_4 is "The suitcase looks strong, and impossible to crack. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_4 is "suitcase".
Understand "suitcase" as c_4.
The c_4 is in r_9.
The c_4 is closed.
The description of c_5 is "The drawer looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_5 is "drawer".
Understand "drawer" as c_5.
The c_5 is in r_8.
The c_5 is closed.
The description of c_6 is "The box looks strong, and impossible to destroy. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_6 is "box".
Understand "box" as c_6.
The c_6 is in r_18.
The c_6 is locked.
The description of c_7 is "The trunk looks strong, and impossible to crack. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_7 is "trunk".
Understand "trunk" as c_7.
The c_7 is in r_5.
The c_7 is closed.
The description of c_8 is "The Microsoft box looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_8 is "Microsoft box".
Understand "Microsoft box" as c_8.
Understand "Microsoft" as c_8.
Understand "box" as c_8.
The c_8 is in r_17.
The c_8 is locked.
The description of f_0 is "You couldn't pay me to eat that ordinary thing.".
The printed name of f_0 is "apple".
Understand "apple" as f_0.
The f_0 is in r_2.
The description of k_2 is "The metal of the Microsoft key is hammered.".
The printed name of k_2 is "Microsoft key".
Understand "Microsoft key" as k_2.
Understand "Microsoft" as k_2.
Understand "key" as k_2.
The k_2 is in r_18.
The matching key of the c_8 is the k_2.
The description of o_0 is "The sponge is expensive looking.".
The printed name of o_0 is "sponge".
Understand "sponge" as o_0.
The o_0 is in r_10.
The description of s_0 is "The counter is reliable.".
The printed name of s_0 is "counter".
Understand "counter" as s_0.
The s_0 is in r_1.
The description of s_1 is "The mantelpiece is undependable.".
The printed name of s_1 is "mantelpiece".
Understand "mantelpiece" as s_1.
The s_1 is in r_9.
The description of s_10 is "The bed stand is wobbly.".
The printed name of s_10 is "bed stand".
Understand "bed stand" as s_10.
Understand "bed" as s_10.
Understand "stand" as s_10.
The s_10 is in r_16.
The description of s_2 is "The desk is stable.".
The printed name of s_2 is "desk".
Understand "desk" as s_2.
The s_2 is in r_9.
The description of s_3 is "The table is stable.".
The printed name of s_3 is "table".
Understand "table" as s_3.
The s_3 is in r_4.
The description of s_4 is "The rack is shaky.".
The printed name of s_4 is "rack".
Understand "rack" as s_4.
The s_4 is in r_4.
The description of s_5 is "The shelf is wobbly.".
The printed name of s_5 is "shelf".
Understand "shelf" as s_5.
The s_5 is in r_19.
The description of s_6 is "The board is unstable.".
The printed name of s_6 is "board".
Understand "board" as s_6.
The s_6 is in r_3.
The description of s_7 is "The bench is stable.".
The printed name of s_7 is "bench".
Understand "bench" as s_7.
The s_7 is in r_7.
The description of s_8 is "The bookshelf is durable.".
The printed name of s_8 is "bookshelf".
Understand "bookshelf" as s_8.
The s_8 is in r_5.
The description of s_9 is "The dusty bench is solidly built.".
The printed name of s_9 is "dusty bench".
Understand "dusty bench" as s_9.
Understand "dusty" as s_9.
Understand "bench" as s_9.
The s_9 is in r_14.
The description of k_0 is "The metal of the type 0 keycard is hammered.".
The printed name of k_0 is "type 0 keycard".
Understand "type 0 keycard" as k_0.
Understand "type" as k_0.
Understand "0" as k_0.
Understand "keycard" as k_0.
The k_0 is in the c_6.
The matching key of the d_3 is the k_0.
The description of k_1 is "The latchkey is surprisingly heavy.".
The printed name of k_1 is "latchkey".
Understand "latchkey" as k_1.
The k_1 is in the c_8.
The matching key of the c_6 is the k_1.
The description of k_3 is "The metal of the key is hammered.".
The printed name of k_3 is "key".
Understand "key" as k_3.
The player carries the k_3.
The matching key of the d_2 is the k_3.


The player is in r_17.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with "unlock portal with key / open portal / go north / take Microsoft key / go south / unlock Microsoft box with Microsoft key / open Microsoft box / take latchkey from Microsoft box / go north / unlock box with latchkey / open box / take type 0 keycard from box / go south / unlock type 0 gateway with type 0 keycard / open type 0 gateway / go south / go south / go south / go east / take sponge"

Every turn:
	if quest0 completed is true:
		do nothing;
	else if The player carries the f_0:
		end the story; [Lost]
	else if The player is in r_10 and The player carries the o_0:
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

The objective part 0 is some text that varies. The objective part 0 is "Your objective is to retrieve the sponge from the floor of the shower.".

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

