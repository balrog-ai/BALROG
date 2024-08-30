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


The r_12 and the r_11 and the r_13 and the r_8 and the r_15 and the r_19 and the r_16 and the r_17 and the r_2 and the r_1 and the r_3 and the r_0 and the r_4 and the r_5 and the r_6 and the r_9 and the r_10 and the r_14 and the r_18 and the r_7 are rooms.

Understand "kitchen" as r_12.
The internal name of r_12 is "kitchen".
The printed name of r_12 is "-= Kitchen =-".
The kitchen part 0 is some text that varies. The kitchen part 0 is "You're now in a kitchen. Let's see what's in here.



You don't like doors? Why not try going south, that entranceway is unguarded. You need an unguarded exit? You should try going west.".
The description of r_12 is "[kitchen part 0]".

The r_11 is mapped west of r_12.
The r_8 is mapped south of r_12.
Understand "laundry place" as r_11.
The internal name of r_11 is "laundry place".
The printed name of r_11 is "-= Laundry Place =-".
The laundry place part 0 is some text that varies. The laundry place part 0 is "You are in a laundry place. It seems to be pretty typical here. You start to take note of what's in the room.

 Look over there! a dresser. Now that's what I call TextWorld![if c_0 is open and there is something in the c_0] The dresser contains [a list of things in the c_0].[end if]".
The laundry place part 1 is some text that varies. The laundry place part 1 is "[if c_0 is open and the c_0 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The laundry place part 2 is some text that varies. The laundry place part 2 is " What's that over there? It looks like it's a basket. You can't wait to tell the folks at home about this![if c_1 is open and there is something in the c_1] The basket contains [a list of things in the c_1].[end if]".
The laundry place part 3 is some text that varies. The laundry place part 3 is "[if c_1 is open and the c_1 contains nothing] The basket is empty! What a waste of a day![end if]".
The laundry place part 4 is some text that varies. The laundry place part 4 is "

There is an unblocked exit to the east. You need an unblocked exit? You should try going south.".
The description of r_11 is "[laundry place part 0][laundry place part 1][laundry place part 2][laundry place part 3][laundry place part 4]".

The r_9 is mapped south of r_11.
The r_12 is mapped east of r_11.
Understand "closet" as r_13.
The internal name of r_13 is "closet".
The printed name of r_13 is "-= Closet =-".
The closet part 0 is some text that varies. The closet part 0 is "You are in a closet.

 You smell an intriguing smell, and follow it to a suitcase.[if c_2 is open and there is something in the c_2] The suitcase contains [a list of things in the c_2].[end if]".
The closet part 1 is some text that varies. The closet part 1 is "[if c_2 is open and the c_2 contains nothing] The suitcase is empty! This is the worst thing that could possibly happen, ever![end if]".
The closet part 2 is some text that varies. The closet part 2 is "

 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The closet part 3 is some text that varies. The closet part 3 is " gateway leading west. There is an unguarded exit to the south.".
The description of r_13 is "[closet part 0][closet part 1][closet part 2][closet part 3]".

west of r_13 and east of r_8 is a door called d_1.
The r_14 is mapped south of r_13.
Understand "steam room" as r_8.
The internal name of r_8 is "steam room".
The printed name of r_8 is "-= Steam Room =-".
The steam room part 0 is some text that varies. The steam room part 0 is "You arrive in a steam room. An ordinary kind of place.

 You can make out a rack. [if there is something on the s_0]On the rack you can make out [a list of things on the s_0].[end if]".
The steam room part 1 is some text that varies. The steam room part 1 is "[if there is nothing on the s_0]However, the rack, like an empty rack, has nothing on it. Oh! Why couldn't there just be stuff on it?[end if]".
The steam room part 2 is some text that varies. The steam room part 2 is "

 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The steam room part 3 is some text that varies. The steam room part 3 is " gateway leading east. There is [if d_2 is open]an open[otherwise]a closed[end if]".
The steam room part 4 is some text that varies. The steam room part 4 is " wooden gate leading south. There is [if d_0 is open]an open[otherwise]a closed[end if]".
The steam room part 5 is some text that varies. The steam room part 5 is " stone passageway leading west. You need an unguarded exit? You should try going north.".
The description of r_8 is "[steam room part 0][steam room part 1][steam room part 2][steam room part 3][steam room part 4][steam room part 5]".

west of r_8 and east of r_9 is a door called d_0.
south of r_8 and north of r_7 is a door called d_2.
The r_12 is mapped north of r_8.
east of r_8 and west of r_13 is a door called d_1.
Understand "pantry" as r_15.
The internal name of r_15 is "pantry".
The printed name of r_15 is "-= Pantry =-".
The pantry part 0 is some text that varies. The pantry part 0 is "You arrive in a pantry. An ordinary one.



 There is [if d_8 is open]an open[otherwise]a closed[end if]".
The pantry part 1 is some text that varies. The pantry part 1 is " TextWorld limited edition door leading east. There is [if d_5 is open]an open[otherwise]a closed[end if]".
The pantry part 2 is some text that varies. The pantry part 2 is " passageway leading north. There is an exit to the west. Don't worry, it is unblocked.".
The description of r_15 is "[pantry part 0][pantry part 1][pantry part 2]".

The r_19 is mapped west of r_15.
north of r_15 and south of r_5 is a door called d_5.
east of r_15 and west of r_16 is a door called d_8.
Understand "bedroom" as r_19.
The internal name of r_19 is "bedroom".
The printed name of r_19 is "-= Bedroom =-".
The bedroom part 0 is some text that varies. The bedroom part 0 is "Look at that board! What does it say? It says Welcome to the bedroom? Well that's cool.

 You see [if c_3 is locked]a locked[else if c_3 is open]an opened[otherwise]a closed[end if]".
The bedroom part 1 is some text that varies. The bedroom part 1 is " coffer.[if c_3 is open and there is something in the c_3] The coffer contains [a list of things in the c_3].[end if]".
The bedroom part 2 is some text that varies. The bedroom part 2 is "[if c_3 is open and the c_3 contains nothing] The coffer is empty! This is the worst thing that could possibly happen, ever![end if]".
The bedroom part 3 is some text that varies. The bedroom part 3 is "

You need an unblocked exit? You should try going east.".
The description of r_19 is "[bedroom part 0][bedroom part 1][bedroom part 2][bedroom part 3]".

The r_15 is mapped east of r_19.
Understand "workshop" as r_16.
The internal name of r_16 is "workshop".
The printed name of r_16 is "-= Workshop =-".
The workshop part 0 is some text that varies. The workshop part 0 is "You've just shown up in a workshop.

 You can make out [if c_4 is locked]a locked[else if c_4 is open]an opened[otherwise]a closed[end if]".
The workshop part 1 is some text that varies. The workshop part 1 is " bureau.[if c_4 is open and there is something in the c_4] The bureau contains [a list of things in the c_4].[end if]".
The workshop part 2 is some text that varies. The workshop part 2 is "[if c_4 is open and the c_4 contains nothing] What a letdown! The bureau is empty![end if]".
The workshop part 3 is some text that varies. The workshop part 3 is " You can see a stand. The stand is standard.[if there is something on the s_1] On the stand you make out [a list of things on the s_1]. Suddenly, you bump your head on the ceiling, but it's not such a bad bump that it's going to prevent you from looking at objects and even things.[end if]".
The workshop part 4 is some text that varies. The workshop part 4 is "[if there is nothing on the s_1] The stand appears to be empty.[end if]".
The workshop part 5 is some text that varies. The workshop part 5 is "

 There is [if d_7 is open]an open[otherwise]a closed[end if]".
The workshop part 6 is some text that varies. The workshop part 6 is " Microsoft gateway leading east. There is [if d_8 is open]an open[otherwise]a closed[end if]".
The workshop part 7 is some text that varies. The workshop part 7 is " TextWorld limited edition door leading west.".
The description of r_16 is "[workshop part 0][workshop part 1][workshop part 2][workshop part 3][workshop part 4][workshop part 5][workshop part 6][workshop part 7]".

west of r_16 and east of r_15 is a door called d_8.
east of r_16 and west of r_17 is a door called d_7.
Understand "study" as r_17.
The internal name of r_17 is "study".
The printed name of r_17 is "-= Study =-".
The study part 0 is some text that varies. The study part 0 is "You're now in the study.

 You can make out [if c_5 is locked]a locked[else if c_5 is open]an opened[otherwise]a closed[end if]".
The study part 1 is some text that varies. The study part 1 is " normal looking chest here.[if c_5 is open and there is something in the c_5] The chest contains [a list of things in the c_5].[end if]".
The study part 2 is some text that varies. The study part 2 is "[if c_5 is open and the c_5 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The study part 3 is some text that varies. The study part 3 is " You hear a noise behind you and spin around, but you can't see anything other than a bookshelf. The bookshelf is typical.[if there is something on the s_2] On the bookshelf you can see [a list of things on the s_2].[end if]".
The study part 4 is some text that varies. The study part 4 is "[if there is nothing on the s_2] But oh no! there's nothing on this piece of garbage. Aw, here you were, all excited for there to be things on it![end if]".
The study part 5 is some text that varies. The study part 5 is " You lean against the wall, inadvertently pressing a secret button. The wall opens up to reveal a mantle. The mantle is standard.[if there is something on the s_3] On the mantle you can see [a list of things on the s_3].[end if]".
The study part 6 is some text that varies. The study part 6 is "[if there is nothing on the s_3] Unfortunately, there isn't a thing on it. Hm. Oh well[end if]".
The study part 7 is some text that varies. The study part 7 is "

 There is [if d_7 is open]an open[otherwise]a closed[end if]".
The study part 8 is some text that varies. The study part 8 is " Microsoft gateway leading west. There is an exit to the north. Don't worry, it is unguarded.".
The description of r_17 is "[study part 0][study part 1][study part 2][study part 3][study part 4][study part 5][study part 6][study part 7][study part 8]".

west of r_17 and east of r_16 is a door called d_7.
The r_18 is mapped north of r_17.
Understand "office" as r_2.
The internal name of r_2 is "office".
The printed name of r_2 is "-= Office =-".
The office part 0 is some text that varies. The office part 0 is "You're now in the office.

 You make out a chair. Why don't you take a picture of it, it'll last longer! The chair is normal.[if there is something on the s_4] On the chair you make out [a list of things on the s_4].[end if]".
The office part 1 is some text that varies. The office part 1 is "[if there is nothing on the s_4] But the thing is empty, unfortunately. Hm. Oh well[end if]".
The office part 2 is some text that varies. The office part 2 is " You can see a desk. [if there is something on the s_5]You see [a list of things on the s_5] on the desk.[end if]".
The office part 3 is some text that varies. The office part 3 is "[if there is nothing on the s_5]However, the desk, like an empty desk, has nothing on it.[end if]".
The office part 4 is some text that varies. The office part 4 is "

There is an exit to the north. Don't worry, it is unguarded. There is an exit to the west. Don't worry, it is unguarded.".
The description of r_2 is "[office part 0][office part 1][office part 2][office part 3][office part 4]".

The r_1 is mapped west of r_2.
The r_3 is mapped north of r_2.
Understand "garage" as r_1.
The internal name of r_1 is "garage".
The printed name of r_1 is "-= Garage =-".
The garage part 0 is some text that varies. The garage part 0 is "You've just walked into a garage. You decide to just list off a complete list of everything you see in the room, because hey, why not?



 There is [if d_9 is open]an open[otherwise]a closed[end if]".
The garage part 1 is some text that varies. The garage part 1 is " hatch leading south. You need an unguarded exit? You should try going east. There is an exit to the north. Don't worry, it is unblocked.".
The description of r_1 is "[garage part 0][garage part 1]".

south of r_1 and north of r_4 is a door called d_9.
The r_0 is mapped north of r_1.
The r_2 is mapped east of r_1.
Understand "cookery" as r_3.
The internal name of r_3 is "cookery".
The printed name of r_3 is "-= Cookery =-".
The cookery part 0 is some text that varies. The cookery part 0 is "You arrive in a cookery. A typical kind of place. You start to take note of what's in the room.



There is an exit to the south. Don't worry, it is unguarded. There is an exit to the west. Don't worry, it is unblocked.".
The description of r_3 is "[cookery part 0]".

The r_0 is mapped west of r_3.
The r_2 is mapped south of r_3.
Understand "dish-pit" as r_0.
The internal name of r_0 is "dish-pit".
The printed name of r_0 is "-= Dish-Pit =-".
The dish-pit part 0 is some text that varies. The dish-pit part 0 is "You are in a dish-pit. An ordinary one. You begin to take stock of what's in the room.

 Look out! It's a- oh, never mind, it's just a board. The board is ordinary.[if there is something on the s_6] On the board you can see [a list of things on the s_6].[end if]".
The dish-pit part 1 is some text that varies. The dish-pit part 1 is "[if there is nothing on the s_6] But the thing is empty. Hm. Oh well[end if]".
The dish-pit part 2 is some text that varies. The dish-pit part 2 is "

There is an exit to the east. Don't worry, it is unblocked. There is an unguarded exit to the south.".
The description of r_0 is "[dish-pit part 0][dish-pit part 1][dish-pit part 2]".

The r_1 is mapped south of r_0.
The r_3 is mapped east of r_0.
Understand "playroom" as r_4.
The internal name of r_4 is "playroom".
The printed name of r_4 is "-= Playroom =-".
The playroom part 0 is some text that varies. The playroom part 0 is "You find yourself in a playroom. A normal one.



 There is [if d_6 is open]an open[otherwise]a closed[end if]".
The playroom part 1 is some text that varies. The playroom part 1 is " door leading west. There is [if d_9 is open]an open[otherwise]a closed[end if]".
The playroom part 2 is some text that varies. The playroom part 2 is " hatch leading north.".
The description of r_4 is "[playroom part 0][playroom part 1][playroom part 2]".

west of r_4 and east of r_5 is a door called d_6.
north of r_4 and south of r_1 is a door called d_9.
Understand "cookhouse" as r_5.
The internal name of r_5 is "cookhouse".
The printed name of r_5 is "-= Cookhouse =-".
The cookhouse part 0 is some text that varies. The cookhouse part 0 is "You arrive in a cookhouse. A normal kind of place. I guess you better just go and list everything you see here.

 You can see a pan. [if there is something on the s_7]On the pan you can see [a list of things on the s_7]. I mean, just wow! Isn't TextWorld just the best?[end if]".
The cookhouse part 1 is some text that varies. The cookhouse part 1 is "[if there is nothing on the s_7]But the thing hasn't got anything on it.[end if]".
The cookhouse part 2 is some text that varies. The cookhouse part 2 is "

 There is [if d_6 is open]an open[otherwise]a closed[end if]".
The cookhouse part 3 is some text that varies. The cookhouse part 3 is " door leading east. There is [if d_5 is open]an open[otherwise]a closed[end if]".
The cookhouse part 4 is some text that varies. The cookhouse part 4 is " passageway leading south. There is [if d_4 is open]an open[otherwise]a closed[end if]".
The cookhouse part 5 is some text that varies. The cookhouse part 5 is " portal leading west.".
The description of r_5 is "[cookhouse part 0][cookhouse part 1][cookhouse part 2][cookhouse part 3][cookhouse part 4][cookhouse part 5]".

west of r_5 and east of r_6 is a door called d_4.
south of r_5 and north of r_15 is a door called d_5.
east of r_5 and west of r_4 is a door called d_6.
Understand "vault" as r_6.
The internal name of r_6 is "vault".
The printed name of r_6 is "-= Vault =-".
The vault part 0 is some text that varies. The vault part 0 is "You are in a vault. An ordinary kind of place.

 You can make out a table. [if there is something on the s_8]On the table you can see [a list of things on the s_8].[end if]".
The vault part 1 is some text that varies. The vault part 1 is "[if there is nothing on the s_8]Looks like someone's already been here and taken everything off it, though.[end if]".
The vault part 2 is some text that varies. The vault part 2 is "

 There is [if d_4 is open]an open[otherwise]a closed[end if]".
The vault part 3 is some text that varies. The vault part 3 is " portal leading east. There is [if d_3 is open]an open[otherwise]a closed[end if]".
The vault part 4 is some text that varies. The vault part 4 is " gate leading north.".
The description of r_6 is "[vault part 0][vault part 1][vault part 2][vault part 3][vault part 4]".

north of r_6 and south of r_7 is a door called d_3.
east of r_6 and west of r_5 is a door called d_4.
Understand "shower" as r_9.
The internal name of r_9 is "shower".
The printed name of r_9 is "-= Shower =-".
The shower part 0 is some text that varies. The shower part 0 is "I am sorry to announce that you are now in the shower.

 As if things weren't amazing enough already, you can even see a cabinet.[if c_6 is open and there is something in the c_6] The cabinet contains [a list of things in the c_6].[end if]".
The shower part 1 is some text that varies. The shower part 1 is "[if c_6 is open and the c_6 contains nothing] What a letdown! The cabinet is empty![end if]".
The shower part 2 is some text that varies. The shower part 2 is "

 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The shower part 3 is some text that varies. The shower part 3 is " stone passageway leading east. There is an unguarded exit to the north. There is an unblocked exit to the south.".
The description of r_9 is "[shower part 0][shower part 1][shower part 2][shower part 3]".

The r_10 is mapped south of r_9.
The r_11 is mapped north of r_9.
east of r_9 and west of r_8 is a door called d_0.
Understand "studio" as r_10.
The internal name of r_10 is "studio".
The printed name of r_10 is "-= Studio =-".
The studio part 0 is some text that varies. The studio part 0 is "I am obligated to announce that you are now in the studio. Okay, just remember what you're here to do, and everything will go great.

 You can make out a mantelpiece. [if there is something on the s_9]You see [a list of things on the s_9] on the mantelpiece. Something scurries by right in the corner of your eye. Probably nothing.[end if]".
The studio part 1 is some text that varies. The studio part 1 is "[if there is nothing on the s_9]But the thing is empty, unfortunately. It would have been so cool if there was stuff on the mantelpiece.[end if]".
The studio part 2 is some text that varies. The studio part 2 is "

There is an exit to the north. Don't worry, it is unblocked.".
The description of r_10 is "[studio part 0][studio part 1][studio part 2]".

The r_9 is mapped north of r_10.
Understand "cellar" as r_14.
The internal name of r_14 is "cellar".
The printed name of r_14 is "-= Cellar =-".
The cellar part 0 is some text that varies. The cellar part 0 is "You find yourself in a cellar. A standard one. The room seems oddly familiar, as though it were only superficially different from the other rooms in the building.

 You make out a counter. Wow, isn't TextWorld just the best? The counter is ordinary.[if there is something on the s_10] On the counter you can make out [a list of things on the s_10].[end if]".
The cellar part 1 is some text that varies. The cellar part 1 is "[if there is nothing on the s_10] Unfortunately, there isn't a thing on it.[end if]".
The cellar part 2 is some text that varies. The cellar part 2 is " You can make out a shelf. The shelf is usual.[if there is something on the s_11] On the shelf you can make out [a list of things on the s_11].[end if]".
The cellar part 3 is some text that varies. The cellar part 3 is "[if there is nothing on the s_11] But oh no! there's nothing on this piece of garbage. What's the point of an empty shelf?[end if]".
The cellar part 4 is some text that varies. The cellar part 4 is "

There is an exit to the north. Don't worry, it is unblocked.".
The description of r_14 is "[cellar part 0][cellar part 1][cellar part 2][cellar part 3][cellar part 4]".

The r_13 is mapped north of r_14.
Understand "basement" as r_18.
The internal name of r_18 is "basement".
The printed name of r_18 is "-= Basement =-".
The basement part 0 is some text that varies. The basement part 0 is "You are in a basement. A typical one.

 You can make out a workbench. The workbench is normal.[if there is something on the s_12] On the workbench you can make out [a list of things on the s_12].[end if]".
The basement part 1 is some text that varies. The basement part 1 is "[if there is nothing on the s_12] The workbench appears to be empty.[end if]".
The basement part 2 is some text that varies. The basement part 2 is "

There is an exit to the south. Don't worry, it is unblocked.".
The description of r_18 is "[basement part 0][basement part 1][basement part 2]".

The r_17 is mapped south of r_18.
Understand "cubicle" as r_7.
The internal name of r_7 is "cubicle".
The printed name of r_7 is "-= Cubicle =-".
The cubicle part 0 is some text that varies. The cubicle part 0 is "You're now in the cubicle. You decide to start listing off everything you see in the room, as if you were in a text adventure.



 There is [if d_2 is open]an open[otherwise]a closed[end if]".
The cubicle part 1 is some text that varies. The cubicle part 1 is " wooden gate leading north. There is [if d_3 is open]an open[otherwise]a closed[end if]".
The cubicle part 2 is some text that varies. The cubicle part 2 is " gate leading south.".
The description of r_7 is "[cubicle part 0][cubicle part 1][cubicle part 2]".

south of r_7 and north of r_6 is a door called d_3.
north of r_7 and south of r_8 is a door called d_2.

The c_0 and the c_1 and the c_2 and the c_3 and the c_4 and the c_5 and the c_6 are containers.
The c_0 and the c_1 and the c_2 and the c_3 and the c_4 and the c_5 and the c_6 are privately-named.
The d_9 and the d_1 and the d_5 and the d_8 and the d_7 and the d_6 and the d_4 and the d_3 and the d_2 and the d_0 are doors.
The d_9 and the d_1 and the d_5 and the d_8 and the d_7 and the d_6 and the d_4 and the d_3 and the d_2 and the d_0 are privately-named.
The f_0 are foods.
The f_0 are privately-named.
The k_0 and the k_1 and the k_2 and the k_3 are keys.
The k_0 and the k_1 and the k_2 and the k_3 are privately-named.
The o_0 are object-likes.
The o_0 are privately-named.
The r_12 and the r_11 and the r_13 and the r_8 and the r_15 and the r_19 and the r_16 and the r_17 and the r_2 and the r_1 and the r_3 and the r_0 and the r_4 and the r_5 and the r_6 and the r_9 and the r_10 and the r_14 and the r_18 and the r_7 are rooms.
The r_12 and the r_11 and the r_13 and the r_8 and the r_15 and the r_19 and the r_16 and the r_17 and the r_2 and the r_1 and the r_3 and the r_0 and the r_4 and the r_5 and the r_6 and the r_9 and the r_10 and the r_14 and the r_18 and the r_7 are privately-named.
The s_0 and the s_1 and the s_10 and the s_11 and the s_12 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 and the s_9 are supporters.
The s_0 and the s_1 and the s_10 and the s_11 and the s_12 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 and the s_9 are privately-named.

The description of d_9 is "it is what it is, a hatch [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_9 is "hatch".
Understand "hatch" as d_9.
The d_9 is open.
The description of d_1 is "it's an imposing gateway [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_1 is "gateway".
Understand "gateway" as d_1.
The d_1 is open.
The description of d_5 is "The passageway looks robust. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_5 is "passageway".
Understand "passageway" as d_5.
The d_5 is locked.
The description of d_8 is "it is what it is, a TextWorld limited edition door [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_8 is "TextWorld limited edition door".
Understand "TextWorld limited edition door" as d_8.
Understand "TextWorld" as d_8.
Understand "limited" as d_8.
Understand "edition" as d_8.
Understand "door" as d_8.
The d_8 is locked.
The description of d_7 is "it is what it is, a Microsoft gateway [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_7 is "Microsoft gateway".
Understand "Microsoft gateway" as d_7.
Understand "Microsoft" as d_7.
Understand "gateway" as d_7.
The d_7 is locked.
The description of d_6 is "it is what it is, a door [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_6 is "door".
Understand "door" as d_6.
The d_6 is closed.
The description of d_4 is "it is what it is, a portal [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_4 is "portal".
Understand "portal" as d_4.
The d_4 is open.
The description of d_3 is "it is what it is, a gate [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_3 is "gate".
Understand "gate" as d_3.
The d_3 is open.
The description of d_2 is "it is what it is, a wooden gate [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_2 is "wooden gate".
Understand "wooden gate" as d_2.
Understand "wooden" as d_2.
Understand "gate" as d_2.
The d_2 is open.
The description of d_0 is "it's a solid passageway [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_0 is "stone passageway".
Understand "stone passageway" as d_0.
Understand "stone" as d_0.
Understand "passageway" as d_0.
The d_0 is open.
The description of c_0 is "The dresser looks strong, and impossible to crack. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_0 is "dresser".
Understand "dresser" as c_0.
The c_0 is in r_11.
The c_0 is open.
The description of c_1 is "The basket looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_1 is "basket".
Understand "basket" as c_1.
The c_1 is in r_11.
The c_1 is locked.
The description of c_2 is "The suitcase looks strong, and impossible to break. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_2 is "suitcase".
Understand "suitcase" as c_2.
The c_2 is in r_13.
The c_2 is locked.
The description of c_3 is "The coffer looks strong, and impossible to crack. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_3 is "coffer".
Understand "coffer" as c_3.
The c_3 is in r_19.
The c_3 is closed.
The description of c_4 is "The bureau looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_4 is "bureau".
Understand "bureau" as c_4.
The c_4 is in r_16.
The c_4 is locked.
The description of c_5 is "The chest looks strong, and impossible to break. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_5 is "chest".
Understand "chest" as c_5.
The c_5 is in r_17.
The c_5 is locked.
The description of c_6 is "The cabinet looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_6 is "cabinet".
Understand "cabinet" as c_6.
The c_6 is in r_9.
The c_6 is open.
The description of f_0 is "You couldn't pay me to eat that typical thing.".
The printed name of f_0 is "stick of butter".
Understand "stick of butter" as f_0.
Understand "stick" as f_0.
Understand "butter" as f_0.
The f_0 is in r_15.
The description of o_0 is "The cushion would seem to be well matched to everything else here".
The printed name of o_0 is "cushion".
Understand "cushion" as o_0.
The o_0 is in r_4.
The description of s_0 is "The rack is reliable.".
The printed name of s_0 is "rack".
Understand "rack" as s_0.
The s_0 is in r_8.
The description of s_1 is "The stand is unstable.".
The printed name of s_1 is "stand".
Understand "stand" as s_1.
The s_1 is in r_16.
The description of s_10 is "The counter is solid.".
The printed name of s_10 is "counter".
Understand "counter" as s_10.
The s_10 is in r_14.
The description of s_11 is "The shelf is reliable.".
The printed name of s_11 is "shelf".
Understand "shelf" as s_11.
The s_11 is in r_14.
The description of s_12 is "The workbench is solidly built.".
The printed name of s_12 is "workbench".
Understand "workbench" as s_12.
The s_12 is in r_18.
The description of s_2 is "The bookshelf is shaky.".
The printed name of s_2 is "bookshelf".
Understand "bookshelf" as s_2.
The s_2 is in r_17.
The description of s_3 is "The mantle is an unstable piece of junk.".
The printed name of s_3 is "mantle".
Understand "mantle" as s_3.
The s_3 is in r_17.
The description of s_4 is "The chair is wobbly.".
The printed name of s_4 is "chair".
Understand "chair" as s_4.
The s_4 is in r_2.
The description of s_5 is "The desk is wobbly.".
The printed name of s_5 is "desk".
Understand "desk" as s_5.
The s_5 is in r_2.
The description of s_6 is "The board is durable.".
The printed name of s_6 is "board".
Understand "board" as s_6.
The s_6 is in r_0.
The description of s_7 is "The pan is shaky.".
The printed name of s_7 is "pan".
Understand "pan" as s_7.
The s_7 is in r_5.
The description of s_8 is "The table is solidly built.".
The printed name of s_8 is "table".
Understand "table" as s_8.
The s_8 is in r_6.
The description of s_9 is "The mantelpiece is shaky.".
The printed name of s_9 is "mantelpiece".
Understand "mantelpiece" as s_9.
The s_9 is in r_10.
The description of k_0 is "The keycard looks useful".
The printed name of k_0 is "keycard".
Understand "keycard" as k_0.
The k_0 is in the c_3.
The matching key of the d_5 is the k_0.
The description of k_1 is "The TextWorld limited edition passkey is heavier than it looks.".
The printed name of k_1 is "TextWorld limited edition passkey".
Understand "TextWorld limited edition passkey" as k_1.
Understand "TextWorld" as k_1.
Understand "limited" as k_1.
Understand "edition" as k_1.
Understand "passkey" as k_1.
The player carries the k_1.
The matching key of the d_8 is the k_1.
The description of k_2 is "The metal of the Microsoft key is satin.".
The printed name of k_2 is "Microsoft key".
Understand "Microsoft key" as k_2.
Understand "Microsoft" as k_2.
Understand "key" as k_2.
The k_2 is in the c_5.
The matching key of the d_7 is the k_2.
The description of k_3 is "The latchkey is light.".
The printed name of k_3 is "latchkey".
Understand "latchkey" as k_3.
The matching key of the c_5 is the k_3.
The k_3 is on the s_2.


The player is in r_17.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with "take latchkey from bookshelf / unlock chest with latchkey / open chest / take Microsoft key from chest / unlock Microsoft gateway with Microsoft key / open Microsoft gateway / go west / unlock TextWorld limited edition door with TextWorld limited edition passkey / open TextWorld limited edition door / go west / go west / open coffer / take keycard from coffer / go east / unlock passageway with keycard / open passageway / go north / open door / go east / take cushion"

Every turn:
	if quest0 completed is true:
		do nothing;
	else if The player carries the f_0:
		end the story; [Lost]
	else if The player is in r_4 and The player carries the o_0:
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

The objective part 0 is some text that varies. The objective part 0 is "You are now playing a exciting game of TextWorld! Your task for today is to pick-up the cushion from the floor of the playroom.".

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

