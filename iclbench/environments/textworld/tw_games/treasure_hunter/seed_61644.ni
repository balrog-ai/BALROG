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


The r_12 and the r_11 and the r_14 and the r_15 and the r_17 and the r_9 and the r_18 and the r_4 and the r_2 and the r_5 and the r_7 and the r_6 and the r_8 and the r_10 and the r_0 and the r_1 and the r_16 and the r_19 and the r_3 and the r_13 are rooms.

Understand "scullery" as r_12.
The internal name of r_12 is "scullery".
The printed name of r_12 is "-= Scullery =-".
The scullery part 0 is some text that varies. The scullery part 0 is "You arrive in a scullery. A normal one.



 There is [if d_6 is open]an open[otherwise]a closed[end if]".
The scullery part 1 is some text that varies. The scullery part 1 is " passageway leading north. There is [if d_4 is open]an open[otherwise]a closed[end if]".
The scullery part 2 is some text that varies. The scullery part 2 is " gateway leading south. There is [if d_5 is open]an open[otherwise]a closed[end if]".
The scullery part 3 is some text that varies. The scullery part 3 is " portal leading east. You need an unguarded exit? You should try going west.".
The description of r_12 is "[scullery part 0][scullery part 1][scullery part 2][scullery part 3]".

The r_11 is mapped west of r_12.
south of r_12 and north of r_13 is a door called d_4.
north of r_12 and south of r_9 is a door called d_6.
east of r_12 and west of r_18 is a door called d_5.
Understand "parlor" as r_11.
The internal name of r_11 is "parlor".
The printed name of r_11 is "-= Parlor =-".
The parlor part 0 is some text that varies. The parlor part 0 is "Well, here we are in the parlor. You decide to start listing off everything you see in the room, as if you were in a text adventure.



There is an exit to the east. Don't worry, it is unguarded. There is an unblocked exit to the north.".
The description of r_11 is "[parlor part 0]".

The r_10 is mapped north of r_11.
The r_12 is mapped east of r_11.
Understand "dish-pit" as r_14.
The internal name of r_14 is "dish-pit".
The printed name of r_14 is "-= Dish-Pit =-".
The dish-pit part 0 is some text that varies. The dish-pit part 0 is "You're now in the dish-pit.



 There is [if d_3 is open]an open[otherwise]a closed[end if]".
The dish-pit part 1 is some text that varies. The dish-pit part 1 is " type 6 portal leading north. There is [if d_2 is open]an open[otherwise]a closed[end if]".
The dish-pit part 2 is some text that varies. The dish-pit part 2 is " door leading west.".
The description of r_14 is "[dish-pit part 0][dish-pit part 1][dish-pit part 2]".

west of r_14 and east of r_15 is a door called d_2.
north of r_14 and south of r_13 is a door called d_3.
Understand "cookery" as r_15.
The internal name of r_15 is "cookery".
The printed name of r_15 is "-= Cookery =-".
The cookery part 0 is some text that varies. The cookery part 0 is "You arrive in a cookery. An ordinary one.

 You see [if c_0 is locked]a locked[else if c_0 is open]an opened[otherwise]a closed[end if]".
The cookery part 1 is some text that varies. The cookery part 1 is " case here.[if c_0 is open and there is something in the c_0] The case contains [a list of things in the c_0].[end if]".
The cookery part 2 is some text that varies. The cookery part 2 is "[if c_0 is open and the c_0 contains nothing] What a letdown! The case is empty![end if]".
The cookery part 3 is some text that varies. The cookery part 3 is " You can see a board. [if there is something on the s_0]On the board you can see [a list of things on the s_0].[end if]".
The cookery part 4 is some text that varies. The cookery part 4 is "[if there is nothing on the s_0]However, the board, like an empty board, has nothing on it.[end if]".
The cookery part 5 is some text that varies. The cookery part 5 is "

 There is [if d_2 is open]an open[otherwise]a closed[end if]".
The cookery part 6 is some text that varies. The cookery part 6 is " door leading east. There is an exit to the north. Don't worry, it is unblocked.".
The description of r_15 is "[cookery part 0][cookery part 1][cookery part 2][cookery part 3][cookery part 4][cookery part 5][cookery part 6]".

The r_16 is mapped north of r_15.
east of r_15 and west of r_14 is a door called d_2.
Understand "laundry place" as r_17.
The internal name of r_17 is "laundry place".
The printed name of r_17 is "-= Laundry Place =-".
The laundry place part 0 is some text that varies. The laundry place part 0 is "You're now in a laundry place. Let's see what's in here.

 You hear a noise behind you and spin around, but you can't see anything other than a table. The table is normal.[if there is something on the s_1] On the table you can see [a list of things on the s_1].[end if]".
The laundry place part 1 is some text that varies. The laundry place part 1 is "[if there is nothing on the s_1] But the thing hasn't got anything on it.[end if]".
The laundry place part 2 is some text that varies. The laundry place part 2 is "

There is an unguarded exit to the north. There is an exit to the west. Don't worry, it is unblocked.".
The description of r_17 is "[laundry place part 0][laundry place part 1][laundry place part 2]".

The r_9 is mapped west of r_17.
The r_7 is mapped north of r_17.
Understand "garage" as r_9.
The internal name of r_9 is "garage".
The printed name of r_9 is "-= Garage =-".
The garage part 0 is some text that varies. The garage part 0 is "You've entered a garage.

 [if c_1 is locked]A locked[else if c_1 is open]An open[otherwise]A closed[end if]".
The garage part 1 is some text that varies. The garage part 1 is " chest is right there by you.[if c_1 is open and there is something in the c_1] The chest contains [a list of things in the c_1]. There's something strange about this thing being here, but you don't have time to worry about that now.[end if]".
The garage part 2 is some text that varies. The garage part 2 is "[if c_1 is open and the c_1 contains nothing] The chest is empty! What a waste of a day![end if]".
The garage part 3 is some text that varies. The garage part 3 is " What's that over there? It looks like it's a counter. The counter is normal.[if there is something on the s_2] On the counter you make out [a list of things on the s_2]. You shudder, but continue examining the room.[end if]".
The garage part 4 is some text that varies. The garage part 4 is "[if there is nothing on the s_2] Looks like someone's already been here and taken everything off it, though.[end if]".
The garage part 5 is some text that varies. The garage part 5 is "

 There is [if d_6 is open]an open[otherwise]a closed[end if]".
The garage part 6 is some text that varies. The garage part 6 is " passageway leading south. There is an unguarded exit to the east. You don't like doors? Why not try going north, that entranceway is unguarded. There is an unguarded exit to the west.".
The description of r_9 is "[garage part 0][garage part 1][garage part 2][garage part 3][garage part 4][garage part 5][garage part 6]".

The r_10 is mapped west of r_9.
south of r_9 and north of r_12 is a door called d_6.
The r_8 is mapped north of r_9.
The r_17 is mapped east of r_9.
Understand "kitchenette" as r_18.
The internal name of r_18 is "kitchenette".
The printed name of r_18 is "-= Kitchenette =-".
The kitchenette part 0 is some text that varies. The kitchenette part 0 is "You find yourself in a kitchenette. An ordinary kind of place.

 You smell an awful smell, and follow it to a refrigerator.[if c_2 is open and there is something in the c_2] The refrigerator contains [a list of things in the c_2].[end if]".
The kitchenette part 1 is some text that varies. The kitchenette part 1 is "[if c_2 is open and the c_2 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The kitchenette part 2 is some text that varies. The kitchenette part 2 is "

 There is [if d_5 is open]an open[otherwise]a closed[end if]".
The kitchenette part 3 is some text that varies. The kitchenette part 3 is " portal leading west. You don't like doors? Why not try going south, that entranceway is unguarded.".
The description of r_18 is "[kitchenette part 0][kitchenette part 1][kitchenette part 2][kitchenette part 3]".

west of r_18 and east of r_12 is a door called d_5.
The r_19 is mapped south of r_18.
Understand "bedroom" as r_4.
The internal name of r_4 is "bedroom".
The printed name of r_4 is "-= Bedroom =-".
The bedroom part 0 is some text that varies. The bedroom part 0 is "You've entered a bedroom.

 You can see a coffer.[if c_3 is open and there is something in the c_3] The coffer contains [a list of things in the c_3].[end if]".
The bedroom part 1 is some text that varies. The bedroom part 1 is "[if c_3 is open and the c_3 contains nothing] The coffer is empty, what a horrible day![end if]".
The bedroom part 2 is some text that varies. The bedroom part 2 is " You make out [if c_4 is locked]a locked[else if c_4 is open]an opened[otherwise]a closed[end if]".
The bedroom part 3 is some text that varies. The bedroom part 3 is " basket close by.[if c_4 is open and there is something in the c_4] The basket contains [a list of things in the c_4].[end if]".
The bedroom part 4 is some text that varies. The bedroom part 4 is "[if c_4 is open and the c_4 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The bedroom part 5 is some text that varies. The bedroom part 5 is "

You don't like doors? Why not try going west, that entranceway is unblocked.".
The description of r_4 is "[bedroom part 0][bedroom part 1][bedroom part 2][bedroom part 3][bedroom part 4][bedroom part 5]".

The r_2 is mapped west of r_4.
Understand "cookhouse" as r_2.
The internal name of r_2 is "cookhouse".
The printed name of r_2 is "-= Cookhouse =-".
The cookhouse part 0 is some text that varies. The cookhouse part 0 is "You arrive in a cookhouse. An ordinary kind of place.



 There is [if d_7 is open]an open[otherwise]a closed[end if]".
The cookhouse part 1 is some text that varies. The cookhouse part 1 is " gate leading north. There is [if d_8 is open]an open[otherwise]a closed[end if]".
The cookhouse part 2 is some text that varies. The cookhouse part 2 is " hatch leading south. There is an exit to the east. Don't worry, it is unblocked.".
The description of r_2 is "[cookhouse part 0][cookhouse part 1][cookhouse part 2]".

south of r_2 and north of r_1 is a door called d_8.
north of r_2 and south of r_3 is a door called d_7.
The r_4 is mapped east of r_2.
Understand "cellar" as r_5.
The internal name of r_5 is "cellar".
The printed name of r_5 is "-= Cellar =-".
The cellar part 0 is some text that varies. The cellar part 0 is "You have stumbled into a cellar. Not the cellar you'd expect. No, this is a cellar. The room is well lit.

 You see a workbench. Make a note of this, you might have to put stuff on or in it later on. The workbench is standard.[if there is something on the s_3] On the workbench you see [a list of things on the s_3].[end if]".
The cellar part 1 is some text that varies. The cellar part 1 is "[if there is nothing on the s_3] But the thing hasn't got anything on it. Oh! Why couldn't there just be stuff on it?[end if]".
The cellar part 2 is some text that varies. The cellar part 2 is "

 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The cellar part 3 is some text that varies. The cellar part 3 is " walnut passageway leading south. There is [if d_0 is open]an open[otherwise]a closed[end if]".
The cellar part 4 is some text that varies. The cellar part 4 is " cedar gateway leading west. You need an unblocked exit? You should try going east.".
The description of r_5 is "[cellar part 0][cellar part 1][cellar part 2][cellar part 3][cellar part 4]".

west of r_5 and east of r_7 is a door called d_0.
south of r_5 and north of r_3 is a door called d_1.
The r_6 is mapped east of r_5.
Understand "kitchen" as r_7.
The internal name of r_7 is "kitchen".
The printed name of r_7 is "-= Kitchen =-".
The kitchen part 0 is some text that varies. The kitchen part 0 is "You are in a kitchen. A normal kind of place.

 You make out a fridge.[if c_5 is open and there is something in the c_5] The fridge contains [a list of things in the c_5].[end if]".
The kitchen part 1 is some text that varies. The kitchen part 1 is "[if c_5 is open and the c_5 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The kitchen part 2 is some text that varies. The kitchen part 2 is "

 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The kitchen part 3 is some text that varies. The kitchen part 3 is " cedar gateway leading east. There is an unblocked exit to the south. You need an unblocked exit? You should try going west.".
The description of r_7 is "[kitchen part 0][kitchen part 1][kitchen part 2][kitchen part 3]".

The r_8 is mapped west of r_7.
The r_17 is mapped south of r_7.
east of r_7 and west of r_5 is a door called d_0.
Understand "restroom" as r_6.
The internal name of r_6 is "restroom".
The printed name of r_6 is "-= Restroom =-".
The restroom part 0 is some text that varies. The restroom part 0 is "I am obligated to announce that you are now in the restroom.



There is an unblocked exit to the west.".
The description of r_6 is "[restroom part 0]".

The r_5 is mapped west of r_6.
Understand "playroom" as r_8.
The internal name of r_8 is "playroom".
The printed name of r_8 is "-= Playroom =-".
The playroom part 0 is some text that varies. The playroom part 0 is "You're not going to believe this, but you've just entered a playroom.

 You can see a locker.[if c_6 is open and there is something in the c_6] The locker contains [a list of things in the c_6].[end if]".
The playroom part 1 is some text that varies. The playroom part 1 is "[if c_6 is open and the c_6 contains nothing] The locker is empty, what a horrible day![end if]".
The playroom part 2 is some text that varies. The playroom part 2 is " You rest your hand against a wall, but you miss the wall and fall onto a box. The light flickers for a second, but nothing else happens.[if c_7 is open and there is something in the c_7] The box contains [a list of things in the c_7]. The light flickers for a second, but nothing else happens.[end if]".
The playroom part 3 is some text that varies. The playroom part 3 is "[if c_7 is open and the c_7 contains nothing] What a letdown! The box is empty![end if]".
The playroom part 4 is some text that varies. The playroom part 4 is "

There is an exit to the east. Don't worry, it is unblocked. There is an exit to the south. Don't worry, it is unblocked.".
The description of r_8 is "[playroom part 0][playroom part 1][playroom part 2][playroom part 3][playroom part 4]".

The r_9 is mapped south of r_8.
The r_7 is mapped east of r_8.
Understand "canteen" as r_10.
The internal name of r_10 is "canteen".
The printed name of r_10 is "-= Canteen =-".
The canteen part 0 is some text that varies. The canteen part 0 is "Well, here we are in the canteen. The room is well lit.

 You smell a great smell, and follow it to a big cabinet.[if c_8 is open and there is something in the c_8] The big cabinet contains [a list of things in the c_8]. Wow, isn't TextWorld just the best?[end if]".
The canteen part 1 is some text that varies. The canteen part 1 is "[if c_8 is open and the c_8 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The canteen part 2 is some text that varies. The canteen part 2 is " You scan the room, seeing a platter. [if there is something on the s_4]On the platter you can see [a list of things on the s_4].[end if]".
The canteen part 3 is some text that varies. The canteen part 3 is "[if there is nothing on the s_4]The platter appears to be empty. Hm. Oh well[end if]".
The canteen part 4 is some text that varies. The canteen part 4 is "

There is an exit to the east. Don't worry, it is unblocked. There is an exit to the south. Don't worry, it is unblocked.".
The description of r_10 is "[canteen part 0][canteen part 1][canteen part 2][canteen part 3][canteen part 4]".

The r_11 is mapped south of r_10.
The r_9 is mapped east of r_10.
Understand "super hot kitchenette" as r_0.
The internal name of r_0 is "super hot kitchenette".
The printed name of r_0 is "-= Super Hot Kitchenette =-".
The super hot kitchenette part 0 is some text that varies. The super hot kitchenette part 0 is "Ah, the kitchenette. This is some kind of kitchenette, really great super hot vibes in this place, a wonderful super hot atmosphere. And now, well, you're in it.



You need an unguarded exit? You should try going north.".
The description of r_0 is "[super hot kitchenette part 0]".

The r_1 is mapped north of r_0.
Understand "lounge" as r_1.
The internal name of r_1 is "lounge".
The printed name of r_1 is "-= Lounge =-".
The lounge part 0 is some text that varies. The lounge part 0 is "You find yourself in a lounge. An usual kind of place. Okay, just remember what you're here to do, and everything will go great.

 You lean against the wall, inadvertently pressing a secret button. The wall opens up to reveal a dresser.[if c_9 is open and there is something in the c_9] The dresser contains [a list of things in the c_9].[end if]".
The lounge part 1 is some text that varies. The lounge part 1 is "[if c_9 is open and the c_9 contains nothing] The dresser is empty! What a waste of a day![end if]".
The lounge part 2 is some text that varies. The lounge part 2 is "

 There is [if d_8 is open]an open[otherwise]a closed[end if]".
The lounge part 3 is some text that varies. The lounge part 3 is " hatch leading north. There is an unguarded exit to the south.".
The description of r_1 is "[lounge part 0][lounge part 1][lounge part 2][lounge part 3]".

The r_0 is mapped south of r_1.
north of r_1 and south of r_2 is a door called d_8.
Understand "sweaty kitchen" as r_16.
The internal name of r_16 is "sweaty kitchen".
The printed name of r_16 is "-= Sweaty Kitchen =-".
The sweaty kitchen part 0 is some text that varies. The sweaty kitchen part 0 is "You arrive in a kitchen. A sweaty one. You start to take note of what's in the room.

 You see a freezer. The light flickers for a second, but nothing else happens.[if c_11 is open and there is something in the c_11] The freezer contains [a list of things in the c_11]. I mean, just wow! Isn't TextWorld just the best?[end if]".
The sweaty kitchen part 1 is some text that varies. The sweaty kitchen part 1 is "[if c_11 is open and the c_11 contains nothing] The freezer is empty! This is the worst thing that could possibly happen, ever![end if]".
The sweaty kitchen part 2 is some text that varies. The sweaty kitchen part 2 is " You can see a saucepan. [if there is something on the s_5]You see [a list of things on the s_5] on the saucepan.[end if]".
The sweaty kitchen part 3 is some text that varies. The sweaty kitchen part 3 is "[if there is nothing on the s_5]But the thing is empty, unfortunately.[end if]".
The sweaty kitchen part 4 is some text that varies. The sweaty kitchen part 4 is " You see a rack. The rack is normal.[if there is something on the s_6] On the rack you see [a list of things on the s_6].[end if]".
The sweaty kitchen part 5 is some text that varies. The sweaty kitchen part 5 is "[if there is nothing on the s_6] Looks like someone's already been here and taken everything off it, though. Aw, here you were, all excited for there to be things on it![end if]".
The sweaty kitchen part 6 is some text that varies. The sweaty kitchen part 6 is "

There is an unguarded exit to the south.".
The description of r_16 is "[sweaty kitchen part 0][sweaty kitchen part 1][sweaty kitchen part 2][sweaty kitchen part 3][sweaty kitchen part 4][sweaty kitchen part 5][sweaty kitchen part 6]".

The r_15 is mapped south of r_16.
Understand "chamber" as r_19.
The internal name of r_19 is "chamber".
The printed name of r_19 is "-= Chamber =-".
The chamber part 0 is some text that varies. The chamber part 0 is "You arrive in a chamber. A normal one.

 You see [if c_12 is locked]a locked[else if c_12 is open]an opened[otherwise]a closed[end if]".
The chamber part 1 is some text that varies. The chamber part 1 is " standard looking safe nearby.[if c_12 is open and there is something in the c_12] The safe contains [a list of things in the c_12]. You can't really describe the safe besides that it's standard.[end if]".
The chamber part 2 is some text that varies. The chamber part 2 is "[if c_12 is open and the c_12 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The chamber part 3 is some text that varies. The chamber part 3 is "

You don't like doors? Why not try going north, that entranceway is unguarded.".
The description of r_19 is "[chamber part 0][chamber part 1][chamber part 2][chamber part 3]".

The r_18 is mapped north of r_19.
Understand "balmy cookhouse" as r_3.
The internal name of r_3 is "balmy cookhouse".
The printed name of r_3 is "-= Balmy Cookhouse =-".
The balmy cookhouse part 0 is some text that varies. The balmy cookhouse part 0 is "You've entered a balmy cookhouse.



 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The balmy cookhouse part 1 is some text that varies. The balmy cookhouse part 1 is " walnut passageway leading north. There is [if d_7 is open]an open[otherwise]a closed[end if]".
The balmy cookhouse part 2 is some text that varies. The balmy cookhouse part 2 is " gate leading south.".
The description of r_3 is "[balmy cookhouse part 0][balmy cookhouse part 1][balmy cookhouse part 2]".

south of r_3 and north of r_2 is a door called d_7.
north of r_3 and south of r_5 is a door called d_1.
Understand "sweaty scullery" as r_13.
The internal name of r_13 is "sweaty scullery".
The printed name of r_13 is "-= Sweaty Scullery =-".
The sweaty scullery part 0 is some text that varies. The sweaty scullery part 0 is "You arrive in a scullery. A sweaty one.

 You see [if c_10 is locked]a locked[else if c_10 is open]an opened[otherwise]a closed[end if]".
The sweaty scullery part 1 is some text that varies. The sweaty scullery part 1 is " standard looking cabinet in the corner.[if c_10 is open and there is something in the c_10] The cabinet contains [a list of things in the c_10]. Hmmm... what else, what else?[end if]".
The sweaty scullery part 2 is some text that varies. The sweaty scullery part 2 is "[if c_10 is open and the c_10 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The sweaty scullery part 3 is some text that varies. The sweaty scullery part 3 is "

 There is [if d_4 is open]an open[otherwise]a closed[end if]".
The sweaty scullery part 4 is some text that varies. The sweaty scullery part 4 is " gateway leading north. There is [if d_3 is open]an open[otherwise]a closed[end if]".
The sweaty scullery part 5 is some text that varies. The sweaty scullery part 5 is " type 6 portal leading south.".
The description of r_13 is "[sweaty scullery part 0][sweaty scullery part 1][sweaty scullery part 2][sweaty scullery part 3][sweaty scullery part 4][sweaty scullery part 5]".

south of r_13 and north of r_14 is a door called d_3.
north of r_13 and south of r_12 is a door called d_4.

The c_0 and the c_1 and the c_10 and the c_11 and the c_12 and the c_2 and the c_3 and the c_4 and the c_5 and the c_6 and the c_7 and the c_8 and the c_9 are containers.
The c_0 and the c_1 and the c_10 and the c_11 and the c_12 and the c_2 and the c_3 and the c_4 and the c_5 and the c_6 and the c_7 and the c_8 and the c_9 are privately-named.
The d_8 and the d_4 and the d_5 and the d_6 and the d_3 and the d_2 and the d_7 and the d_1 and the d_0 are doors.
The d_8 and the d_4 and the d_5 and the d_6 and the d_3 and the d_2 and the d_7 and the d_1 and the d_0 are privately-named.
The f_0 and the f_1 are foods.
The f_0 and the f_1 are privately-named.
The k_0 and the k_1 are keys.
The k_0 and the k_1 are privately-named.
The r_12 and the r_11 and the r_14 and the r_15 and the r_17 and the r_9 and the r_18 and the r_4 and the r_2 and the r_5 and the r_7 and the r_6 and the r_8 and the r_10 and the r_0 and the r_1 and the r_16 and the r_19 and the r_3 and the r_13 are rooms.
The r_12 and the r_11 and the r_14 and the r_15 and the r_17 and the r_9 and the r_18 and the r_4 and the r_2 and the r_5 and the r_7 and the r_6 and the r_8 and the r_10 and the r_0 and the r_1 and the r_16 and the r_19 and the r_3 and the r_13 are privately-named.
The s_0 and the s_1 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 are supporters.
The s_0 and the s_1 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 are privately-named.

The description of d_8 is "The hatch looks imposing. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_8 is "hatch".
Understand "hatch" as d_8.
The d_8 is open.
The description of d_4 is "The gateway looks hefty. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_4 is "gateway".
Understand "gateway" as d_4.
The d_4 is locked.
The description of d_5 is "The portal looks solid. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_5 is "portal".
Understand "portal" as d_5.
The d_5 is open.
The description of d_6 is "it is what it is, a passageway [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_6 is "passageway".
Understand "passageway" as d_6.
The d_6 is closed.
The description of d_3 is "it is what it is, a type 6 portal [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_3 is "type 6 portal".
Understand "type 6 portal" as d_3.
Understand "type" as d_3.
Understand "6" as d_3.
Understand "portal" as d_3.
The d_3 is locked.
The description of d_2 is "The door looks robust. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_2 is "door".
Understand "door" as d_2.
The d_2 is open.
The description of d_7 is "The gate looks imposing. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_7 is "gate".
Understand "gate" as d_7.
The d_7 is open.
The description of d_1 is "The walnut passageway looks commanding. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_1 is "walnut passageway".
Understand "walnut passageway" as d_1.
Understand "walnut" as d_1.
Understand "passageway" as d_1.
The d_1 is closed.
The description of d_0 is "it is what it is, a cedar gateway [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_0 is "cedar gateway".
Understand "cedar gateway" as d_0.
Understand "cedar" as d_0.
Understand "gateway" as d_0.
The d_0 is open.
The description of c_0 is "The case looks strong, and impossible to break. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_0 is "case".
Understand "case" as c_0.
The c_0 is in r_15.
The c_0 is open.
The description of c_1 is "The chest looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_1 is "chest".
Understand "chest" as c_1.
The c_1 is in r_9.
The c_1 is locked.
The description of c_10 is "The cabinet looks strong, and impossible to destroy. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_10 is "cabinet".
Understand "cabinet" as c_10.
The c_10 is in r_13.
The c_10 is open.
The description of c_11 is "The freezer looks strong, and impossible to destroy. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_11 is "freezer".
Understand "freezer" as c_11.
The c_11 is in r_16.
The c_11 is open.
The description of c_12 is "The safe looks strong, and impossible to crack. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_12 is "safe".
Understand "safe" as c_12.
The c_12 is in r_19.
The c_12 is closed.
The description of c_2 is "The refrigerator looks strong, and impossible to crack. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_2 is "refrigerator".
Understand "refrigerator" as c_2.
The c_2 is in r_18.
The c_2 is open.
The description of c_3 is "The coffer looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_3 is "coffer".
Understand "coffer" as c_3.
The c_3 is in r_4.
The c_3 is locked.
The description of c_4 is "The basket looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_4 is "basket".
Understand "basket" as c_4.
The c_4 is in r_4.
The c_4 is locked.
The description of c_5 is "The fridge looks strong, and impossible to break. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_5 is "fridge".
Understand "fridge" as c_5.
The c_5 is in r_7.
The c_5 is open.
The description of c_6 is "The locker looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_6 is "locker".
Understand "locker" as c_6.
The c_6 is in r_8.
The c_6 is locked.
The description of c_7 is "The box looks strong, and impossible to crack. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_7 is "box".
Understand "box" as c_7.
The c_7 is in r_8.
The c_7 is locked.
The description of c_8 is "The big cabinet looks strong, and impossible to crack. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_8 is "big cabinet".
Understand "big cabinet" as c_8.
Understand "big" as c_8.
Understand "cabinet" as c_8.
The c_8 is in r_10.
The c_8 is locked.
The description of c_9 is "The dresser looks strong, and impossible to destroy. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_9 is "dresser".
Understand "dresser" as c_9.
The c_9 is in r_1.
The c_9 is open.
The description of f_0 is "You couldn't pay me to eat that ordinary thing.".
The printed name of f_0 is "pizza".
Understand "pizza" as f_0.
The f_0 is in r_4.
The description of f_1 is "that's a typical cauliflower!".
The printed name of f_1 is "cauliflower".
Understand "cauliflower" as f_1.
The f_1 is in r_2.
The description of k_0 is "The passkey is light.".
The printed name of k_0 is "passkey".
Understand "passkey" as k_0.
The k_0 is in r_13.
The matching key of the d_4 is the k_0.
The description of k_1 is "The type 6 key is heavier than it looks.".
The printed name of k_1 is "type 6 key".
Understand "type 6 key" as k_1.
Understand "type" as k_1.
Understand "6" as k_1.
Understand "key" as k_1.
The k_1 is in r_14.
The matching key of the d_3 is the k_1.
The description of s_0 is "The board is solid.".
The printed name of s_0 is "board".
Understand "board" as s_0.
The s_0 is in r_15.
The description of s_1 is "The table is reliable.".
The printed name of s_1 is "table".
Understand "table" as s_1.
The s_1 is in r_17.
The description of s_2 is "The counter is wobbly.".
The printed name of s_2 is "counter".
Understand "counter" as s_2.
The s_2 is in r_9.
The description of s_3 is "The workbench is wobbly.".
The printed name of s_3 is "workbench".
Understand "workbench" as s_3.
The s_3 is in r_5.
The description of s_4 is "The platter is unstable.".
The printed name of s_4 is "platter".
Understand "platter" as s_4.
The s_4 is in r_10.
The description of s_5 is "The saucepan is durable.".
The printed name of s_5 is "saucepan".
Understand "saucepan" as s_5.
The s_5 is in r_16.
The description of s_6 is "The rack is shaky.".
The printed name of s_6 is "rack".
Understand "rack" as s_6.
The s_6 is in r_16.


The player is in r_16.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with "go south / go east / take type 6 key / unlock type 6 portal with type 6 key / open type 6 portal / go north / take passkey / unlock gateway with passkey / open gateway / go north / open passageway / go north / go north / go east / go east / open walnut passageway / go south / go south / go east / take pizza"

Every turn:
	if quest0 completed is true:
		do nothing;
	else if The player carries the f_1:
		end the story; [Lost]
	else if The player is in r_4 and The player carries the f_0:
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

The objective part 0 is some text that varies. The objective part 0 is "Welcome to another life changing episode of TextWorld! Your task for today is to pick-up the pizza from the floor of the bedroom.".

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

