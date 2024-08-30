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


The r_10 and the r_9 and the r_11 and the r_12 and the r_13 and the r_16 and the r_15 and the r_17 and the r_14 and the r_18 and the r_5 and the r_3 and the r_2 and the r_4 and the r_1 and the r_7 and the r_6 and the r_8 and the r_0 and the r_19 are rooms.

Understand "attic" as r_10.
The internal name of r_10 is "attic".
The printed name of r_10 is "-= Attic =-".
The attic part 0 is some text that varies. The attic part 0 is "Fancy seeing you here. Here, by the way, being the attic. I guess you better just go and list everything you see here.



You need an unguarded exit? You should try going west.".
The description of r_10 is "[attic part 0]".

The r_9 is mapped west of r_10.
Understand "office" as r_9.
The internal name of r_9 is "office".
The printed name of r_9 is "-= Office =-".
The office part 0 is some text that varies. The office part 0 is "You find yourself in an office. A standard kind of place. You begin to take stock of what's here.



 There is [if d_2 is open]an open[otherwise]a closed[end if]".
The office part 1 is some text that varies. The office part 1 is " Henderson's limited edition passageway leading west. There is [if d_1 is open]an open[otherwise]a closed[end if]".
The office part 2 is some text that varies. The office part 2 is " door leading north. There is an exit to the east. Don't worry, it is unblocked.".
The description of r_9 is "[office part 0][office part 1][office part 2]".

west of r_9 and east of r_8 is a door called d_2.
north of r_9 and south of r_11 is a door called d_1.
The r_10 is mapped east of r_9.
Understand "lounge" as r_11.
The internal name of r_11 is "lounge".
The printed name of r_11 is "-= Lounge =-".
The lounge part 0 is some text that varies. The lounge part 0 is "You arrive in a lounge. An ordinary one.



 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The lounge part 1 is some text that varies. The lounge part 1 is " door leading south. There is [if d_4 is open]an open[otherwise]a closed[end if]".
The lounge part 2 is some text that varies. The lounge part 2 is " hatch leading west. You don't like doors? Why not try going east, that entranceway is unblocked.".
The description of r_11 is "[lounge part 0][lounge part 1][lounge part 2]".

west of r_11 and east of r_12 is a door called d_4.
south of r_11 and north of r_9 is a door called d_1.
The r_13 is mapped east of r_11.
Understand "vault" as r_12.
The internal name of r_12 is "vault".
The printed name of r_12 is "-= Vault =-".
The vault part 0 is some text that varies. The vault part 0 is "A well framed placard tells you that you are now in the vault. Let's see what's in here.



 There is [if d_3 is open]an open[otherwise]a closed[end if]".
The vault part 1 is some text that varies. The vault part 1 is " passageway leading north. There is [if d_4 is open]an open[otherwise]a closed[end if]".
The vault part 2 is some text that varies. The vault part 2 is " hatch leading east. There is [if d_5 is open]an open[otherwise]a closed[end if]".
The vault part 3 is some text that varies. The vault part 3 is " portal leading south.".
The description of r_12 is "[vault part 0][vault part 1][vault part 2][vault part 3]".

south of r_12 and north of r_8 is a door called d_5.
north of r_12 and south of r_14 is a door called d_3.
east of r_12 and west of r_11 is a door called d_4.
Understand "canteen" as r_13.
The internal name of r_13 is "canteen".
The printed name of r_13 is "-= Canteen =-".
The canteen part 0 is some text that varies. The canteen part 0 is "Look around you. Take it all in. It's not every day someone gets to be in a canteen.

 You can see a plate. [if there is something on the s_0]On the plate you make out [a list of things on the s_0].[end if]".
The canteen part 1 is some text that varies. The canteen part 1 is "[if there is nothing on the s_0]But there isn't a thing on it. Oh! Why couldn't there just be stuff on it?[end if]".
The canteen part 2 is some text that varies. The canteen part 2 is "

There is an unblocked exit to the west.".
The description of r_13 is "[canteen part 0][canteen part 1][canteen part 2]".

The r_11 is mapped west of r_13.
Understand "cookhouse" as r_16.
The internal name of r_16 is "cookhouse".
The printed name of r_16 is "-= Cookhouse =-".
The cookhouse part 0 is some text that varies. The cookhouse part 0 is "Of every cookhouse you could have shown up in, you had to walk into a normal one.



There is an exit to the south. Don't worry, it is unguarded. You don't like doors? Why not try going west, that entranceway is unguarded.".
The description of r_16 is "[cookhouse part 0]".

The r_15 is mapped west of r_16.
The r_17 is mapped south of r_16.
Understand "laundromat" as r_15.
The internal name of r_15 is "laundromat".
The printed name of r_15 is "-= Laundromat =-".
The laundromat part 0 is some text that varies. The laundromat part 0 is "You find yourself in a laundromat. An usual one.

 What's that over there? It looks like it's a drawer. Hmm. You always thought you'd be more excited to see a drawer in a room.[if c_0 is open and there is something in the c_0] The drawer contains [a list of things in the c_0].[end if]".
The laundromat part 1 is some text that varies. The laundromat part 1 is "[if c_0 is open and the c_0 contains nothing] What a letdown! The drawer is empty![end if]".
The laundromat part 2 is some text that varies. The laundromat part 2 is " You lean against the wall, inadvertently pressing a secret button. The wall opens up to reveal a shelf. [if there is something on the s_1]You see [a list of things on the s_1] on the shelf.[end if]".
The laundromat part 3 is some text that varies. The laundromat part 3 is "[if there is nothing on the s_1]But there isn't a thing on it.[end if]".
The laundromat part 4 is some text that varies. The laundromat part 4 is "

There is an unblocked exit to the east. You don't like doors? Why not try going south, that entranceway is unguarded.".
The description of r_15 is "[laundromat part 0][laundromat part 1][laundromat part 2][laundromat part 3][laundromat part 4]".

The r_14 is mapped south of r_15.
The r_16 is mapped east of r_15.
Understand "kitchenette" as r_17.
The internal name of r_17 is "kitchenette".
The printed name of r_17 is "-= Kitchenette =-".
The kitchenette part 0 is some text that varies. The kitchenette part 0 is "You've come into an usual room. Your mind races to think of what kind of room would be usual. And then it hits you. Of course. You're in the kitchenette.



You don't like doors? Why not try going north, that entranceway is unblocked. There is an unguarded exit to the west.".
The description of r_17 is "[kitchenette part 0]".

The r_14 is mapped west of r_17.
The r_16 is mapped north of r_17.
Understand "basement" as r_14.
The internal name of r_14 is "basement".
The printed name of r_14 is "-= Basement =-".
The basement part 0 is some text that varies. The basement part 0 is "You've entered a basement.



 There is [if d_3 is open]an open[otherwise]a closed[end if]".
The basement part 1 is some text that varies. The basement part 1 is " passageway leading south. There is an unblocked exit to the east. You don't like doors? Why not try going north, that entranceway is unguarded.".
The description of r_14 is "[basement part 0][basement part 1]".

south of r_14 and north of r_12 is a door called d_3.
The r_15 is mapped north of r_14.
The r_17 is mapped east of r_14.
Understand "scullery" as r_18.
The internal name of r_18 is "scullery".
The printed name of r_18 is "-= Scullery =-".
The scullery part 0 is some text that varies. The scullery part 0 is "Look around you. Take it all in. It's not every day someone gets to be in a scullery.

 You can make out [if c_1 is locked]a locked[else if c_1 is open]an opened[otherwise]a closed[end if]".
The scullery part 1 is some text that varies. The scullery part 1 is " standard looking case nearby.[if c_1 is open and there is something in the c_1] The case contains [a list of things in the c_1].[end if]".
The scullery part 2 is some text that varies. The scullery part 2 is "[if c_1 is open and the c_1 contains nothing] What a letdown! The case is empty![end if]".
The scullery part 3 is some text that varies. The scullery part 3 is "

 There is [if d_7 is open]an open[otherwise]a closed[end if]".
The scullery part 4 is some text that varies. The scullery part 4 is " stone door leading south. There is [if d_8 is open]an open[otherwise]a closed[end if]".
The scullery part 5 is some text that varies. The scullery part 5 is " stone gate leading west. You don't like doors? Why not try going north, that entranceway is unblocked.".
The description of r_18 is "[scullery part 0][scullery part 1][scullery part 2][scullery part 3][scullery part 4][scullery part 5]".

west of r_18 and east of r_5 is a door called d_8.
south of r_18 and north of r_7 is a door called d_7.
The r_19 is mapped north of r_18.
Understand "workshop" as r_5.
The internal name of r_5 is "workshop".
The printed name of r_5 is "-= Workshop =-".
The workshop part 0 is some text that varies. The workshop part 0 is "You've entered a workshop. You begin looking for stuff.

 You can make out [if c_2 is locked]a locked[else if c_2 is open]an opened[otherwise]a closed[end if]".
The workshop part 1 is some text that varies. The workshop part 1 is " display right there by you.[if c_2 is open and there is something in the c_2] The display contains [a list of things in the c_2].[end if]".
The workshop part 2 is some text that varies. The workshop part 2 is "[if c_2 is open and the c_2 contains nothing] The display is empty! This is the worst thing that could possibly happen, ever![end if]".
The workshop part 3 is some text that varies. The workshop part 3 is "

 There is [if d_8 is open]an open[otherwise]a closed[end if]".
The workshop part 4 is some text that varies. The workshop part 4 is " stone gate leading east. There is [if d_9 is open]an open[otherwise]a closed[end if]".
The workshop part 5 is some text that varies. The workshop part 5 is " gateway leading north. You need an unblocked exit? You should try going south.".
The description of r_5 is "[workshop part 0][workshop part 1][workshop part 2][workshop part 3][workshop part 4][workshop part 5]".

The r_6 is mapped south of r_5.
north of r_5 and south of r_0 is a door called d_9.
east of r_5 and west of r_18 is a door called d_8.
Understand "studio" as r_3.
The internal name of r_3 is "studio".
The printed name of r_3 is "-= Studio =-".
The studio part 0 is some text that varies. The studio part 0 is "You make another one of your grand eccentric entrances into a studio.



There is an unblocked exit to the south. There is an exit to the west. Don't worry, it is unblocked.".
The description of r_3 is "[studio part 0]".

The r_2 is mapped west of r_3.
The r_4 is mapped south of r_3.
Understand "kitchen" as r_2.
The internal name of r_2 is "kitchen".
The printed name of r_2 is "-= Kitchen =-".
The kitchen part 0 is some text that varies. The kitchen part 0 is "You are in a kitchen. An ordinary kind of place.

 You see a table. The table is ordinary.[if there is something on the s_2] On the table you can see [a list of things on the s_2].[end if]".
The kitchen part 1 is some text that varies. The kitchen part 1 is "[if there is nothing on the s_2] However, the table, like an empty table, has nothing on it. Hm. Oh well[end if]".
The kitchen part 2 is some text that varies. The kitchen part 2 is "

There is an unblocked exit to the east. There is an unblocked exit to the south.".
The description of r_2 is "[kitchen part 0][kitchen part 1][kitchen part 2]".

The r_1 is mapped south of r_2.
The r_3 is mapped east of r_2.
Understand "cookery" as r_4.
The internal name of r_4 is "cookery".
The printed name of r_4 is "-= Cookery =-".
The cookery part 0 is some text that varies. The cookery part 0 is "Okay, so you're in a cookery, cool, but is it normal? You better believe it is.

 You can see a rack. [if there is something on the s_3]On the rack you can see [a list of things on the s_3]. Suddenly, you bump your head on the ceiling, but it's not such a bad bump that it's going to prevent you from looking at objects and even things.[end if]".
The cookery part 1 is some text that varies. The cookery part 1 is "[if there is nothing on the s_3]But oh no! there's nothing on this piece of garbage.[end if]".
The cookery part 2 is some text that varies. The cookery part 2 is " You make out a pan. You shudder, but continue examining the pan. The pan is ordinary.[if there is something on the s_4] On the pan you can make out [a list of things on the s_4].[end if]".
The cookery part 3 is some text that varies. The cookery part 3 is "[if there is nothing on the s_4] But the thing hasn't got anything on it.[end if]".
The cookery part 4 is some text that varies. The cookery part 4 is " Look out! It's a- oh, never mind, it's just a board. The board is normal.[if there is something on the s_5] On the board you can make out [a list of things on the s_5].[end if]".
The cookery part 5 is some text that varies. The cookery part 5 is "[if there is nothing on the s_5] But the thing hasn't got anything on it. Oh! Why couldn't there just be stuff on it?[end if]".
The cookery part 6 is some text that varies. The cookery part 6 is "

You need an unblocked exit? You should try going north. You don't like doors? Why not try going west, that entranceway is unguarded.".
The description of r_4 is "[cookery part 0][cookery part 1][cookery part 2][cookery part 3][cookery part 4][cookery part 5][cookery part 6]".

The r_1 is mapped west of r_4.
The r_3 is mapped north of r_4.
Understand "shower" as r_1.
The internal name of r_1 is "shower".
The printed name of r_1 is "-= Shower =-".
The shower part 0 is some text that varies. The shower part 0 is "You arrive in a shower. A typical kind of place.

 You can see a cabinet.[if c_3 is open and there is something in the c_3] The cabinet contains [a list of things in the c_3].[end if]".
The shower part 1 is some text that varies. The shower part 1 is "[if c_3 is open and the c_3 contains nothing] What a letdown! The cabinet is empty![end if]".
The shower part 2 is some text that varies. The shower part 2 is " You rest your hand against a wall, but you miss the wall and fall onto a box. You shudder, but continue examining the room.[if c_4 is open and there is something in the c_4] The box contains [a list of things in the c_4].[end if]".
The shower part 3 is some text that varies. The shower part 3 is "[if c_4 is open and the c_4 contains nothing] The box is empty, what a horrible day![end if]".
The shower part 4 is some text that varies. The shower part 4 is " You can make out a bench. [if there is something on the s_6]You see [a list of things on the s_6] on the bench. Something scurries by right in the corner of your eye. Probably nothing.[end if]".
The shower part 5 is some text that varies. The shower part 5 is "[if there is nothing on the s_6]The bench appears to be empty. Oh! Why couldn't there just be stuff on it?[end if]".
The shower part 6 is some text that varies. The shower part 6 is "

 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The shower part 7 is some text that varies. The shower part 7 is " gate leading south. There is an unblocked exit to the east. You don't like doors? Why not try going north, that entranceway is unguarded.".
The description of r_1 is "[shower part 0][shower part 1][shower part 2][shower part 3][shower part 4][shower part 5][shower part 6][shower part 7]".

south of r_1 and north of r_0 is a door called d_0.
The r_2 is mapped north of r_1.
The r_4 is mapped east of r_1.
Understand "chamber" as r_7.
The internal name of r_7 is "chamber".
The printed name of r_7 is "-= Chamber =-".
The chamber part 0 is some text that varies. The chamber part 0 is "You've just shown up in a chamber. You decide to just list off a complete list of everything you see in the room, because hey, why not?



 There is [if d_6 is open]an open[otherwise]a closed[end if]".
The chamber part 1 is some text that varies. The chamber part 1 is " balsam door leading east. There is [if d_7 is open]an open[otherwise]a closed[end if]".
The chamber part 2 is some text that varies. The chamber part 2 is " stone door leading north. There is an exit to the west. Don't worry, it is unblocked.".
The description of r_7 is "[chamber part 0][chamber part 1][chamber part 2]".

The r_6 is mapped west of r_7.
north of r_7 and south of r_18 is a door called d_7.
east of r_7 and west of r_8 is a door called d_6.
Understand "dish-pit" as r_6.
The internal name of r_6 is "dish-pit".
The printed name of r_6 is "-= Dish-Pit =-".
The dish-pit part 0 is some text that varies. The dish-pit part 0 is "I am sorry to announce that you are now in the dish-pit.



There is an unblocked exit to the east. There is an exit to the north. Don't worry, it is unblocked.".
The description of r_6 is "[dish-pit part 0]".

The r_5 is mapped north of r_6.
The r_7 is mapped east of r_6.
Understand "restroom" as r_8.
The internal name of r_8 is "restroom".
The printed name of r_8 is "-= Restroom =-".
The restroom part 0 is some text that varies. The restroom part 0 is "I just think it's awesome that you're in a restroom now.

 You hear a noise behind you and spin around, but you can't see anything other than a shiny counter. [if there is something on the s_7]You see [a list of things on the s_7] on the counter.[end if]".
The restroom part 1 is some text that varies. The restroom part 1 is "[if there is nothing on the s_7]But there isn't a thing on it.[end if]".
The restroom part 2 is some text that varies. The restroom part 2 is " Hey, want to see a shelf? Look over there, a shelf. The shelf is chipped.[if there is something on the s_8] On the chipped shelf you make out [a list of things on the s_8].[end if]".
The restroom part 3 is some text that varies. The restroom part 3 is "[if there is nothing on the s_8] But the thing is empty, unfortunately.[end if]".
The restroom part 4 is some text that varies. The restroom part 4 is " You can make out a shiny board. [if there is something on the s_9]You see [a list of things on the s_9] on the board. Something scurries by right in the corner of your eye. Probably nothing.[end if]".
The restroom part 5 is some text that varies. The restroom part 5 is "[if there is nothing on the s_9]But oh no! there's nothing on this piece of garbage. Oh! Why couldn't there just be stuff on it?[end if]".
The restroom part 6 is some text that varies. The restroom part 6 is "

 There is [if d_2 is open]an open[otherwise]a closed[end if]".
The restroom part 7 is some text that varies. The restroom part 7 is " Henderson's limited edition passageway leading east. There is [if d_6 is open]an open[otherwise]a closed[end if]".
The restroom part 8 is some text that varies. The restroom part 8 is " balsam door leading west. There is [if d_5 is open]an open[otherwise]a closed[end if]".
The restroom part 9 is some text that varies. The restroom part 9 is " portal leading north.".
The description of r_8 is "[restroom part 0][restroom part 1][restroom part 2][restroom part 3][restroom part 4][restroom part 5][restroom part 6][restroom part 7][restroom part 8][restroom part 9]".

west of r_8 and east of r_7 is a door called d_6.
north of r_8 and south of r_12 is a door called d_5.
east of r_8 and west of r_9 is a door called d_2.
Understand "sauna" as r_0.
The internal name of r_0 is "sauna".
The printed name of r_0 is "-= Sauna =-".
The sauna part 0 is some text that varies. The sauna part 0 is "You find yourself in a sauna. An ordinary one.

 You see [if c_5 is locked]a locked[else if c_5 is open]an opened[otherwise]a closed[end if]".
The sauna part 1 is some text that varies. The sauna part 1 is " dresser here.[if c_5 is open and there is something in the c_5] The dresser contains [a list of things in the c_5].[end if]".
The sauna part 2 is some text that varies. The sauna part 2 is "[if c_5 is open and the c_5 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The sauna part 3 is some text that varies. The sauna part 3 is " You hear a noise behind you and spin around, but you can't see anything other than a safe.[if c_6 is open and there is something in the c_6] The safe contains [a list of things in the c_6]. Make a note of this, you might have to put stuff on or in it later on.[end if]".
The sauna part 4 is some text that varies. The sauna part 4 is "[if c_6 is open and the c_6 contains nothing] What a letdown! The safe is empty![end if]".
The sauna part 5 is some text that varies. The sauna part 5 is " You can see a counter. [if there is something on the s_10]You see [a list of things on the s_10] on the counter.[end if]".
The sauna part 6 is some text that varies. The sauna part 6 is "[if there is nothing on the s_10]However, the counter, like an empty counter, has nothing on it.[end if]".
The sauna part 7 is some text that varies. The sauna part 7 is "

 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The sauna part 8 is some text that varies. The sauna part 8 is " gate leading north. There is [if d_9 is open]an open[otherwise]a closed[end if]".
The sauna part 9 is some text that varies. The sauna part 9 is " gateway leading south.".
The description of r_0 is "[sauna part 0][sauna part 1][sauna part 2][sauna part 3][sauna part 4][sauna part 5][sauna part 6][sauna part 7][sauna part 8][sauna part 9]".

south of r_0 and north of r_5 is a door called d_9.
north of r_0 and south of r_1 is a door called d_0.
Understand "washroom" as r_19.
The internal name of r_19 is "washroom".
The printed name of r_19 is "-= Washroom =-".
The washroom part 0 is some text that varies. The washroom part 0 is "You're not going to believe this, but you've just entered a washroom. The room seems oddly familiar, as though it were only superficially different from the other rooms in the building.

 You make out [if c_7 is locked]a locked[else if c_7 is open]an opened[otherwise]a closed[end if]".
The washroom part 1 is some text that varies. The washroom part 1 is " ordinary looking trunk here.[if c_7 is open and there is something in the c_7] The trunk contains [a list of things in the c_7].[end if]".
The washroom part 2 is some text that varies. The washroom part 2 is "[if c_7 is open and the c_7 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The washroom part 3 is some text that varies. The washroom part 3 is " You can see a basket.[if c_8 is open and there is something in the c_8] The basket contains [a list of things in the c_8]. Wow, isn't TextWorld just the best?[end if]".
The washroom part 4 is some text that varies. The washroom part 4 is "[if c_8 is open and the c_8 contains nothing] The basket is empty, what a horrible day![end if]".
The washroom part 5 is some text that varies. The washroom part 5 is "

You don't like doors? Why not try going south, that entranceway is unblocked.".
The description of r_19 is "[washroom part 0][washroom part 1][washroom part 2][washroom part 3][washroom part 4][washroom part 5]".

The r_18 is mapped south of r_19.

The c_0 and the c_1 and the c_2 and the c_3 and the c_4 and the c_5 and the c_6 and the c_7 and the c_8 are containers.
The c_0 and the c_1 and the c_2 and the c_3 and the c_4 and the c_5 and the c_6 and the c_7 and the c_8 are privately-named.
The d_0 and the d_9 and the d_1 and the d_4 and the d_3 and the d_5 and the d_7 and the d_8 and the d_6 and the d_2 are doors.
The d_0 and the d_9 and the d_1 and the d_4 and the d_3 and the d_5 and the d_7 and the d_8 and the d_6 and the d_2 are privately-named.
The f_0 are foods.
The f_0 are privately-named.
The k_0 and the k_1 and the k_2 are keys.
The k_0 and the k_1 and the k_2 are privately-named.
The r_10 and the r_9 and the r_11 and the r_12 and the r_13 and the r_16 and the r_15 and the r_17 and the r_14 and the r_18 and the r_5 and the r_3 and the r_2 and the r_4 and the r_1 and the r_7 and the r_6 and the r_8 and the r_0 and the r_19 are rooms.
The r_10 and the r_9 and the r_11 and the r_12 and the r_13 and the r_16 and the r_15 and the r_17 and the r_14 and the r_18 and the r_5 and the r_3 and the r_2 and the r_4 and the r_1 and the r_7 and the r_6 and the r_8 and the r_0 and the r_19 are privately-named.
The s_0 and the s_1 and the s_10 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 and the s_9 are supporters.
The s_0 and the s_1 and the s_10 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 and the s_9 are privately-named.

The description of d_0 is "it's a durable gate [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_0 is "gate".
Understand "gate" as d_0.
The d_0 is open.
The description of d_9 is "it's a noble gateway [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_9 is "gateway".
Understand "gateway" as d_9.
The d_9 is open.
The description of d_1 is "it's a durable door [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_1 is "door".
Understand "door" as d_1.
The d_1 is open.
The description of d_4 is "it is what it is, a hatch [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_4 is "hatch".
Understand "hatch" as d_4.
The d_4 is open.
The description of d_3 is "The passageway looks rugged. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_3 is "passageway".
Understand "passageway" as d_3.
The d_3 is locked.
The description of d_5 is "it's a durable portal [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_5 is "portal".
Understand "portal" as d_5.
The d_5 is open.
The description of d_7 is "it is what it is, a stone door [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_7 is "stone door".
Understand "stone door" as d_7.
Understand "stone" as d_7.
Understand "door" as d_7.
The d_7 is open.
The description of d_8 is "it is what it is, a stone gate [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_8 is "stone gate".
Understand "stone gate" as d_8.
Understand "stone" as d_8.
Understand "gate" as d_8.
The d_8 is open.
The description of d_6 is "it is what it is, a balsam door [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_6 is "balsam door".
Understand "balsam door" as d_6.
Understand "balsam" as d_6.
Understand "door" as d_6.
The d_6 is closed.
The description of d_2 is "The Henderson's limited edition passageway looks sturdy. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_2 is "Henderson's limited edition passageway".
Understand "Henderson's limited edition passageway" as d_2.
Understand "Henderson's" as d_2.
Understand "limited" as d_2.
Understand "edition" as d_2.
Understand "passageway" as d_2.
The d_2 is locked.
The description of c_0 is "The drawer looks strong, and impossible to crack. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_0 is "drawer".
Understand "drawer" as c_0.
The c_0 is in r_15.
The c_0 is open.
The description of c_1 is "The case looks strong, and impossible to break. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_1 is "case".
Understand "case" as c_1.
The c_1 is in r_18.
The c_1 is open.
The description of c_2 is "The display looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_2 is "display".
Understand "display" as c_2.
The c_2 is in r_5.
The c_2 is open.
The description of c_3 is "The cabinet looks strong, and impossible to destroy. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_3 is "cabinet".
Understand "cabinet" as c_3.
The c_3 is in r_1.
The c_3 is locked.
The description of c_4 is "The box looks strong, and impossible to destroy. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_4 is "box".
Understand "box" as c_4.
The c_4 is in r_1.
The c_4 is closed.
The description of c_5 is "The dresser looks strong, and impossible to destroy. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_5 is "dresser".
Understand "dresser" as c_5.
The c_5 is in r_0.
The c_5 is locked.
The description of c_6 is "The safe looks strong, and impossible to destroy. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_6 is "safe".
Understand "safe" as c_6.
The c_6 is in r_0.
The c_6 is closed.
The description of c_7 is "The trunk looks strong, and impossible to crack. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_7 is "trunk".
Understand "trunk" as c_7.
The c_7 is in r_19.
The c_7 is open.
The description of c_8 is "The basket looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_8 is "basket".
Understand "basket" as c_8.
The c_8 is in r_19.
The c_8 is open.
The description of f_0 is "You couldn't pay me to eat that standard thing.".
The printed name of f_0 is "fondue".
Understand "fondue" as f_0.
The f_0 is in r_4.
The description of k_0 is "The metal of the passkey is satin.".
The printed name of k_0 is "passkey".
Understand "passkey" as k_0.
The k_0 is in r_10.
The description of k_1 is "The Henderson's limited edition key looks useful".
The printed name of k_1 is "Henderson's limited edition key".
Understand "Henderson's limited edition key" as k_1.
Understand "Henderson's" as k_1.
Understand "limited" as k_1.
Understand "edition" as k_1.
Understand "key" as k_1.
The k_1 is in r_11.
The matching key of the d_2 is the k_1.
The description of k_2 is "The key is cold to the touch".
The printed name of k_2 is "key".
Understand "key" as k_2.
The k_2 is in r_15.
The matching key of the d_3 is the k_2.
The description of s_0 is "The plate is reliable.".
The printed name of s_0 is "plate".
Understand "plate" as s_0.
The s_0 is in r_13.
The description of s_1 is "The shelf is reliable.".
The printed name of s_1 is "shelf".
Understand "shelf" as s_1.
The s_1 is in r_15.
The description of s_10 is "The counter is durable.".
The printed name of s_10 is "counter".
Understand "counter" as s_10.
The s_10 is in r_0.
The description of s_2 is "The table is durable.".
The printed name of s_2 is "table".
Understand "table" as s_2.
The s_2 is in r_2.
The description of s_3 is "The rack is stable.".
The printed name of s_3 is "rack".
Understand "rack" as s_3.
The s_3 is in r_4.
The description of s_4 is "The pan is solidly built.".
The printed name of s_4 is "pan".
Understand "pan" as s_4.
The s_4 is in r_4.
The description of s_5 is "The board is wobbly.".
The printed name of s_5 is "board".
Understand "board" as s_5.
The s_5 is in r_4.
The description of s_6 is "The bench is shaky.".
The printed name of s_6 is "bench".
Understand "bench" as s_6.
The s_6 is in r_1.
The description of s_7 is "The shiny counter is solid.".
The printed name of s_7 is "shiny counter".
Understand "shiny counter" as s_7.
Understand "shiny" as s_7.
Understand "counter" as s_7.
The s_7 is in r_8.
The description of s_8 is "The chipped shelf is unstable.".
The printed name of s_8 is "chipped shelf".
Understand "chipped shelf" as s_8.
Understand "chipped" as s_8.
Understand "shelf" as s_8.
The s_8 is in r_8.
The description of s_9 is "The shiny board is an unstable piece of junk.".
The printed name of s_9 is "shiny board".
Understand "shiny board" as s_9.
Understand "shiny" as s_9.
Understand "board" as s_9.
The s_9 is in r_8.


The player is in r_14.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with "go north / take key / go south / unlock passageway with key / open passageway / go south / go east / take Henderson's limited edition key / go south / unlock Henderson's limited edition passageway with Henderson's limited edition key / open Henderson's limited edition passageway / go west / open balsam door / go west / go west / go north / go north / go north / go east / take fondue"

Every turn:
	if quest0 completed is true:
		do nothing;
	else if The player carries the k_0:
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

The objective part 0 is some text that varies. The objective part 0 is "Who's got a virtual machine and is about to play through an exciting round of TextWorld? You do! Your task for today is to retrieve the fondue from the floor of the cookery.".

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

