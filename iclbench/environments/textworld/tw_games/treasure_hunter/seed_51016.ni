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


The r_1 and the r_7 and the r_11 and the r_12 and the r_15 and the r_14 and the r_16 and the r_17 and the r_18 and the r_2 and the r_3 and the r_4 and the r_6 and the r_0 and the r_9 and the r_10 and the r_13 and the r_19 and the r_5 and the r_8 are rooms.

Understand "workshop" as r_1.
The internal name of r_1 is "workshop".
The printed name of r_1 is "-= Workshop =-".
The workshop part 0 is some text that varies. The workshop part 0 is "You've just sauntered into a workshop.

 You can see a stand. The stand is typical.[if there is something on the s_0] On the stand you make out [a list of things on the s_0].[end if]".
The workshop part 1 is some text that varies. The workshop part 1 is "[if there is nothing on the s_0] But the thing is empty. You think about smashing the stand to bits, throwing the bits in the dump, etc, until you get bored.[end if]".
The workshop part 2 is some text that varies. The workshop part 2 is "

 There is [if d_10 is open]an open[otherwise]a closed[end if]".
The workshop part 3 is some text that varies. The workshop part 3 is " passageway leading west. You need an unguarded exit? You should try going east. There is an unblocked exit to the south.".
The description of r_1 is "[workshop part 0][workshop part 1][workshop part 2][workshop part 3]".

west of r_1 and east of r_7 is a door called d_10.
The r_0 is mapped south of r_1.
The r_2 is mapped east of r_1.
Understand "studio" as r_7.
The internal name of r_7 is "studio".
The printed name of r_7 is "-= Studio =-".
The studio part 0 is some text that varies. The studio part 0 is "You are in a studio. A standard kind of place.



 There is [if d_9 is open]an open[otherwise]a closed[end if]".
The studio part 1 is some text that varies. The studio part 1 is " cuboid door leading west. There is [if d_10 is open]an open[otherwise]a closed[end if]".
The studio part 2 is some text that varies. The studio part 2 is " passageway leading east. There is [if d_8 is open]an open[otherwise]a closed[end if]".
The studio part 3 is some text that varies. The studio part 3 is " stone door leading south.".
The description of r_7 is "[studio part 0][studio part 1][studio part 2][studio part 3]".

west of r_7 and east of r_9 is a door called d_9.
south of r_7 and north of r_8 is a door called d_8.
east of r_7 and west of r_1 is a door called d_10.
Understand "spare room" as r_11.
The internal name of r_11 is "spare room".
The printed name of r_11 is "-= Spare Room =-".
The spare room part 0 is some text that varies. The spare room part 0 is "You've entered a spare room.

 You see a rack. [if there is something on the s_1]You see [a list of things on the s_1] on the rack. Wow! Just like in the movies![end if]".
The spare room part 1 is some text that varies. The spare room part 1 is "[if there is nothing on the s_1]But the thing hasn't got anything on it.[end if]".
The spare room part 2 is some text that varies. The spare room part 2 is "

 There is [if d_7 is open]an open[otherwise]a closed[end if]".
The spare room part 3 is some text that varies. The spare room part 3 is " portal leading north. There is [if d_6 is open]an open[otherwise]a closed[end if]".
The spare room part 4 is some text that varies. The spare room part 4 is " hatch leading west.".
The description of r_11 is "[spare room part 0][spare room part 1][spare room part 2][spare room part 3][spare room part 4]".

west of r_11 and east of r_12 is a door called d_6.
north of r_11 and south of r_8 is a door called d_7.
Understand "pantry" as r_12.
The internal name of r_12 is "pantry".
The printed name of r_12 is "-= Pantry =-".
The pantry part 0 is some text that varies. The pantry part 0 is "You are in a pantry. It seems to be pretty usual here.

 You can make out [if c_0 is locked]a locked[else if c_0 is open]an opened[otherwise]a closed[end if]".
The pantry part 1 is some text that varies. The pantry part 1 is " case.[if c_0 is open and there is something in the c_0] The case contains [a list of things in the c_0].[end if]".
The pantry part 2 is some text that varies. The pantry part 2 is "[if c_0 is open and the c_0 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The pantry part 3 is some text that varies. The pantry part 3 is "

 There is [if d_5 is open]an open[otherwise]a closed[end if]".
The pantry part 4 is some text that varies. The pantry part 4 is " door leading south. There is [if d_6 is open]an open[otherwise]a closed[end if]".
The pantry part 5 is some text that varies. The pantry part 5 is " hatch leading east.".
The description of r_12 is "[pantry part 0][pantry part 1][pantry part 2][pantry part 3][pantry part 4][pantry part 5]".

south of r_12 and north of r_13 is a door called d_5.
east of r_12 and west of r_11 is a door called d_6.
Understand "dish-pit" as r_15.
The internal name of r_15 is "dish-pit".
The printed name of r_15 is "-= Dish-Pit =-".
The dish-pit part 0 is some text that varies. The dish-pit part 0 is "You're now in a dish-pit. You try to gain information on your surroundings by using a technique you call 'looking.'



 There is [if d_2 is open]an open[otherwise]a closed[end if]".
The dish-pit part 1 is some text that varies. The dish-pit part 1 is " cedar door leading east. There is [if d_3 is open]an open[otherwise]a closed[end if]".
The dish-pit part 2 is some text that varies. The dish-pit part 2 is " gateway leading west.".
The description of r_15 is "[dish-pit part 0][dish-pit part 1][dish-pit part 2]".

west of r_15 and east of r_14 is a door called d_3.
east of r_15 and west of r_16 is a door called d_2.
Understand "attic" as r_14.
The internal name of r_14 is "attic".
The printed name of r_14 is "-= Attic =-".
The attic part 0 is some text that varies. The attic part 0 is "You're now in an attic. The room is well lit.

 You rest your hand against a wall, but you miss the wall and fall onto a cabinet.[if c_1 is open and there is something in the c_1] The cabinet contains [a list of things in the c_1]. You check the price tag that's still affixed to the cabinet. Sixty dollars?! That's ridiculous! Did you ever meet my pal, they work for the mayor? I'm sure my friend could get you one of those for 85 bucks![end if]".
The attic part 1 is some text that varies. The attic part 1 is "[if c_1 is open and the c_1 contains nothing] What a letdown! The cabinet is empty![end if]".
The attic part 2 is some text that varies. The attic part 2 is " You smell a fine smell, and follow it to a workbench. [if there is something on the s_2]You see [a list of things on the s_2] on the workbench.[end if]".
The attic part 3 is some text that varies. The attic part 3 is "[if there is nothing on the s_2]But the thing hasn't got anything on it. Aw, here you were, all excited for there to be things on it![end if]".
The attic part 4 is some text that varies. The attic part 4 is "

 There is [if d_3 is open]an open[otherwise]a closed[end if]".
The attic part 5 is some text that varies. The attic part 5 is " gateway leading east. There is [if d_4 is open]an open[otherwise]a closed[end if]".
The attic part 6 is some text that varies. The attic part 6 is " gate leading north.".
The description of r_14 is "[attic part 0][attic part 1][attic part 2][attic part 3][attic part 4][attic part 5][attic part 6]".

north of r_14 and south of r_13 is a door called d_4.
east of r_14 and west of r_15 is a door called d_3.
Understand "bedchamber" as r_16.
The internal name of r_16 is "bedchamber".
The printed name of r_16 is "-= Bedchamber =-".
The bedchamber part 0 is some text that varies. The bedchamber part 0 is "You arrive in a bedchamber. An usual one.



 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The bedchamber part 1 is some text that varies. The bedchamber part 1 is " stone hatch leading east. There is [if d_2 is open]an open[otherwise]a closed[end if]".
The bedchamber part 2 is some text that varies. The bedchamber part 2 is " cedar door leading west.".
The description of r_16 is "[bedchamber part 0][bedchamber part 1][bedchamber part 2]".

west of r_16 and east of r_15 is a door called d_2.
east of r_16 and west of r_17 is a door called d_1.
Understand "vault" as r_17.
The internal name of r_17 is "vault".
The printed name of r_17 is "-= Vault =-".
The vault part 0 is some text that varies. The vault part 0 is "Well, here we are in a vault. You begin looking for stuff.

 [if c_2 is locked]A locked[else if c_2 is open]An open[otherwise]A closed[end if]".
The vault part 1 is some text that varies. The vault part 1 is " safe is in the room.[if c_2 is open and there is something in the c_2] The safe contains [a list of things in the c_2]. You shudder, but continue examining the room.[end if]".
The vault part 2 is some text that varies. The vault part 2 is "[if c_2 is open and the c_2 contains nothing] The safe is empty, what a horrible day![end if]".
The vault part 3 is some text that varies. The vault part 3 is "

 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The vault part 4 is some text that varies. The vault part 4 is " stone gateway leading east. There is [if d_1 is open]an open[otherwise]a closed[end if]".
The vault part 5 is some text that varies. The vault part 5 is " stone hatch leading west.".
The description of r_17 is "[vault part 0][vault part 1][vault part 2][vault part 3][vault part 4][vault part 5]".

west of r_17 and east of r_16 is a door called d_1.
east of r_17 and west of r_18 is a door called d_0.
Understand "recreation zone" as r_18.
The internal name of r_18 is "recreation zone".
The printed name of r_18 is "-= Recreation Zone =-".
The recreation zone part 0 is some text that varies. The recreation zone part 0 is "You've entered a recreation zone. Okay, just remember what you're here to do, and everything will go great.

 You see a recliner. [if there is something on the s_3]On the recliner you see [a list of things on the s_3].[end if]".
The recreation zone part 1 is some text that varies. The recreation zone part 1 is "[if there is nothing on the s_3]But oh no! there's nothing on this piece of trash.[end if]".
The recreation zone part 2 is some text that varies. The recreation zone part 2 is "

 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The recreation zone part 3 is some text that varies. The recreation zone part 3 is " stone gateway leading west. There is an unguarded exit to the north.".
The description of r_18 is "[recreation zone part 0][recreation zone part 1][recreation zone part 2][recreation zone part 3]".

west of r_18 and east of r_17 is a door called d_0.
The r_19 is mapped north of r_18.
Understand "steam room" as r_2.
The internal name of r_2 is "steam room".
The printed name of r_2 is "-= Steam Room =-".
The steam room part 0 is some text that varies. The steam room part 0 is "If you're wondering why everything seems so usual all of a sudden, it's because you've just walked into the steam room. You start to take note of what's in the room.

 You can see [if c_3 is locked]a locked[else if c_3 is open]an opened[otherwise]a closed[end if]".
The steam room part 1 is some text that varies. The steam room part 1 is " basket nearby.[if c_3 is open and there is something in the c_3] The basket contains [a list of things in the c_3].[end if]".
The steam room part 2 is some text that varies. The steam room part 2 is "[if c_3 is open and the c_3 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The steam room part 3 is some text that varies. The steam room part 3 is "

There is an exit to the east. Don't worry, it is unguarded. You don't like doors? Why not try going west, that entranceway is unguarded.".
The description of r_2 is "[steam room part 0][steam room part 1][steam room part 2][steam room part 3]".

The r_1 is mapped west of r_2.
The r_3 is mapped east of r_2.
Understand "canteen" as r_3.
The internal name of r_3 is "canteen".
The printed name of r_3 is "-= Canteen =-".
The canteen part 0 is some text that varies. The canteen part 0 is "You've just sauntered into a canteen.



There is an unblocked exit to the south. There is an exit to the west. Don't worry, it is unblocked.".
The description of r_3 is "[canteen part 0]".

The r_2 is mapped west of r_3.
The r_4 is mapped south of r_3.
Understand "study" as r_4.
The internal name of r_4 is "study".
The printed name of r_4 is "-= Study =-".
The study part 0 is some text that varies. The study part 0 is "Well, here we are in a study. Let's see what's in here.

 You can make out a table. The table is standard.[if there is something on the s_4] On the table you make out [a list of things on the s_4]. Hmmm... what else, what else?[end if]".
The study part 1 is some text that varies. The study part 1 is "[if there is nothing on the s_4] Looks like someone's already been here and taken everything off it, though.[end if]".
The study part 2 is some text that varies. The study part 2 is " You can see a desk. Make a note of this, you might have to put stuff on or in it later on. The desk is normal.[if there is something on the s_5] On the desk you can see [a list of things on the s_5]. There's something strange about this thing being here, but you don't have time to worry about that now.[end if]".
The study part 3 is some text that varies. The study part 3 is "[if there is nothing on the s_5] But there isn't a thing on it.[end if]".
The study part 4 is some text that varies. The study part 4 is "

You don't like doors? Why not try going north, that entranceway is unblocked. You don't like doors? Why not try going south, that entranceway is unguarded. There is an exit to the west. Don't worry, it is unblocked.".
The description of r_4 is "[study part 0][study part 1][study part 2][study part 3][study part 4]".

The r_6 is mapped west of r_4.
The r_5 is mapped south of r_4.
The r_3 is mapped north of r_4.
Understand "office" as r_6.
The internal name of r_6 is "office".
The printed name of r_6 is "-= Office =-".
The office part 0 is some text that varies. The office part 0 is "Look at that board! What does it say? It says Welcome to the office? Well that's cool.

 You see a shelf. [if there is something on the s_6]On the shelf you can see [a list of things on the s_6].[end if]".
The office part 1 is some text that varies. The office part 1 is "[if there is nothing on the s_6]But there isn't a thing on it.[end if]".
The office part 2 is some text that varies. The office part 2 is "

There is an unblocked exit to the east. There is an unguarded exit to the west.".
The description of r_6 is "[office part 0][office part 1][office part 2]".

The r_0 is mapped west of r_6.
The r_4 is mapped east of r_6.
Understand "cubicle" as r_0.
The internal name of r_0 is "cubicle".
The printed name of r_0 is "-= Cubicle =-".
The cubicle part 0 is some text that varies. The cubicle part 0 is "You have stumbled into the most normal of all possible cubicles.

 You can make out a trunk.[if c_4 is open and there is something in the c_4] The trunk contains [a list of things in the c_4]. Classic TextWorld.[end if]".
The cubicle part 1 is some text that varies. The cubicle part 1 is "[if c_4 is open and the c_4 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The cubicle part 2 is some text that varies. The cubicle part 2 is " You scan the room, seeing a box.[if c_5 is open and there is something in the c_5] The box contains [a list of things in the c_5].[end if]".
The cubicle part 3 is some text that varies. The cubicle part 3 is "[if c_5 is open and the c_5 contains nothing] The box is empty! This is the worst thing that could possibly happen, ever![end if]".
The cubicle part 4 is some text that varies. The cubicle part 4 is " You can see a mantle. The mantle is standard.[if there is something on the s_7] On the mantle you can make out [a list of things on the s_7].[end if]".
The cubicle part 5 is some text that varies. The cubicle part 5 is "[if there is nothing on the s_7] But oh no! there's nothing on this piece of junk.[end if]".
The cubicle part 6 is some text that varies. The cubicle part 6 is " You can make out a bookshelf. [if there is something on the s_8]You see [a list of things on the s_8] on the bookshelf.[end if]".
The cubicle part 7 is some text that varies. The cubicle part 7 is "[if there is nothing on the s_8]However, the bookshelf, like an empty bookshelf, has nothing on it.[end if]".
The cubicle part 8 is some text that varies. The cubicle part 8 is "

You need an unblocked exit? You should try going east. You don't like doors? Why not try going north, that entranceway is unblocked.".
The description of r_0 is "[cubicle part 0][cubicle part 1][cubicle part 2][cubicle part 3][cubicle part 4][cubicle part 5][cubicle part 6][cubicle part 7][cubicle part 8]".

The r_1 is mapped north of r_0.
The r_6 is mapped east of r_0.
Understand "kitchenette" as r_9.
The internal name of r_9 is "kitchenette".
The printed name of r_9 is "-= Kitchenette =-".
The kitchenette part 0 is some text that varies. The kitchenette part 0 is "Welcome to the kitchenette.

 [if c_6 is locked]A locked[else if c_6 is open]An open[otherwise]A closed[end if]".
The kitchenette part 1 is some text that varies. The kitchenette part 1 is " refrigerator is close by.[if c_6 is open and there is something in the c_6] The refrigerator contains [a list of things in the c_6]![end if]".
The kitchenette part 2 is some text that varies. The kitchenette part 2 is "[if c_6 is open and the c_6 contains nothing] The refrigerator is empty! What a waste of a day![end if]".
The kitchenette part 3 is some text that varies. The kitchenette part 3 is " [if c_7 is locked]A locked[else if c_7 is open]An open[otherwise]A closed[end if]".
The kitchenette part 4 is some text that varies. The kitchenette part 4 is " American limited edition locker is close by.[if c_7 is open and there is something in the c_7] The American limited edition locker contains [a list of things in the c_7].[end if]".
The kitchenette part 5 is some text that varies. The kitchenette part 5 is "[if c_7 is open and the c_7 contains nothing] What a letdown! The American limited edition locker is empty![end if]".
The kitchenette part 6 is some text that varies. The kitchenette part 6 is "

 There is [if d_9 is open]an open[otherwise]a closed[end if]".
The kitchenette part 7 is some text that varies. The kitchenette part 7 is " cuboid door leading east. You need an unguarded exit? You should try going south.".
The description of r_9 is "[kitchenette part 0][kitchenette part 1][kitchenette part 2][kitchenette part 3][kitchenette part 4][kitchenette part 5][kitchenette part 6][kitchenette part 7]".

The r_10 is mapped south of r_9.
east of r_9 and west of r_7 is a door called d_9.
Understand "cellar" as r_10.
The internal name of r_10 is "cellar".
The printed name of r_10 is "-= Cellar =-".
The cellar part 0 is some text that varies. The cellar part 0 is "You arrive in a cellar. A standard one.

 You see a non-euclidean safe.[if c_8 is open and there is something in the c_8] The non-euclidean safe contains [a list of things in the c_8]. There's something strange about this being here, but you can't put your finger on it.[end if]".
The cellar part 1 is some text that varies. The cellar part 1 is "[if c_8 is open and the c_8 contains nothing] The non-euclidean safe is empty! This is the worst thing that could possibly happen, ever![end if]".
The cellar part 2 is some text that varies. The cellar part 2 is " You make out [if c_9 is locked]a locked[else if c_9 is open]an opened[otherwise]a closed[end if]".
The cellar part 3 is some text that varies. The cellar part 3 is " locker.[if c_9 is open and there is something in the c_9] The locker contains [a list of things in the c_9].[end if]".
The cellar part 4 is some text that varies. The cellar part 4 is "[if c_9 is open and the c_9 contains nothing] The locker is empty! This is the worst thing that could possibly happen, ever![end if]".
The cellar part 5 is some text that varies. The cellar part 5 is "

There is an unguarded exit to the north.".
The description of r_10 is "[cellar part 0][cellar part 1][cellar part 2][cellar part 3][cellar part 4][cellar part 5]".

The r_9 is mapped north of r_10.
Understand "salon" as r_13.
The internal name of r_13 is "salon".
The printed name of r_13 is "-= Salon =-".
The salon part 0 is some text that varies. The salon part 0 is "You are in a salon. An ordinary one.

 Hey, want to see a chest? Look over there, a chest.[if c_10 is open and there is something in the c_10] The chest contains [a list of things in the c_10].[end if]".
The salon part 1 is some text that varies. The salon part 1 is "[if c_10 is open and the c_10 contains nothing] The chest is empty! This is the worst thing that could possibly happen, ever![end if]".
The salon part 2 is some text that varies. The salon part 2 is "

 There is [if d_5 is open]an open[otherwise]a closed[end if]".
The salon part 3 is some text that varies. The salon part 3 is " door leading north. There is [if d_4 is open]an open[otherwise]a closed[end if]".
The salon part 4 is some text that varies. The salon part 4 is " gate leading south.".
The description of r_13 is "[salon part 0][salon part 1][salon part 2][salon part 3][salon part 4]".

south of r_13 and north of r_14 is a door called d_4.
north of r_13 and south of r_12 is a door called d_5.
Understand "closet" as r_19.
The internal name of r_19 is "closet".
The printed name of r_19 is "-= Closet =-".
The closet part 0 is some text that varies. The closet part 0 is "You find yourself in a closet. A standard kind of place.



There is an unguarded exit to the south.".
The description of r_19 is "[closet part 0]".

The r_18 is mapped south of r_19.
Understand "bar" as r_5.
The internal name of r_5 is "bar".
The printed name of r_5 is "-= Bar =-".
The bar part 0 is some text that varies. The bar part 0 is "You've just sauntered into a bar.



There is an unguarded exit to the north.".
The description of r_5 is "[bar part 0]".

The r_4 is mapped north of r_5.
Understand "scullery" as r_8.
The internal name of r_8 is "scullery".
The printed name of r_8 is "-= Scullery =-".
The scullery part 0 is some text that varies. The scullery part 0 is "You've entered a scullery.



 There is [if d_7 is open]an open[otherwise]a closed[end if]".
The scullery part 1 is some text that varies. The scullery part 1 is " portal leading south. There is [if d_8 is open]an open[otherwise]a closed[end if]".
The scullery part 2 is some text that varies. The scullery part 2 is " stone door leading north.".
The description of r_8 is "[scullery part 0][scullery part 1][scullery part 2]".

south of r_8 and north of r_11 is a door called d_7.
north of r_8 and south of r_7 is a door called d_8.

The c_0 and the c_1 and the c_10 and the c_2 and the c_3 and the c_4 and the c_5 and the c_6 and the c_7 and the c_8 and the c_9 are containers.
The c_0 and the c_1 and the c_10 and the c_2 and the c_3 and the c_4 and the c_5 and the c_6 and the c_7 and the c_8 and the c_9 are privately-named.
The d_10 and the d_6 and the d_7 and the d_5 and the d_4 and the d_3 and the d_2 and the d_1 and the d_0 and the d_8 and the d_9 are doors.
The d_10 and the d_6 and the d_7 and the d_5 and the d_4 and the d_3 and the d_2 and the d_1 and the d_0 and the d_8 and the d_9 are privately-named.
The k_2 and the k_0 and the k_1 are keys.
The k_2 and the k_0 and the k_1 are privately-named.
The o_1 and the o_0 are object-likes.
The o_1 and the o_0 are privately-named.
The r_1 and the r_7 and the r_11 and the r_12 and the r_15 and the r_14 and the r_16 and the r_17 and the r_18 and the r_2 and the r_3 and the r_4 and the r_6 and the r_0 and the r_9 and the r_10 and the r_13 and the r_19 and the r_5 and the r_8 are rooms.
The r_1 and the r_7 and the r_11 and the r_12 and the r_15 and the r_14 and the r_16 and the r_17 and the r_18 and the r_2 and the r_3 and the r_4 and the r_6 and the r_0 and the r_9 and the r_10 and the r_13 and the r_19 and the r_5 and the r_8 are privately-named.
The s_0 and the s_1 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 are supporters.
The s_0 and the s_1 and the s_2 and the s_3 and the s_4 and the s_5 and the s_6 and the s_7 and the s_8 are privately-named.

The description of d_10 is "it is what it is, a passageway [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_10 is "passageway".
Understand "passageway" as d_10.
The d_10 is open.
The description of d_6 is "it's a noble hatch [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_6 is "hatch".
Understand "hatch" as d_6.
The d_6 is open.
The description of d_7 is "The portal looks imposing. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_7 is "portal".
Understand "portal" as d_7.
The d_7 is closed.
The description of d_5 is "The door looks imposing. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_5 is "door".
Understand "door" as d_5.
The d_5 is closed.
The description of d_4 is "The gate looks noble. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_4 is "gate".
Understand "gate" as d_4.
The d_4 is open.
The description of d_3 is "The gateway looks stuffy. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_3 is "gateway".
Understand "gateway" as d_3.
The d_3 is open.
The description of d_2 is "it is what it is, a cedar door [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_2 is "cedar door".
Understand "cedar door" as d_2.
Understand "cedar" as d_2.
Understand "door" as d_2.
The d_2 is open.
The description of d_1 is "The stone hatch looks stuffy. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_1 is "stone hatch".
Understand "stone hatch" as d_1.
Understand "stone" as d_1.
Understand "hatch" as d_1.
The d_1 is open.
The description of d_0 is "it's an ominous gateway [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_0 is "stone gateway".
Understand "stone gateway" as d_0.
Understand "stone" as d_0.
Understand "gateway" as d_0.
The d_0 is open.
The description of d_8 is "it is what it is, a stone door [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_8 is "stone door".
Understand "stone door" as d_8.
Understand "stone" as d_8.
Understand "door" as d_8.
The d_8 is open.
The description of d_9 is "it's an ominous cuboid door [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_9 is "cuboid door".
Understand "cuboid door" as d_9.
Understand "cuboid" as d_9.
Understand "door" as d_9.
The d_9 is locked.
The description of c_0 is "The case looks strong, and impossible to destroy. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_0 is "case".
Understand "case" as c_0.
The c_0 is in r_12.
The c_0 is locked.
The description of c_1 is "The cabinet looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_1 is "cabinet".
Understand "cabinet" as c_1.
The c_1 is in r_14.
The c_1 is locked.
The description of c_10 is "The chest looks strong, and impossible to destroy. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_10 is "chest".
Understand "chest" as c_10.
The c_10 is in r_13.
The c_10 is open.
The description of c_2 is "The safe looks strong, and impossible to crack. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_2 is "safe".
Understand "safe" as c_2.
The c_2 is in r_17.
The c_2 is locked.
The description of c_3 is "The basket looks strong, and impossible to destroy. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_3 is "basket".
Understand "basket" as c_3.
The c_3 is in r_2.
The c_3 is open.
The description of c_4 is "The trunk looks strong, and impossible to crack. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_4 is "trunk".
Understand "trunk" as c_4.
The c_4 is in r_0.
The c_4 is open.
The description of c_5 is "The box looks strong, and impossible to break. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_5 is "box".
Understand "box" as c_5.
The c_5 is in r_0.
The c_5 is closed.
The description of c_6 is "The refrigerator looks strong, and impossible to crack. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_6 is "refrigerator".
Understand "refrigerator" as c_6.
The c_6 is in r_9.
The c_6 is locked.
The description of c_7 is "The American limited edition locker looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_7 is "American limited edition locker".
Understand "American limited edition locker" as c_7.
Understand "American" as c_7.
Understand "limited" as c_7.
Understand "edition" as c_7.
Understand "locker" as c_7.
The c_7 is in r_9.
The c_7 is locked.
The description of c_8 is "The non-euclidean safe looks strong, and impossible to crack. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_8 is "non-euclidean safe".
Understand "non-euclidean safe" as c_8.
Understand "non-euclidean" as c_8.
Understand "safe" as c_8.
The c_8 is in r_10.
The c_8 is locked.
The description of c_9 is "The locker looks strong, and impossible to crack. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_9 is "locker".
Understand "locker" as c_9.
The c_9 is in r_10.
The c_9 is open.
The description of k_2 is "The non-euclidean keycard is heavier than it looks.".
The printed name of k_2 is "non-euclidean keycard".
Understand "non-euclidean keycard" as k_2.
Understand "non-euclidean" as k_2.
Understand "keycard" as k_2.
The k_2 is in r_9.
The matching key of the c_8 is the k_2.
The description of o_1 is "The worm is well-used.".
The printed name of o_1 is "worm".
Understand "worm" as o_1.
The o_1 is in r_1.
The description of s_0 is "The stand is unstable.".
The printed name of s_0 is "stand".
Understand "stand" as s_0.
The s_0 is in r_1.
The description of s_1 is "The rack is shaky.".
The printed name of s_1 is "rack".
Understand "rack" as s_1.
The s_1 is in r_11.
The description of s_2 is "The workbench is undependable.".
The printed name of s_2 is "workbench".
Understand "workbench" as s_2.
The s_2 is in r_14.
The description of s_3 is "The recliner is shaky.".
The printed name of s_3 is "recliner".
Understand "recliner" as s_3.
The s_3 is in r_18.
The description of s_4 is "The table is balanced.".
The printed name of s_4 is "table".
Understand "table" as s_4.
The s_4 is in r_4.
The description of s_5 is "The desk is wobbly.".
The printed name of s_5 is "desk".
Understand "desk" as s_5.
The s_5 is in r_4.
The description of s_6 is "The shelf is solid.".
The printed name of s_6 is "shelf".
Understand "shelf" as s_6.
The s_6 is in r_6.
The description of s_7 is "The mantle is balanced.".
The printed name of s_7 is "mantle".
Understand "mantle" as s_7.
The s_7 is in r_0.
The description of s_8 is "The bookshelf is solid.".
The printed name of s_8 is "bookshelf".
Understand "bookshelf" as s_8.
The s_8 is in r_0.
The description of k_0 is "The cuboid key looks useful".
The printed name of k_0 is "cuboid key".
Understand "cuboid key" as k_0.
Understand "cuboid" as k_0.
Understand "key" as k_0.
The k_0 is in the c_7.
The matching key of the d_9 is the k_0.
The description of k_1 is "The metal of the American limited edition latchkey is hammered.".
The printed name of k_1 is "American limited edition latchkey".
Understand "American limited edition latchkey" as k_1.
Understand "American" as k_1.
Understand "limited" as k_1.
Understand "edition" as k_1.
Understand "latchkey" as k_1.
The k_1 is in the c_8.
The matching key of the c_7 is the k_1.
The description of o_0 is "The lamp is antiquated.".
The printed name of o_0 is "lamp".
Understand "lamp" as o_0.
The o_0 is in the c_10.


The player is in r_10.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with "go north / take non-euclidean keycard / go south / unlock non-euclidean safe with non-euclidean keycard / open non-euclidean safe / take American limited edition latchkey from non-euclidean safe / go north / unlock American limited edition locker with American limited edition latchkey / open American limited edition locker / take cuboid key from American limited edition locker / unlock cuboid door with cuboid key / open cuboid door / go east / go south / open portal / go south / go west / open door / go south / take lamp from chest"

Every turn:
	if quest0 completed is true:
		do nothing;
	else if The player carries the o_1:
		end the story; [Lost]
	else if The player is in r_13 and The c_10 is in r_13 and The c_10 is open and The player carries the o_0:
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

The objective part 0 is some text that varies. The objective part 0 is "Hey, thanks for coming over to TextWorld! Please pick up the lamp from the chest in the salon.".

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

