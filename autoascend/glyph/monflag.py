# /include/monflag.h
MS_SILENT = 0     #/* makes no sound */
MS_BARK = 1       #/* if full moon, may howl */
MS_MEW = 2        #/* mews or hisses */
MS_ROAR = 3       #/* roars */
MS_GROWL = 4      #/* growls */
MS_SQEEK = 5      #/* squeaks, as a rodent */
MS_SQAWK = 6      #/* squawks, as a bird */
MS_HISS = 7       #/* hisses */
MS_BUZZ = 8       #/* buzzes (killer bee) */
MS_GRUNT = 9      #/* grunts (or speaks own language) */
MS_NEIGH = 10     #/* neighs, as an equine */
MS_WAIL = 11      #/* wails, as a tortured soul */
MS_GURGLE = 12    #/* gurgles, as liquid or through saliva */
MS_BURBLE = 13    #/* burbles (jabberwock) */
MS_ANIMAL = 13    #/* up to here are animal noises */
MS_SHRIEK = 15    #/* wakes up others */
MS_BONES = 16     #/* rattles bones (skeleton) */
MS_LAUGH = 17     #/* grins, smiles, giggles, and laughs */
MS_MUMBLE = 18    #/* says something or other */
MS_IMITATE = 19   #/* imitates others (leocrotta) */
MS_ORC = MS_GRUNT #/* intelligent brutes */
MS_HUMANOID = 20  #/* generic traveling companion */
MS_ARREST = 21    #/* "Stop in the name of the law!" (Kops) */
MS_SOLDIER = 22   #/* army and watchmen expressions */
MS_GUARD = 23     #/* "Please drop that gold and follow me." */
MS_DJINNI = 24    #/* "Thank you for freeing me!" */
MS_NURSE = 25     #/* "Take off your shirt, please." */
MS_SEDUCE = 26    #/* "Hello, sailor." (Nymphs) */
MS_VAMPIRE = 27   #/* vampiric seduction, Vlad's exclamations */
MS_BRIBE = 28     #/* asks for money, or berates you */
MS_CUSS = 29      #/* berates (demons) or intimidates (Wiz) */
MS_RIDER = 30     #/* astral level special monsters */
MS_LEADER = 31    #/* your class leader */
MS_NEMESIS = 32   #/* your nemesis */
MS_GUARDIAN = 33  #/* your leader's guards */
MS_SELL = 34      #/* demand payment, complain about shoplifters */
MS_ORACLE = 35    #/* do a consultation */
MS_PRIEST = 36    #/* ask for contribution; do cleansing */
MS_SPELL = 37     #/* spellcaster not matching any of the above */
MS_WERE = 38      #/* lycanthrope in human form */
MS_BOAST = 39     #/* giants */

MR_FIRE = 0x01   #/* resists fire */
MR_COLD = 0x02   #/* resists cold */
MR_SLEEP = 0x04  #/* resists sleep */
MR_DISINT = 0x08 #/* resists disintegration */
MR_ELEC = 0x10   #/* resists electricity */
MR_POISON = 0x20 #/* resists poison */
MR_ACID = 0x40   #/* resists acid */
MR_STONE = 0x80  #/* resists petrification */
#/* other resistances: magic, sickness */
#/* other conveyances: teleport, teleport control, telepathy */

#/* individual resistances */
MR2_SEE_INVIS = 0x0100 #/* see invisible */
MR2_LEVITATE = 0x0200  #/* levitation */
MR2_WATERWALK = 0x0400 #/* water walking */
MR2_MAGBREATH = 0x0800 #/* magical breathing */
MR2_DISPLACED = 0x1000 #/* displaced */
MR2_STRENGTH = 0x2000  #/* gauntlets of power */
MR2_FUMBLING = 0x4000  #/* clumsy */

M1_FLY = 0x00000001         #/* can fly or float */
M1_SWIM = 0x00000002        #/* can traverse water */
M1_AMORPHOUS = 0x00000004   #/* can flow under doors */
M1_WALLWALK = 0x00000008    #/* can phase thru rock */
M1_CLING = 0x00000010       #/* can cling to ceiling */
M1_TUNNEL = 0x00000020      #/* can tunnel thru rock */
M1_NEEDPICK = 0x00000040    #/* needs pick to tunnel */
M1_CONCEAL = 0x00000080     #/* hides under objects */
M1_HIDE = 0x00000100        #/* mimics, blends in with ceiling */
M1_AMPHIBIOUS = 0x00000200  #/* can survive underwater */
M1_BREATHLESS = 0x00000400  #/* doesn't need to breathe */
M1_NOTAKE = 0x00000800      #/* cannot pick up objects */
M1_NOEYES = 0x00001000      #/* no eyes to gaze into or blind */
M1_NOHANDS = 0x00002000     #/* no hands to handle things */
M1_NOLIMBS = 0x00006000     #/* no arms/legs to kick/wear on */
M1_NOHEAD = 0x00008000      #/* no head to behead */
M1_MINDLESS = 0x00010000    #/* has no mind--golem, zombie, mold */
M1_HUMANOID = 0x00020000    #/* has humanoid head/arms/torso */
M1_ANIMAL = 0x00040000      #/* has animal body */
M1_SLITHY = 0x00080000      #/* has serpent body */
M1_UNSOLID = 0x00100000     #/* has no solid or liquid body */
M1_THICK_HIDE = 0x00200000  #/* has thick hide or scales */
M1_OVIPAROUS = 0x00400000   #/* can lay eggs */
M1_REGEN = 0x00800000       #/* regenerates hit points */
M1_SEE_INVIS = 0x01000000   #/* can see invisible creatures */
M1_TPORT = 0x02000000       #/* can teleport */
M1_TPORT_CNTRL = 0x04000000 #/* controls where it teleports to */
M1_ACID = 0x08000000        #/* acidic to eat */
M1_POIS = 0x10000000        #/* poisonous to eat */
M1_CARNIVORE = 0x20000000   #/* eats corpses */
M1_HERBIVORE = 0x40000000   #/* eats fruits */
M1_OMNIVORE = 0x60000000    #/* eats both */
#ifdef NHSTDC
#define M1_METALLIVORE 0x80000000UL /* eats metal */
#else
M1_METALLIVORE = 0x80000000 #/* eats metal */
#endif

M2_NOPOLY = 0x00000001       #/* players mayn't poly into one */
M2_UNDEAD = 0x00000002       #/* is walking dead */
M2_WERE = 0x00000004         #/* is a lycanthrope */
M2_HUMAN = 0x00000008        #/* is a human */
M2_ELF = 0x00000010          #/* is an elf */
M2_DWARF = 0x00000020        #/* is a dwarf */
M2_GNOME = 0x00000040        #/* is a gnome */
M2_ORC = 0x00000080          #/* is an orc */
M2_DEMON = 0x00000100        #/* is a demon */
M2_MERC = 0x00000200         #/* is a guard or soldier */
M2_LORD = 0x00000400         #/* is a lord to its kind */
M2_PRINCE = 0x00000800       #/* is an overlord to its kind */
M2_MINION = 0x00001000       #/* is a minion of a deity */
M2_GIANT = 0x00002000        #/* is a giant */
M2_SHAPESHIFTER = 0x00004000 #/* is a shapeshifting species */
M2_MALE = 0x00010000         #/* always male */
M2_FEMALE = 0x00020000       #/* always female */
M2_NEUTER = 0x00040000       #/* neither male nor female */
M2_PNAME = 0x00080000        #/* monster name is a proper name */
M2_HOSTILE = 0x00100000      #/* always starts hostile */
M2_PEACEFUL = 0x00200000     #/* always starts peaceful */
M2_DOMESTIC = 0x00400000     #/* can be tamed by feeding */
M2_WANDER = 0x00800000       #/* wanders randomly */
M2_STALK = 0x01000000        #/* follows you to other levels */
M2_NASTY = 0x02000000        #/* extra-nasty monster (more xp) */
M2_STRONG = 0x04000000       #/* strong (or big) monster */
M2_ROCKTHROW = 0x08000000    #/* throws boulders */
M2_GREEDY = 0x10000000       #/* likes gold */
M2_JEWELS = 0x20000000       #/* likes gems */
M2_COLLECT = 0x40000000      #/* picks up weapons and food */
#ifdef NHSTDC
#define M2_MAGIC 0x80000000UL /* picks up magic items */
#else
M2_MAGIC = 0x80000000 #/* picks up magic items */
#endif

M3_WANTSAMUL = 0x0001 #/* would like to steal the amulet */
M3_WANTSBELL = 0x0002 #/* wants the bell */
M3_WANTSBOOK = 0x0004 #/* wants the book */
M3_WANTSCAND = 0x0008 #/* wants the candelabrum */
M3_WANTSARTI = 0x0010 #/* wants the quest artifact */
M3_WANTSALL = 0x001f  #/* wants any major artifact */
M3_WAITFORU = 0x0040  #/* waits to see you or get attacked */
M3_CLOSE = 0x0080     #/* lets you close unless attacked */

M3_COVETOUS = 0x001f #/* wants something */
M3_WAITMASK = 0x00c0 #/* waiting... */

#/* Infravision is currently implemented for players only */
M3_INFRAVISION = 0x0100  #/* has infravision */
M3_INFRAVISIBLE = 0x0200 #/* visible by infravision */

M3_DISPLACES = 0x0400 #/* moves monsters out of its way */

MZ_TINY = 0          #/* < 2' */
MZ_SMALL = 1         #/* 2-4' */
MZ_MEDIUM = 2        #/* 4-7' */
MZ_HUMAN = MZ_MEDIUM #/* human-sized */
MZ_LARGE = 3         #/* 7-12' */
MZ_HUGE = 4          #/* 12-25' */
MZ_GIGANTIC = 7      #/* off the scale */

#/* Monster races -- must stay within ROLE_RACEMASK */
#/* Eventually this may become its own field */
MH_HUMAN = M2_HUMAN
MH_ELF = M2_ELF
MH_DWARF = M2_DWARF
MH_GNOME = M2_GNOME
MH_ORC = M2_ORC

#/* for mons[].geno (constant during game) */
G_UNIQ = 0x1000     #/* generated only once */
G_NOHELL = 0x0800   #/* not generated in "hell" */
G_HELL = 0x0400     #/* generated only in "hell" */
G_NOGEN = 0x0200    #/* generated only specially */
G_SGROUP = 0x0080   #/* appear in small groups normally */
G_LGROUP = 0x0040   #/* appear in large groups normally */
G_GENO = 0x0020     #/* can be genocided */
G_NOCORPSE = 0x0010 #/* no corpse left ever */
G_FREQ = 0x0007     #/* creation frequency mask */

#/* for mvitals[].mvflags (variant during game), along with G_NOCORPSE */
G_KNOWN = 0x0004 #/* have been encountered */
G_GENOD = 0x0002 #/* have been genocided */
G_EXTINCT = 0x0001 #/* have been extinguished as population control */
G_GONE = (G_GENOD | G_EXTINCT)
MV_KNOWS_EGG = 0x0008 #/* player recognizes egg of this monster type */

S_ANT        =  1#, /* a */
S_BLOB       =  2#, /* b */
S_COCKATRICE =  3#, /* c */
S_DOG        =  4#, /* d */
S_EYE        =  5#, /* e */
S_FELINE     =  6#, /* f: cats */
S_GREMLIN    =  7#, /* g */
S_HUMANOID   =  8#, /* h: small humanoids: hobbit, dwarf */
S_IMP        =  9#, /* i: minor demons */
S_JELLY      = 10#, /* j */
S_KOBOLD     = 11#, /* k */
S_LEPRECHAUN = 12#, /* l */
S_MIMIC      = 13#, /* m */
S_NYMPH      = 14#, /* n */
S_ORC        = 15#, /* o */
S_PIERCER    = 16#, /* p */
S_QUADRUPED  = 17#, /* q: excludes horses */
S_RODENT     = 18#, /* r */
S_SPIDER     = 19#, /* s */
S_TRAPPER    = 20#, /* t */
S_UNICORN    = 21#, /* u: includes horses */
S_VORTEX     = 22#, /* v */
S_WORM       = 23#, /* w */
S_XAN        = 24#, /* x */
S_LIGHT      = 25#, /* y: yellow light, black light */
S_ZRUTY      = 26#, /* z */
S_ANGEL      = 27#, /* A */
S_BAT        = 28#, /* B */
S_CENTAUR    = 29#, /* C */
S_DRAGON     = 30#, /* D */
S_ELEMENTAL  = 31#, /* E: includes invisible stalker */
S_FUNGUS     = 32#, /* F */
S_GNOME      = 33#, /* G */
S_GIANT      = 34#, /* H: large humanoid: giant, ettin, minotaur */
S_invisible  = 35#, /* I: non-class present in def_monsyms[] */
S_JABBERWOCK = 36#, /* J */
S_KOP        = 37#, /* K */
S_LICH       = 38#, /* L */
S_MUMMY      = 39#, /* M */
S_NAGA       = 40#, /* N */
S_OGRE       = 41#, /* O */
S_PUDDING    = 42#, /* P */
S_QUANTMECH  = 43#, /* Q */
S_RUSTMONST  = 44#, /* R */
S_SNAKE      = 45#, /* S */
S_TROLL      = 46#, /* T */
S_UMBER      = 47#, /* U: umber hulk */
S_VAMPIRE    = 48#, /* V */
S_WRAITH     = 49#, /* W */
S_XORN       = 50#, /* X */
S_YETI       = 51#, /* Y: includes owlbear, monkey */
S_ZOMBIE     = 52#, /* Z */
S_HUMAN      = 53#, /* @ */
S_GHOST      = 54#, /* <space> */
S_GOLEM      = 55#, /* ' */
S_DEMON      = 56#, /* & */
S_EEL        = 57#, /* ; (fish) */
S_LIZARD     = 58#, /* : (reptiles) */

S_WORM_TAIL  = 59#, /* ~ */
S_MIMIC_DEF  = 60#, /* ] */

MAXMCLASSES  = 61#  /* number of monster classes */
