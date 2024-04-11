import nle.nethack as nh


S_stone     =nh.GLYPH_CMAP_OFF+  0#
S_vwall     =nh.GLYPH_CMAP_OFF+  1#
S_hwall     =nh.GLYPH_CMAP_OFF+  2#
S_tlcorn    =nh.GLYPH_CMAP_OFF+  3#
S_trcorn    =nh.GLYPH_CMAP_OFF+  4#
S_blcorn    =nh.GLYPH_CMAP_OFF+  5#
S_brcorn    =nh.GLYPH_CMAP_OFF+  6#
S_crwall    =nh.GLYPH_CMAP_OFF+  7#
S_tuwall    =nh.GLYPH_CMAP_OFF+  8#
S_tdwall    =nh.GLYPH_CMAP_OFF+  9#
S_tlwall    =nh.GLYPH_CMAP_OFF+ 10#
S_trwall    =nh.GLYPH_CMAP_OFF+ 11#
S_ndoor     =nh.GLYPH_CMAP_OFF+ 12#
S_vodoor    =nh.GLYPH_CMAP_OFF+ 13#
S_hodoor    =nh.GLYPH_CMAP_OFF+ 14#
S_vcdoor    =nh.GLYPH_CMAP_OFF+ 15# /* closed door, vertical wall */
S_hcdoor    =nh.GLYPH_CMAP_OFF+ 16# /* closed door, horizontal wall */
S_bars      =nh.GLYPH_CMAP_OFF+ 17# /* KMH -- iron bars */
S_tree      =nh.GLYPH_CMAP_OFF+ 18# /* KMH */
S_room      =nh.GLYPH_CMAP_OFF+ 19#
S_darkroom  =nh.GLYPH_CMAP_OFF+ 20#
S_corr      =nh.GLYPH_CMAP_OFF+ 21#
S_litcorr   =nh.GLYPH_CMAP_OFF+ 22#
S_upstair   =nh.GLYPH_CMAP_OFF+ 23#
S_dnstair   =nh.GLYPH_CMAP_OFF+ 24#
S_upladder  =nh.GLYPH_CMAP_OFF+ 25#
S_dnladder  =nh.GLYPH_CMAP_OFF+ 26#
S_altar     =nh.GLYPH_CMAP_OFF+ 27#
S_grave     =nh.GLYPH_CMAP_OFF+ 28#
S_throne    =nh.GLYPH_CMAP_OFF+ 29#
S_sink      =nh.GLYPH_CMAP_OFF+ 30#
S_fountain  =nh.GLYPH_CMAP_OFF+ 31#
S_pool      =nh.GLYPH_CMAP_OFF+ 32#
S_ice       =nh.GLYPH_CMAP_OFF+ 33#
S_lava      =nh.GLYPH_CMAP_OFF+ 34#
S_vodbridge =nh.GLYPH_CMAP_OFF+ 35#
S_hodbridge =nh.GLYPH_CMAP_OFF+ 36#
S_vcdbridge =nh.GLYPH_CMAP_OFF+ 37# /* closed drawbridge, vertical wall */
S_hcdbridge =nh.GLYPH_CMAP_OFF+ 38# /* closed drawbridge, horizontal wall */
S_air       =nh.GLYPH_CMAP_OFF+ 39#
S_cloud     =nh.GLYPH_CMAP_OFF+ 40#
S_water     =nh.GLYPH_CMAP_OFF+ 41#

#/* end dungeon characters, begin traps */

S_arrow_trap           =nh.GLYPH_CMAP_OFF+ 42#
S_dart_trap            =nh.GLYPH_CMAP_OFF+ 43#
S_falling_rock_trap    =nh.GLYPH_CMAP_OFF+ 44#
S_squeaky_board        =nh.GLYPH_CMAP_OFF+ 45#
S_bear_trap            =nh.GLYPH_CMAP_OFF+ 46#
S_land_mine            =nh.GLYPH_CMAP_OFF+ 47#
S_rolling_boulder_trap =nh.GLYPH_CMAP_OFF+ 48#
S_sleeping_gas_trap    =nh.GLYPH_CMAP_OFF+ 49#
S_rust_trap            =nh.GLYPH_CMAP_OFF+ 50#
S_fire_trap            =nh.GLYPH_CMAP_OFF+ 51#
S_pit                  =nh.GLYPH_CMAP_OFF+ 52#
S_spiked_pit           =nh.GLYPH_CMAP_OFF+ 53#
S_hole                 =nh.GLYPH_CMAP_OFF+ 54#
S_trap_door            =nh.GLYPH_CMAP_OFF+ 55#
S_teleportation_trap   =nh.GLYPH_CMAP_OFF+ 56#
S_level_teleporter     =nh.GLYPH_CMAP_OFF+ 57#
S_magic_portal         =nh.GLYPH_CMAP_OFF+ 58#
S_web                  =nh.GLYPH_CMAP_OFF+ 59#
S_statue_trap          =nh.GLYPH_CMAP_OFF+ 60#
S_magic_trap           =nh.GLYPH_CMAP_OFF+ 61#
S_anti_magic_trap      =nh.GLYPH_CMAP_OFF+ 62#
S_polymorph_trap       =nh.GLYPH_CMAP_OFF+ 63#
S_vibrating_square     =nh.GLYPH_CMAP_OFF+ 64# /* for display rather than any trap effect */

#/* end traps, begin special effects */

S_vbeam       =nh.GLYPH_CMAP_OFF+ 65# /* The 4 zap beam symbols.  Do NOT separate. */
S_hbeam       =nh.GLYPH_CMAP_OFF+ 66# /* To change order or add, see function      */
S_lslant      =nh.GLYPH_CMAP_OFF+ 67# /* zapdir_to_glyph() in display.c.           */
S_rslant      =nh.GLYPH_CMAP_OFF+ 68#
S_digbeam     =nh.GLYPH_CMAP_OFF+ 69# /* dig beam symbol */
S_flashbeam   =nh.GLYPH_CMAP_OFF+ 70# /* camera flash symbol */
S_boomleft    =nh.GLYPH_CMAP_OFF+ 71# /* thrown boomerang, open left, e.g ')'    */
S_boomright   =nh.GLYPH_CMAP_OFF+ 72# /* thrown boomerang, open right, e.g. '('  */
S_ss1         =nh.GLYPH_CMAP_OFF+ 73# /* 4 magic shield ("resistance sparkle") glyphs */
S_ss2         =nh.GLYPH_CMAP_OFF+ 74#
S_ss3         =nh.GLYPH_CMAP_OFF+ 75#
S_ss4         =nh.GLYPH_CMAP_OFF+ 76#
S_poisoncloud =nh.GLYPH_CMAP_OFF+ 77
S_goodpos     =nh.GLYPH_CMAP_OFF+ 78# /* valid position for targeting via getpos() */

#/* The 8 swallow symbols.  Do NOT separate.  To change order or add, */
#/* see the function swallow_to_glyph() in display.c.                 */
S_sw_tl     =nh.GLYPH_CMAP_OFF+ 79# /* swallow top left [1]             */
S_sw_tc     =nh.GLYPH_CMAP_OFF+ 80# /* swallow top center [2]    Order: */
S_sw_tr     =nh.GLYPH_CMAP_OFF+ 81# /* swallow top right [3]            */
S_sw_ml     =nh.GLYPH_CMAP_OFF+ 82# /* swallow middle left [4]   1 2 3  */
S_sw_mr     =nh.GLYPH_CMAP_OFF+ 83# /* swallow middle right [6]  4 5 6  */
S_sw_bl     =nh.GLYPH_CMAP_OFF+ 84# /* swallow bottom left [7]   7 8 9  */
S_sw_bc     =nh.GLYPH_CMAP_OFF+ 85# /* swallow bottom center [8]        */
S_sw_br     =nh.GLYPH_CMAP_OFF+ 86# /* swallow bottom right [9]         */

S_explode1  =nh.GLYPH_CMAP_OFF+ 87# /* explosion top left               */
S_explode2  =nh.GLYPH_CMAP_OFF+ 88# /* explosion top center             */
S_explode3  =nh.GLYPH_CMAP_OFF+ 89# /* explosion top right        Ex.   */
S_explode4  =nh.GLYPH_CMAP_OFF+ 90# /* explosion middle left            */
S_explode5  =nh.GLYPH_CMAP_OFF+ 91# /* explosion middle center    /-\   */
S_explode6  =nh.GLYPH_CMAP_OFF+ 92# /* explosion middle right     |@|   */
S_explode7  =nh.GLYPH_CMAP_OFF+ 93# /* explosion bottom left      \-/   */
S_explode8  =nh.GLYPH_CMAP_OFF+ 94# /* explosion bottom center          */
S_explode9  =nh.GLYPH_CMAP_OFF+ 95# /* explosion bottom right           */

#/* end effects */

MAXPCHARS   = 96# /* maximum number of mapped characters */