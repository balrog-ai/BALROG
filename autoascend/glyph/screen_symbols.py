from . import screen_symbol_consts
from .screen_symbol_consts import *


def find(glyph):
    for k, v in vars(screen_symbol_const).items():
        if k.startswith('S_') and v == glyph:
            return f'SS.{k}'
