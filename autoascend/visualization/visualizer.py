import multiprocessing
import queue
import time

import cv2
import nle.nethack as nh
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# avoid importing agent modules here, because it makes agent reloading less reliable
from .scopes import DrawTilesScope, DebugLogScope
from .utils import put_text, draw_frame, draw_grid, FONT_SIZE, VideoWriter

HISTORY_SIZE = 13
RENDERS_HISTORY_SIZE = 128


class Visualizer:

    def __init__(self, env, tileset_path='/tilesets/3.6.1tiles32.png', tile_size=32,
                 start_visualize=None, show=False, output_dir=None, frame_skipping=None, output_video_path=None):
        self.env = env
        self.tile_size = tile_size
        self._window_name = 'NetHackVis'
        self.show = show
        self.start_visualize = start_visualize
        self.output_dir = output_dir

        self.last_obs = None

        self.tileset = cv2.imread(tileset_path)[..., ::-1]
        if self.tileset is None:
            raise FileNotFoundError(f'Tileset {tileset_path} not found')
        if self.tileset.shape[0] % tile_size != 0 or self.tileset.shape[1] % tile_size != 0:
            raise ValueError("Tileset and tile_size doesn't match modulo")

        h = self.tileset.shape[0] // tile_size
        w = self.tileset.shape[1] // tile_size
        tiles = []
        for y in range(h):
            y *= tile_size
            for x in range(w):
                x *= tile_size
                tiles.append(self.tileset[y:y + tile_size, x:x + tile_size])
        self.tileset = np.array(tiles)

        # note that this file is a symlink (acutall file is in the docker container)
        from .glyph2tile import glyph2tile

        self.glyph2tile = np.array(glyph2tile)

        if self.show:
            print('Read tileset of size:', self.tileset.shape)

        self.action_history = list()

        self.message_history = list()
        self.popup_history = list()

        self.drawers = []
        self.log_messages = list()
        self.log_messages_history = list()

        self.frame_skipping = frame_skipping
        self.frame_counter = -1
        self._force_next_frame = False
        self._dynamic_frame_skipping_exp = lambda: min(0.95, 1 - 1 / (self.env.step_count + 1))
        self._dynamic_frame_skipping_render_time = 0
        self._dynamic_frame_skipping_agent_time = 1e-6
        self._dynamic_frame_skipping_threshold = 0.3  # for render_time / agent_time
        self._dynamic_frame_skipping_last_end_time = None
        self.total_time = 0

        self.renders_history = None
        if not self.show and output_video_path is None:
            assert output_dir is not None
            self.renders_history = queue.deque(maxlen=RENDERS_HISTORY_SIZE)
            self.output_dir = output_dir
            self.output_dir.mkdir(exist_ok=True, parents=True)

        self._start_display_thread()

        self.last_obs = None

        self.video_writer = None
        if output_video_path is not None:
            self.video_writer = VideoWriter(output_video_path, fps=10)

        self.tty_downscale = 1.0  # consider changing for better performance
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                                       int(26 * self.tty_downscale))

    def debug_tiles(self, *args, **kwargs):
        return DrawTilesScope(self, *args, **kwargs)

    def debug_log(self, txt, color):
        return DebugLogScope(self, txt, color)

    def step(self, obs, action):
        self.last_obs = obs
        self.action_history.append(action)
        self._update_log_message_history()
        self._update_message_history()
        self._update_popup_history()

        if self.video_writer is not None:
            frame = self._render()
            if frame is not None:
                self.video_writer.write(frame)

    def render(self):
        if self.video_writer is not None:
            return False

        self.frame_counter += 1
        render_start_time = None

        try:
            t = time.time()
            frame = self._render()
            if frame is None:
                return False
            render_start_time = t

            if self.show:
                self._display_queue.put(frame[..., ::-1].copy())

            if self.renders_history is not None:
                self.renders_history.append(frame)

        finally:
            self._update_dynamic_frame_skipping(render_start_time)

        return True

    def _render(self):
        if not self._force_next_frame and self.frame_skipping is not None:
            # static frame skipping
            if self.frame_counter % self.frame_skipping != 0:
                return None

        if self.frame_skipping is None:
            # dynamic frame skipping
            frame_skipping = self._dynamic_frame_skipping_render_time / self._dynamic_frame_skipping_agent_time / \
                             self._dynamic_frame_skipping_threshold
            if not self._force_next_frame and self.frame_counter <= frame_skipping:
                return None
            else:
                self.frame_counter = 0

        if self.last_obs is None:
            return None

        if self.start_visualize is not None:
            if self.env.step_count < self.start_visualize:
                return None

        if self._force_next_frame:
            self.frame_counter = 0
        self._force_next_frame = False

        glyphs = self.last_obs['glyphs']
        tiles_idx = self.glyph2tile[glyphs]
        tiles = self.tileset[tiles_idx.reshape(-1)]
        scene_vis = draw_grid(tiles, glyphs.shape[1])
        for drawer in self.drawers:
            scene_vis = drawer(scene_vis)
        draw_frame(scene_vis)
        topbar = self._draw_topbar(scene_vis.shape[1])
        bottombar = self._draw_bottombar(scene_vis.shape[1])

        rendered = np.concatenate([topbar, scene_vis, bottombar], axis=0)
        inventory = self._draw_inventory(rendered.shape[0])
        return np.concatenate([rendered, inventory], axis=1)

    def save_end_history(self):
        print('SAVING', self.output_dir)
        for i, render in enumerate(list(self.renders_history)):
            render = render[..., ::-1]
            out_path = self.output_dir / (str(i).rjust(5, '0') + '.jpg')
            cv2.imwrite(str(out_path), render)

    def force_next_frame(self):
        self._force_next_frame = True

    def stop_display_thread(self):
        if self.show:
            self._display_process.terminate()
            self._display_process.join()

    def _display_thread(self):
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

        last_size = (None, None)
        image = None
        while 1:
            is_new_image = False
            try:
                while 1:
                    try:
                        image = self._display_queue.get(timeout=0.03)
                        is_new_image = True
                    except queue.Empty:
                        break

                if image is None:
                    image = self._display_queue.get()
                    is_new_image = True

                width = cv2.getWindowImageRect(self._window_name)[2]
                height = cv2.getWindowImageRect(self._window_name)[3]
                ratio = min(width / image.shape[1], height / image.shape[0])
                width, height = round(image.shape[1] * ratio), round(image.shape[0] * ratio)

                if last_size != (width, height) or is_new_image:
                    last_size = (width, height)

                    resized_image = cv2.resize(image, (width, height), cv2.INTER_AREA)
                    cv2.imshow(self._window_name, resized_image)

                cv2.waitKey(1)
            except KeyboardInterrupt:
                pass
            except (ConnectionResetError, EOFError):
                return

        cv2.destroyWindow(self._window_name)

    def _start_display_thread(self):
        if self.show:
            self._display_queue = multiprocessing.Manager().Queue()
            self._display_process = multiprocessing.Process(target=self._display_thread, daemon=False)
            self._display_process.start()

    def _update_dynamic_frame_skipping(self, render_start_time):
        if self._dynamic_frame_skipping_last_end_time is not None:
            self.total_time += time.time() - self._dynamic_frame_skipping_last_end_time
            if render_start_time is not None:
                render_time = time.time() - render_start_time
            else:
                render_time = None
            agent_time = time.time() - self._dynamic_frame_skipping_last_end_time - \
                         (render_time if render_time is not None else 0)

            if render_start_time is not None:
                self._dynamic_frame_skipping_render_time = \
                    self._dynamic_frame_skipping_render_time * self._dynamic_frame_skipping_exp() + \
                    render_time * (1 - self._dynamic_frame_skipping_exp())
            self._dynamic_frame_skipping_agent_time = \
                self._dynamic_frame_skipping_agent_time * self._dynamic_frame_skipping_exp() + \
                agent_time * (1 - self._dynamic_frame_skipping_exp())

        self._dynamic_frame_skipping_last_end_time = time.time()

    def _draw_bottombar(self, width):
        height = FONT_SIZE * len(self.last_obs['tty_chars'])
        tty = self._draw_tty(self.last_obs, width - width // 2, height)
        stats = self._draw_stats(width // 2, height)
        return np.concatenate([tty, stats], axis=1)

    def _draw_stats(self, width, height):
        ret = np.zeros((height, width, 3), dtype=np.uint8)
        ch = self.env.agent.character
        if ch.role is None:
            return ret

        # game info
        i = 0
        txt = [f'Level num: {self.env.agent.current_level().level_number}',
               f'Dung num: {self.env.agent.current_level().dungeon_number}',
               f'Step: {self.env.step_count}',
               f'Turn: {self.env.agent._last_turn}',
               f'Score: {self.env.score}',
               ]
        put_text(ret, ' | '.join(txt), (0, i * FONT_SIZE), color=(255, 255, 255))
        i += 3

        # general character info
        txt = [
            {v: k for k, v in ch.name_to_role.items()}[ch.role],
            {v: k for k, v in ch.name_to_race.items()}[ch.race],
            {v: k for k, v in ch.name_to_alignment.items()}[ch.alignment],
            {v: k for k, v in ch.name_to_gender.items()}[ch.gender],
        ]
        put_text(ret, ' | '.join(txt), (0, i * FONT_SIZE))
        i += 1
        txt = [f'HP: {self.env.agent.blstats.hitpoints} / {self.env.agent.blstats.max_hitpoints}',
               f'LVL: {self.env.agent.blstats.experience_level}',
               f'ENERGY: {self.env.agent.blstats.energy} / {self.env.agent.blstats.max_energy}',
               ]
        hp_ratio = self.env.agent.blstats.hitpoints / self.env.agent.blstats.max_hitpoints
        hp_color = cv2.applyColorMap(np.array([[130 - int((1 - hp_ratio) * 110)]], dtype=np.uint8),
                                     cv2.COLORMAP_TURBO)[0, 0]
        put_text(ret, ' | '.join(txt), (0, i * FONT_SIZE), color=tuple(map(int, hp_color)))
        i += 2

        # proficiency info
        colors = {
            'Basic': (100, 100, 255),
            'Skilled': (100, 255, 100),
            'Expert': (100, 255, 255),
            'Master': (255, 255, 100),
            'Grand Master': (255, 100, 100),
        }
        for line in ch.get_skill_str_list():
            if 'Unskilled' not in line:
                put_text(ret, line, (0, i * FONT_SIZE), color=colors[line.split('-')[-1]])
                i += 1
        unskilled = []
        for line in ch.get_skill_str_list():
            if 'Unskilled' in line:
                unskilled.append(line.split('-')[0])
        put_text(ret, '|'.join(unskilled), (0, i * FONT_SIZE), color=(100, 100, 100))
        i += 2
        put_text(ret, 'Unarmed bonus: ' + str(ch.get_melee_bonus(None)), (0, i * FONT_SIZE))
        i += 2

        stats = list(self.env.agent.stats_logger.get_stats_dict().items())
        stats = [(k, v) for k, v in stats if v != 0]
        for j in range((len(stats) + 2) // 3):
            def format_value(v):
                if isinstance(v, float):
                    return f'{v:.2f}'
                return str(v)

            put_text(ret, ' | '.join(f'{k}={format_value(v)}' for k, v in stats[j * 3: (j + 1) * 3]),
                     (0, i * FONT_SIZE), color=(100, 100, 100))
            i += 1
        i += 1

        if hasattr(self.env.agent.character, 'known_spells'):
            put_text(ret, 'Known spells: ' + str(list(self.env.agent.character.known_spells)), (0, i * FONT_SIZE))
            i += 1

        monsters = [(dis, y, x, mon.mname) for dis, y, x, mon, _ in self.env.agent.get_visible_monsters()]
        put_text(ret, 'Monsters: ' + str(monsters), (0, i * FONT_SIZE))

        draw_frame(ret)
        return ret

    def _draw_topbar(self, width):
        actions_vis = self._draw_action_history(width // 25)
        messages_vis = self._draw_message_history(width // 4)
        popup_vis = self._draw_popup_history(width // 4)
        log_messages_vis = self._draw_debug_message_log(width - width // 25 - width // 4 - width // 4)
        ret = np.concatenate([actions_vis, messages_vis, popup_vis, log_messages_vis], axis=1)
        assert ret.shape[1] == width
        return ret

    def _draw_debug_message_log(self, width):
        vis = np.zeros((FONT_SIZE * HISTORY_SIZE, width, 3)).astype(np.uint8)
        for i in range(HISTORY_SIZE):
            if i >= len(self.log_messages_history):
                break
            txt = self.log_messages_history[-i - 1]
            if i == 0:
                put_text(vis, txt, (0, i * FONT_SIZE), color=(255, 255, 255))
            else:
                put_text(vis, txt, (0, i * FONT_SIZE), color=(120, 120, 120))
        draw_frame(vis)
        return vis

    def _update_log_message_history(self):
        txt = ''
        if self.env.agent is not None:
            txt = ' | '.join(self.log_messages)
        # if txt:
        self.log_messages_history.append(txt)

    def _draw_action_history(self, width):
        vis = np.zeros((FONT_SIZE * HISTORY_SIZE, width, 3)).astype(np.uint8)
        for i in range(HISTORY_SIZE):
            if i >= len(self.action_history):
                break
            txt = self.action_history[-i - 1]
            if i == 0:
                put_text(vis, txt, (0, i * FONT_SIZE), color=(255, 255, 255))
            else:
                put_text(vis, txt, (0, i * FONT_SIZE), color=(120, 120, 120))
        draw_frame(vis)
        return vis

    def _draw_message_history(self, width):
        messages_vis = np.zeros((FONT_SIZE * HISTORY_SIZE, width, 3)).astype(np.uint8)
        for i in range(HISTORY_SIZE):
            if i >= len(self.message_history):
                break
            txt = self.message_history[-i - 1]
            if i == 0:
                put_text(messages_vis, txt, (0, i * FONT_SIZE), color=(255, 255, 255))
            else:
                put_text(messages_vis, txt, (0, i * FONT_SIZE), color=(120, 120, 120))
        draw_frame(messages_vis)
        return messages_vis

    def _draw_popup_history(self, width):
        messages_vis = np.zeros((FONT_SIZE * HISTORY_SIZE, width, 3)).astype(np.uint8)
        for i in range(HISTORY_SIZE):
            if i >= len(self.popup_history):
                break
            txt = '|'.join(self.popup_history[-i - 1])
            if i == 0:
                put_text(messages_vis, txt, (0, i * FONT_SIZE), color=(255, 255, 255))
            else:
                put_text(messages_vis, txt, (0, i * FONT_SIZE), color=(120, 120, 120))
        draw_frame(messages_vis)
        return messages_vis

    def _update_message_history(self):
        txt = ''
        if self.env.agent is not None:
            txt = self.env.agent.message
        # if txt:
        self.message_history.append(txt)

    def _update_popup_history(self):
        txt = ''
        if self.env.agent is not None:
            txt = self.env.agent.popup
        # if txt:
        self.popup_history.append(txt)

    def _draw_tty(self, obs, width, height):
        vis = np.zeros((int(height * self.tty_downscale),
                        int(width * self.tty_downscale), 3)).astype(np.uint8)

        vis = Image.fromarray(vis)
        draw = ImageDraw.Draw(vis)

        for i, line in enumerate(obs['tty_chars']):
            txt = ''.join([chr(i) for i in line])
            draw.text((int(5 * self.tty_downscale), int((5 + i * 31) * self.tty_downscale)),
                      txt, (255, 255, 255), font=self.font)

        vis = np.array(vis.resize((width, height), Image.ANTIALIAS))
        draw_frame(vis)
        return vis

    def _draw_item(self, letter, item, width, height, indent=0):
        from ..item import Item

        bg_color = {
            nh.WAND_CLASS: np.array([0, 50, 50], dtype=np.uint8),
            nh.FOOD_CLASS: np.array([0, 50, 0], dtype=np.uint8),
            nh.ARMOR_CLASS: np.array([50, 50, 0], dtype=np.uint8),
            nh.RING_CLASS: np.array([50, 50, 0], dtype=np.uint8),
            nh.SCROLL_CLASS: np.array([30, 30, 30], dtype=np.uint8),
            nh.POTION_CLASS: np.array([0, 0, 50], dtype=np.uint8),
        }

        indent = int((width - 1) * (1 - 0.9 ** indent))

        vis = np.zeros((round(height * 0.9), width - indent, 3)).astype(np.uint8)
        if item.category in bg_color:
            vis += bg_color[item.category]
        if item.is_weapon():
            if item.is_thrown_projectile() or item.is_fired_projectile():
                vis += np.array([50, 0, 50], dtype=np.uint8)
            else:
                vis += np.array([50, 0, 0], dtype=np.uint8)
        if letter is not None:
            put_text(vis, str(letter), (0, 0))

        status_str, status_col = {
            Item.UNKNOWN: (' ', (255, 255, 255)),
            Item.CURSED: ('C', (255, 0, 0)),
            Item.UNCURSED: ('U', (0, 255, 255)),
            Item.BLESSED: ('B', (0, 255, 0)),
        }[item.status]
        put_text(vis, status_str, (FONT_SIZE, 0), color=status_col)

        if item.modifier is not None:
            put_text(vis, str(item.modifier), (FONT_SIZE * 2, 0))

        best_launcher, best_ammo = self.env.agent.inventory.get_best_ranged_set()
        best_melee = self.env.agent.inventory.get_best_melee_weapon()
        if item == best_launcher:
            put_text(vis, 'L', (FONT_SIZE * 3, 0), color=(255, 255, 255))
        if item == best_ammo:
            put_text(vis, 'A', (FONT_SIZE * 3, 0), color=(255, 255, 255))
        if item == best_melee:
            put_text(vis, 'M', (FONT_SIZE * 3, 0), color=(255, 255, 255))

        if item.is_weapon():
            put_text(vis, str(self.env.agent.character.get_melee_bonus(item)), (FONT_SIZE * 4, 0))

        put_text(vis, str(item), (FONT_SIZE * 8, round(FONT_SIZE * -0.1)), scale=FONT_SIZE / 40)
        # if len(item.objs) > 1:
        vis = np.concatenate([vis, np.zeros((vis.shape[0] // 2, vis.shape[1], 3), dtype=np.uint8)])
        put_text(vis, str(len(item.objs)) + ' | ' + ' | '.join((o.name for o in item.objs)),
                 (0, round(FONT_SIZE * 0.8)), scale=FONT_SIZE / 40)

        draw_frame(vis, color=(80, 80, 80), thickness=2)

        if item.equipped:
            cv2.rectangle(vis, (0, 0), (int(FONT_SIZE * 1.4), vis.shape[0] - 1), (0, 255, 255), 6)

        if indent != 0:
            vis = np.concatenate([np.zeros((vis.shape[0], width - vis.shape[1], 3), dtype=np.uint8), vis], 1)

        return vis

    def _draw_inventory(self, height):
        width = 800
        vis = np.zeros((height, width, 3), dtype=np.uint8)
        if self.env.agent:
            item_h = round(FONT_SIZE * 1.4)
            tiles = []
            for i, (letter, item) in enumerate(zip(self.env.agent.inventory.items.all_letters,
                                                   self.env.agent.inventory.items.all_items)):

                def rec_draw(item, letter, indent=0):
                    tiles.append(self._draw_item(letter, item, width, item_h, indent=indent))
                    if item.is_container():
                        for it in item.content:
                            rec_draw(it, None, indent + 1)

                rec_draw(item, letter, 0)
            if tiles:
                vis = np.concatenate(tiles, axis=0)
                if vis.shape[0] < height:
                    vis = np.concatenate([vis, np.zeros((height - vis.shape[0], width, 3), dtype=np.uint8)], axis=0)
                else:
                    vis = cv2.resize(vis, (width, height))
        draw_frame(vis)
        return vis
