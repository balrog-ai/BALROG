import cv2
import numpy as np

from .utils import VideoWriter


class DrawTilesScope():

    def __init__(self, visualizer, tiles, color, is_path=False, is_heatmap=False, mode='fill'):
        from ..glyph import C  # imported here to allow agent reloading
        self.visualizer = visualizer
        self.is_heatmap = is_heatmap
        self.color = color
        self.mode = mode
        if self.is_heatmap:
            assert not is_path
            assert self.mode == 'fill'
            assert isinstance(self.color, str)
            self.tiles = tiles
        else:
            if isinstance(tiles, np.ndarray) and tiles.shape == (C.SIZE_Y, C.SIZE_X):
                self.tiles = list(zip(*tiles.nonzero()))
            else:
                self.tiles = tiles
            self.is_path = is_path

    def draw_fun(self, rendered):
        if self.is_heatmap:
            if self.is_heatmap:
                grayscale = np.zeros(rendered.shape, dtype=float)
                mask = np.ones_like(grayscale).astype(bool)
                for y in range(self.tiles.shape[0]):
                    for x in range(self.tiles.shape[1]):
                        y1 = y * self.visualizer.tile_size
                        x1 = x * self.visualizer.tile_size
                        slic = (slice(y1, y1 + self.visualizer.tile_size),
                                slice(x1, x1 + self.visualizer.tile_size))
                        if np.isnan(self.tiles[y, x]):
                            mask[slic] = False
                        else:
                            grayscale[slic] = self.tiles[y, x]
                grayscale[mask] -= np.min(grayscale[mask])
                grayscale[mask] /= np.max(grayscale[mask])
                grayscale = (grayscale * 255).astype(np.uint8)
                grayscale = cv2.blur(grayscale, (15, 15))
                # https://docs.opencv.org/4.5.2/d3/d50/group__imgproc__colormap.html
                heatmap = cv2.applyColorMap(grayscale, cv2.__dict__[f'COLORMAP_{self.color.upper()}'])[..., ::-1]
                return (rendered * 0.5 + heatmap * 0.5).astype(np.uint8) * mask + (rendered * ~mask)

        color = self.color
        alpha = 1
        if len(color) == 4:
            alpha = color[-1] / 255
            color = color[:-1]

        if alpha != 1:
            orig_rendered, rendered = rendered, np.zeros_like(rendered)

        if self.is_path:
            for p1, p2 in zip(self.tiles, self.tiles[1:]):
                p1 = [round((i + 0.5) * self.visualizer.tile_size) for i in p1][::-1]
                p2 = [round((i + 0.5) * self.visualizer.tile_size) for i in p2][::-1]
                cv2.line(rendered, p1, p2, color, 2)
        else:
            for p in self.tiles:
                p1 = [round(i * self.visualizer.tile_size) for i in p][::-1]
                p2 = [round((i + 1) * self.visualizer.tile_size) for i in p][::-1]
                if self.mode == 'fill':
                    cv2.rectangle(rendered, p1, p2, color, -1)
                if self.mode == 'frame':
                    cv2.rectangle(rendered, p1, p2, color, 3)

        if alpha != 1:
            rendered = np.clip(orig_rendered.astype(np.int16) + (rendered * alpha).astype(np.int16), 0, 255).astype(
                np.uint8)

        return rendered.copy()

    def __enter__(self):
        self.fun_instance = lambda x: self.draw_fun(x)
        self.visualizer.drawers.append(self.fun_instance)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.visualizer.drawers.remove(self.fun_instance)


class DebugLogScope():

    def __init__(self, visualizer, txt, color):
        self.visualizer = visualizer
        self.txt = txt
        self.color = color

    def __enter__(self):
        self.visualizer.log_messages.append(self.txt)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.visualizer.log_messages.remove(self.txt)
