import cv2
import numpy as np

FONT_SIZE = 32


def put_text(img, text, pos, scale=FONT_SIZE / 35, thickness=1, color=(255, 255, 0), console=False):
    # TODO: figure out how exactly opencv anchors the text
    pos = (pos[0] + FONT_SIZE // 2, pos[1] + FONT_SIZE // 2 + 8)

    if console:
        # TODO: implement equal characters size font
        # scale *= 2
        # font = cv2.FONT_HERSHEY_PLAIN
        font = cv2.FONT_HERSHEY_SIMPLEX
    else:
        font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.putText(img, text, pos, font,
                       scale, color, thickness, cv2.LINE_AA)


def draw_frame(img, color=(90, 90, 90), thickness=3):
    return cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, thickness)


def draw_grid(imgs, ncol):
    grid = imgs.reshape((-1, ncol, *imgs[0].shape))
    rows = []
    for row in grid:
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)
    return img


class VideoWriter:
    def __init__(self, path, fps, resolution=1080):
        self.path = path
        self.path.parent.mkdir(exist_ok=True, parents=True)
        self.out = None  # lazy init
        self.fps = fps
        self.resolution = (round(resolution * 16 / 9), resolution)

    def _make_writer(self, frame):
        h, w = frame.shape[:2]
        print(f'Initializing video writer with resolution {w}x{h}: {self.path}')
        return cv2.VideoWriter(str(self.path),
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               self.fps, (w, h))

    def write(self, frame):
        frame = cv2.resize(frame, self.resolution)
        frame = frame.astype(np.uint8)[..., ::-1]
        if self.out is None:
            self.out = self._make_writer(frame)
        self.out.write(frame)

    def close(self):
        self.out.release()