import numpy as np

from ORION.config import Config


class ROIManager:
    def __init__(self, config: Config):
        self.config = config
        self.locked = False
        self.force_relock = True
        self.use_full_frame = False
        self.full_frame_cooldown = 0
        self.center_x = None
        self.center_y = None
        self.roi_w = int(config.ROI_MIN_SIZE_PX)
        self.roi_h = int(config.ROI_MIN_SIZE_PX)
        self.expand_votes = 0
        self.shrink_votes = 0
        self.miss_count = 0

    def reset(self) -> None:
        self.locked = False
        self.force_relock = True
        self.use_full_frame = False
        self.full_frame_cooldown = 0
        self.center_x = None
        self.center_y = None
        self.roi_w = int(self.config.ROI_MIN_SIZE_PX)
        self.roi_h = int(self.config.ROI_MIN_SIZE_PX)
        self.expand_votes = 0
        self.shrink_votes = 0
        self.miss_count = 0

    def _find_peak_maxpool(self, img: np.ndarray) -> tuple[float, float]:
        h, w = img.shape
        block = max(1, int(self.config.ROI_SEARCH_BLOCK))

        h_trim = (h // block) * block
        w_trim = (w // block) * block
        if h_trim == 0 or w_trim == 0:
            y, x = np.unravel_index(np.argmax(img), img.shape)
            return float(x), float(y)

        pooled = img[:h_trim, :w_trim].reshape(
            h_trim // block, block, w_trim // block, block
        ).max(axis=(1, 3))

        by, bx = np.unravel_index(np.argmax(pooled), pooled.shape)
        x0 = int(bx * block)
        y0 = int(by * block)

        sub = img[y0:min(y0 + block, h), x0:min(x0 + block, w)]
        sy, sx = np.unravel_index(np.argmax(sub), sub.shape)
        return float(x0 + sx), float(y0 + sy)

    def _clamp_bounds(
        self, cx: float, cy: float, size_w: int, size_h: int, shape: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        h, w = shape
        size_w = int(np.clip(size_w, self.config.ROI_MIN_SIZE_PX, min(self.config.ROI_MAX_SIZE_PX, w)))
        size_h = int(np.clip(size_h, self.config.ROI_MIN_SIZE_PX, min(self.config.ROI_MAX_SIZE_PX, h)))
        half_w = size_w // 2
        half_h = size_h // 2

        x0 = int(round(cx)) - half_w
        y0 = int(round(cy)) - half_h
        x0 = max(0, min(x0, w - size_w))
        y0 = max(0, min(y0, h - size_h))
        x1 = x0 + size_w
        y1 = y0 + size_h
        return x0, y0, x1, y1

    def get_crop(self, img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        h, w = img.shape
        if self.use_full_frame or self.full_frame_cooldown > 0:
            if self.full_frame_cooldown > 0:
                self.full_frame_cooldown -= 1
            return img, (0, 0)

        if self.force_relock or not self.locked or self.center_x is None or self.center_y is None:
            peak_x, peak_y = self._find_peak_maxpool(img)
            self.center_x = peak_x
            self.center_y = peak_y
            self.locked = True
            self.force_relock = False
            self.expand_votes = 0
            self.shrink_votes = 0

        x0, y0, x1, y1 = self._clamp_bounds(self.center_x, self.center_y, self.roi_w, self.roi_h, (h, w))
        return img[y0:y1, x0:x1], (x0, y0)

    def on_measurement(self, global_cx: float, global_cy: float, d4s_x_px: float, d4s_y_px: float) -> None:
        self.center_x = global_cx
        self.center_y = global_cy
        self.miss_count = 0

        span_x = 3.0 * float(d4s_x_px)
        span_y = 3.0 * float(d4s_y_px)
        if span_x <= 0 or span_y <= 0:
            return

        expand_limit_x = self.roi_w * float(self.config.ROI_EXPAND_THRESHOLD)
        expand_limit_y = self.roi_h * float(self.config.ROI_EXPAND_THRESHOLD)
        shrink_limit_x = self.roi_w * float(self.config.ROI_SHRINK_THRESHOLD)
        shrink_limit_y = self.roi_h * float(self.config.ROI_SHRINK_THRESHOLD)

        if span_x > expand_limit_x or span_y > expand_limit_y:
            self.expand_votes += 1
            self.shrink_votes = 0
        elif span_x < shrink_limit_x and span_y < shrink_limit_y:
            self.shrink_votes += 1
            self.expand_votes = 0
        else:
            self.expand_votes = 0
            self.shrink_votes = 0

        votes_needed = int(max(1, self.config.ROI_ADAPT_HYSTERESIS_FRAMES))
        target_w = int(np.clip(np.ceil(5.0 * d4s_x_px), self.config.ROI_MIN_SIZE_PX, self.config.ROI_MAX_SIZE_PX))
        target_h = int(np.clip(np.ceil(5.0 * d4s_y_px), self.config.ROI_MIN_SIZE_PX, self.config.ROI_MAX_SIZE_PX))

        if self.expand_votes >= votes_needed:
            self.roi_w = max(int(self.roi_w * 1.30), target_w)
            self.roi_h = max(int(self.roi_h * 1.30), target_h)
            self.expand_votes = 0
        elif self.shrink_votes >= votes_needed:
            self.roi_w = min(int(self.roi_w * 0.90), target_w)
            self.roi_h = min(int(self.roi_h * 0.90), target_h)
            self.shrink_votes = 0

        self.roi_w = int(np.clip(self.roi_w, self.config.ROI_MIN_SIZE_PX, self.config.ROI_MAX_SIZE_PX))
        self.roi_h = int(np.clip(self.roi_h, self.config.ROI_MIN_SIZE_PX, self.config.ROI_MAX_SIZE_PX))
        if self.use_full_frame and span_x < (self.roi_w * 0.2) and span_y < (self.roi_h * 0.2):
            self.use_full_frame = False
        if span_x > (self.roi_w * 0.55) or span_y > (self.roi_h * 0.55):
            # Enter full-frame temporarily when sizing confidence is low.
            self.full_frame_cooldown = max(self.full_frame_cooldown, 3)

    def on_border_signal(self) -> None:
        # Border energy indicates clipping; expand immediately and force a full-frame pass.
        prev_w, prev_h = self.roi_w, self.roi_h
        self.roi_w = int(np.clip(int(self.roi_w * 1.40), self.config.ROI_MIN_SIZE_PX, self.config.ROI_MAX_SIZE_PX))
        self.roi_h = int(np.clip(int(self.roi_h * 1.40), self.config.ROI_MIN_SIZE_PX, self.config.ROI_MAX_SIZE_PX))
        self.full_frame_cooldown = max(self.full_frame_cooldown, 3)
        if self.roi_w == prev_w and self.roi_h == prev_h:
            self.use_full_frame = True
        self.expand_votes = 0
        self.shrink_votes = 0

    def on_miss(self) -> None:
        self.miss_count += 1
        if self.miss_count >= 3:
            self.force_relock = True
            self.locked = False
            self.miss_count = 0

    def maybe_relock_if_near_edge(
        self, global_cx: float, global_cy: float, offset: tuple[int, int], crop_shape: tuple[int, int]
    ) -> None:
        x0, y0 = offset
        h, w = crop_shape
        margin_x = max(8.0, w * 0.1)
        margin_y = max(8.0, h * 0.1)

        lx = global_cx - x0
        ly = global_cy - y0
        if lx < margin_x or lx > (w - margin_x) or ly < margin_y or ly > (h - margin_y):
            self.force_relock = True
