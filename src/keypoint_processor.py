import numpy as np
import config

class KeypointProcessor:
    """
    Handles keypoint validation, confidence filtering, body-relative normalization,
    imputation for missing joints, and temporal smoothing for both training and inference.
    """
    def __init__(self, conf_threshold=config.KEYPOINT_CONF_THRESHOLD, enable_smoothing=False, alpha=config.TEMPORAL_SMOOTHING_ALPHA):
        self.conf_threshold = conf_threshold
        self.enable_smoothing = enable_smoothing
        self.alpha = alpha
        self.prev_keypoints = None

    def process(self, raw_keypoints, is_live=False):
        """
        Accepts raw YOLO keypoints of shape (17, 2) or (17, 3).
        Returns normalized array of shape (17, 3) where columns are [norm_x, norm_y, confidence].
        """
        if raw_keypoints is None:
            return None

        kpts = np.array(raw_keypoints, dtype=np.float32)

        # Validate shape
        if kpts.ndim != 2 or kpts.shape[0] != 17:
            raise ValueError(f"Expected keypoints shape (17, 2) or (17, 3), got {kpts.shape}")

        # If shape is (17, 2), add confidence column of 1.0
        if kpts.shape[1] == 2:
            conf = np.ones((17, 1), dtype=np.float32)
            kpts = np.hstack([kpts, conf])
        elif kpts.shape[1] > 3:
            kpts = kpts[:, :3]

        xy = kpts[:, :2].copy()
        conf = kpts[:, 2:3].copy()

        # Low-confidence mask
        valid_mask = (conf >= self.conf_threshold).squeeze()

        # Handle missing keypoints: zero out low confidence xy coordinates
        xy[~valid_mask] = 0.0

        # Calculate body center reference point (mid-hip or mid-shoulder fallback)
        l_hip, r_hip = xy[11], xy[12]
        l_hip_conf, r_hip_conf = conf[11, 0], conf[12, 0]

        if l_hip_conf >= self.conf_threshold and r_hip_conf >= self.conf_threshold:
            center = (l_hip + r_hip) / 2.0
        elif l_hip_conf >= self.conf_threshold:
            center = l_hip
        elif r_hip_conf >= self.conf_threshold:
            center = r_hip
        else:
            # Fallback to mid-shoulder
            l_sh, r_sh = xy[5], xy[6]
            if conf[5, 0] >= self.conf_threshold or conf[6, 0] >= self.conf_threshold:
                center = (l_sh + r_sh) / 2.0
            else:
                # Absolute centroid of valid points
                valid_xy = xy[valid_mask]
                center = valid_xy.mean(axis=0) if len(valid_xy) > 0 else np.array([0.0, 0.0], dtype=np.float32)

        # Body-relative translation
        translated_xy = xy - center
        translated_xy[~valid_mask] = 0.0

        # Body scale normalization (torso length or bounding box size)
        l_sh, r_sh = xy[5], xy[6]
        mid_sh = (l_sh + r_sh) / 2.0
        mid_hip = (l_hip + r_hip) / 2.0
        torso_len = np.linalg.norm(mid_sh - mid_hip)

        if torso_len < 10.0 or not (valid_mask[5] or valid_mask[6]):
            # Fallback to bounding box diagonal of valid keypoints
            valid_pts = xy[valid_mask]
            if len(valid_pts) >= 2:
                bbox_min = valid_pts.min(axis=0)
                bbox_max = valid_pts.max(axis=0)
                scale = np.linalg.norm(bbox_max - bbox_min)
            else:
                scale = 1.0
        else:
            scale = torso_len

        if scale < 1e-5:
            scale = 1.0

        normalized_xy = translated_xy / scale

        # Optional temporal smoothing (EMA) for live stream
        if self.enable_smoothing and is_live:
            if self.prev_keypoints is not None:
                normalized_xy = self.alpha * normalized_xy + (1 - self.alpha) * self.prev_keypoints
            self.prev_keypoints = normalized_xy.copy()

        # Combine normalized xy and confidence
        result = np.hstack([normalized_xy, conf])
        return result

    def get_conv1d_input(self, normalized_kpts):
        """
        Converts (17, 3) normalized keypoints into shape (3, 17) for PyTorch Conv1D input.
        """
        if normalized_kpts is None:
            return None
        # Transpose (17, 3) -> (3, 17)
        return normalized_kpts.T.astype(np.float32)

    def reset_temporal_state(self):
        self.prev_keypoints = None
