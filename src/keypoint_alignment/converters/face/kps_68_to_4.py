import numpy as np

from src.keypoint_alignment.converters.keypoints_converter import KeypointMapper


class FaceKeypoint68To4Mapper(KeypointMapper):
    def convert_points(self, points):
        # From stylegan mapping
        face_landmarks = points
        lm = np.array(face_landmarks)
        lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise

        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]

        out_dict = {'eye_left': eye_left,
                    'eye_right': eye_right,
                    'mouth_left': mouth_left,
                    'mouth_right': mouth_right}
        return out_dict
