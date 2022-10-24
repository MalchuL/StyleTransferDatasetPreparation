import numpy as np

from src.keypoint_alignment.converters.keypoints_converter import KeypointMapper


class FaceKeypoint28To4Mapper(KeypointMapper):
    def convert_points(self, points):
        # From stylegan mapping
        face_landmarks = points
        lm = np.array(face_landmarks)

        lm_eye_left = lm[11: 17]  # left-clockwise
        lm_eye_right = lm[17: 23]  # left-clockwise


        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        mouth_left   = lm[24]
        mouth_right  = lm[26]

        out_dict = {'eye_left': eye_left,
                    'eye_right': eye_right,
                    'mouth_left': mouth_left,
                    'mouth_right': mouth_right}
        return out_dict
