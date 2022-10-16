from src.human_detection.mmpose_detection.keypoints_utils.keypoints_set import KeypointsSet


def test_set():
    # frpm configs/_base_/datasets/coco_wholebody.py
    wholebody = {0: {'name': 'nose', 'id': 0, 'color': [51, 153, 255], 'type': 'upper', 'swap': ''},
            1: {'name': 'left_eye', 'id': 1, 'color': [51, 153, 255], 'type': 'upper', 'swap': 'right_eye'},
            2: {'name': 'right_eye', 'id': 2, 'color': [51, 153, 255], 'type': 'upper', 'swap': 'left_eye'},
            3: {'name': 'left_ear', 'id': 3, 'color': [51, 153, 255], 'type': 'upper', 'swap': 'right_ear'},
            4: {'name': 'right_ear', 'id': 4, 'color': [51, 153, 255], 'type': 'upper', 'swap': 'left_ear'},
            5: {'name': 'left_shoulder', 'id': 5, 'color': [0, 255, 0], 'type': 'upper',
                'swap': 'right_shoulder'},
            6: {'name': 'right_shoulder', 'id': 6, 'color': [255, 128, 0], 'type': 'upper',
                'swap': 'left_shoulder'},
            7: {'name': 'left_elbow', 'id': 7, 'color': [0, 255, 0], 'type': 'upper',
                'swap': 'right_elbow'},
            8: {'name': 'right_elbow', 'id': 8, 'color': [255, 128, 0], 'type': 'upper',
                'swap': 'left_elbow'},
            9: {'name': 'left_wrist', 'id': 9, 'color': [0, 255, 0], 'type': 'upper',
                'swap': 'right_wrist'},
            10: {'name': 'right_wrist', 'id': 10, 'color': [255, 128, 0], 'type': 'upper',
                 'swap': 'left_wrist'},
            11: {'name': 'left_hip', 'id': 11, 'color': [0, 255, 0], 'type': 'lower', 'swap': 'right_hip'},
            12: {'name': 'right_hip', 'id': 12, 'color': [255, 128, 0], 'type': 'lower',
                 'swap': 'left_hip'},
            13: {'name': 'left_knee', 'id': 13, 'color': [0, 255, 0], 'type': 'lower',
                 'swap': 'right_knee'},
            14: {'name': 'right_knee', 'id': 14, 'color': [255, 128, 0], 'type': 'lower',
                 'swap': 'left_knee'},
            15: {'name': 'left_ankle', 'id': 15, 'color': [0, 255, 0], 'type': 'lower',
                 'swap': 'right_ankle'},
            16: {'name': 'right_ankle', 'id': 16, 'color': [255, 128, 0], 'type': 'lower',
                 'swap': 'left_ankle'},
            17: {'name': 'left_big_toe', 'id': 17, 'color': [255, 128, 0], 'type': 'lower',
                 'swap': 'right_big_toe'},
            18: {'name': 'left_small_toe', 'id': 18, 'color': [255, 128, 0], 'type': 'lower',
                 'swap': 'right_small_toe'},
            19: {'name': 'left_heel', 'id': 19, 'color': [255, 128, 0], 'type': 'lower',
                 'swap': 'right_heel'},
            20: {'name': 'right_big_toe', 'id': 20, 'color': [255, 128, 0], 'type': 'lower',
                 'swap': 'left_big_toe'},
            21: {'name': 'right_small_toe', 'id': 21, 'color': [255, 128, 0], 'type': 'lower',
                 'swap': 'left_small_toe'},
            22: {'name': 'right_heel', 'id': 22, 'color': [255, 128, 0], 'type': 'lower',
                 'swap': 'left_heel'},
            23: {'name': 'face-0', 'id': 23, 'color': [255, 255, 255], 'type': '', 'swap': 'face-16'},
            24: {'name': 'face-1', 'id': 24, 'color': [255, 255, 255], 'type': '', 'swap': 'face-15'},
            25: {'name': 'face-2', 'id': 25, 'color': [255, 255, 255], 'type': '', 'swap': 'face-14'},
            26: {'name': 'face-3', 'id': 26, 'color': [255, 255, 255], 'type': '', 'swap': 'face-13'},
            27: {'name': 'face-4', 'id': 27, 'color': [255, 255, 255], 'type': '', 'swap': 'face-12'},
            28: {'name': 'face-5', 'id': 28, 'color': [255, 255, 255], 'type': '', 'swap': 'face-11'},
            29: {'name': 'face-6', 'id': 29, 'color': [255, 255, 255], 'type': '', 'swap': 'face-10'},
            30: {'name': 'face-7', 'id': 30, 'color': [255, 255, 255], 'type': '', 'swap': 'face-9'},
            31: {'name': 'face-8', 'id': 31, 'color': [255, 255, 255], 'type': '', 'swap': ''},
            32: {'name': 'face-9', 'id': 32, 'color': [255, 255, 255], 'type': '', 'swap': 'face-7'},
            33: {'name': 'face-10', 'id': 33, 'color': [255, 255, 255], 'type': '', 'swap': 'face-6'},
            34: {'name': 'face-11', 'id': 34, 'color': [255, 255, 255], 'type': '', 'swap': 'face-5'},
            35: {'name': 'face-12', 'id': 35, 'color': [255, 255, 255], 'type': '', 'swap': 'face-4'},
            36: {'name': 'face-13', 'id': 36, 'color': [255, 255, 255], 'type': '', 'swap': 'face-3'},
            37: {'name': 'face-14', 'id': 37, 'color': [255, 255, 255], 'type': '', 'swap': 'face-2'},
            38: {'name': 'face-15', 'id': 38, 'color': [255, 255, 255], 'type': '', 'swap': 'face-1'},
            39: {'name': 'face-16', 'id': 39, 'color': [255, 255, 255], 'type': '', 'swap': 'face-0'},
            40: {'name': 'face-17', 'id': 40, 'color': [255, 255, 255], 'type': '', 'swap': 'face-26'},
            41: {'name': 'face-18', 'id': 41, 'color': [255, 255, 255], 'type': '', 'swap': 'face-25'},
            42: {'name': 'face-19', 'id': 42, 'color': [255, 255, 255], 'type': '', 'swap': 'face-24'},
            43: {'name': 'face-20', 'id': 43, 'color': [255, 255, 255], 'type': '', 'swap': 'face-23'},
            44: {'name': 'face-21', 'id': 44, 'color': [255, 255, 255], 'type': '', 'swap': 'face-22'},
            45: {'name': 'face-22', 'id': 45, 'color': [255, 255, 255], 'type': '', 'swap': 'face-21'},
            46: {'name': 'face-23', 'id': 46, 'color': [255, 255, 255], 'type': '', 'swap': 'face-20'},
            47: {'name': 'face-24', 'id': 47, 'color': [255, 255, 255], 'type': '', 'swap': 'face-19'},
            48: {'name': 'face-25', 'id': 48, 'color': [255, 255, 255], 'type': '', 'swap': 'face-18'},
            49: {'name': 'face-26', 'id': 49, 'color': [255, 255, 255], 'type': '', 'swap': 'face-17'},
            50: {'name': 'face-27', 'id': 50, 'color': [255, 255, 255], 'type': '', 'swap': ''},
            51: {'name': 'face-28', 'id': 51, 'color': [255, 255, 255], 'type': '', 'swap': ''},
            52: {'name': 'face-29', 'id': 52, 'color': [255, 255, 255], 'type': '', 'swap': ''},
            53: {'name': 'face-30', 'id': 53, 'color': [255, 255, 255], 'type': '', 'swap': ''},
            54: {'name': 'face-31', 'id': 54, 'color': [255, 255, 255], 'type': '', 'swap': 'face-35'},
            55: {'name': 'face-32', 'id': 55, 'color': [255, 255, 255], 'type': '', 'swap': 'face-34'},
            56: {'name': 'face-33', 'id': 56, 'color': [255, 255, 255], 'type': '', 'swap': ''},
            57: {'name': 'face-34', 'id': 57, 'color': [255, 255, 255], 'type': '', 'swap': 'face-32'},
            58: {'name': 'face-35', 'id': 58, 'color': [255, 255, 255], 'type': '', 'swap': 'face-31'},
            59: {'name': 'face-36', 'id': 59, 'color': [255, 255, 255], 'type': '', 'swap': 'face-45'},
            60: {'name': 'face-37', 'id': 60, 'color': [255, 255, 255], 'type': '', 'swap': 'face-44'},
            61: {'name': 'face-38', 'id': 61, 'color': [255, 255, 255], 'type': '', 'swap': 'face-43'},
            62: {'name': 'face-39', 'id': 62, 'color': [255, 255, 255], 'type': '', 'swap': 'face-42'},
            63: {'name': 'face-40', 'id': 63, 'color': [255, 255, 255], 'type': '', 'swap': 'face-47'},
            64: {'name': 'face-41', 'id': 64, 'color': [255, 255, 255], 'type': '', 'swap': 'face-46'},
            65: {'name': 'face-42', 'id': 65, 'color': [255, 255, 255], 'type': '', 'swap': 'face-39'},
            66: {'name': 'face-43', 'id': 66, 'color': [255, 255, 255], 'type': '', 'swap': 'face-38'},
            67: {'name': 'face-44', 'id': 67, 'color': [255, 255, 255], 'type': '', 'swap': 'face-37'},
            68: {'name': 'face-45', 'id': 68, 'color': [255, 255, 255], 'type': '', 'swap': 'face-36'},
            69: {'name': 'face-46', 'id': 69, 'color': [255, 255, 255], 'type': '', 'swap': 'face-41'},
            70: {'name': 'face-47', 'id': 70, 'color': [255, 255, 255], 'type': '', 'swap': 'face-40'},
            71: {'name': 'face-48', 'id': 71, 'color': [255, 255, 255], 'type': '', 'swap': 'face-54'},
            72: {'name': 'face-49', 'id': 72, 'color': [255, 255, 255], 'type': '', 'swap': 'face-53'},
            73: {'name': 'face-50', 'id': 73, 'color': [255, 255, 255], 'type': '', 'swap': 'face-52'},
            74: {'name': 'face-51', 'id': 74, 'color': [255, 255, 255], 'type': '', 'swap': ''},
            75: {'name': 'face-52', 'id': 75, 'color': [255, 255, 255], 'type': '', 'swap': 'face-50'},
            76: {'name': 'face-53', 'id': 76, 'color': [255, 255, 255], 'type': '', 'swap': 'face-49'},
            77: {'name': 'face-54', 'id': 77, 'color': [255, 255, 255], 'type': '', 'swap': 'face-48'},
            78: {'name': 'face-55', 'id': 78, 'color': [255, 255, 255], 'type': '', 'swap': 'face-59'},
            79: {'name': 'face-56', 'id': 79, 'color': [255, 255, 255], 'type': '', 'swap': 'face-58'},
            80: {'name': 'face-57', 'id': 80, 'color': [255, 255, 255], 'type': '', 'swap': ''},
            81: {'name': 'face-58', 'id': 81, 'color': [255, 255, 255], 'type': '', 'swap': 'face-56'},
            82: {'name': 'face-59', 'id': 82, 'color': [255, 255, 255], 'type': '', 'swap': 'face-55'},
            83: {'name': 'face-60', 'id': 83, 'color': [255, 255, 255], 'type': '', 'swap': 'face-64'},
            84: {'name': 'face-61', 'id': 84, 'color': [255, 255, 255], 'type': '', 'swap': 'face-63'},
            85: {'name': 'face-62', 'id': 85, 'color': [255, 255, 255], 'type': '', 'swap': ''},
            86: {'name': 'face-63', 'id': 86, 'color': [255, 255, 255], 'type': '', 'swap': 'face-61'},
            87: {'name': 'face-64', 'id': 87, 'color': [255, 255, 255], 'type': '', 'swap': 'face-60'},
            88: {'name': 'face-65', 'id': 88, 'color': [255, 255, 255], 'type': '', 'swap': 'face-67'},
            89: {'name': 'face-66', 'id': 89, 'color': [255, 255, 255], 'type': '', 'swap': ''},
            90: {'name': 'face-67', 'id': 90, 'color': [255, 255, 255], 'type': '', 'swap': 'face-65'},
            91: {'name': 'left_hand_root', 'id': 91, 'color': [255, 255, 255], 'type': '',
                 'swap': 'right_hand_root'},
            92: {'name': 'left_thumb1', 'id': 92, 'color': [255, 128, 0], 'type': '',
                 'swap': 'right_thumb1'},
            93: {'name': 'left_thumb2', 'id': 93, 'color': [255, 128, 0], 'type': '',
                 'swap': 'right_thumb2'},
            94: {'name': 'left_thumb3', 'id': 94, 'color': [255, 128, 0], 'type': '',
                 'swap': 'right_thumb3'},
            95: {'name': 'left_thumb4', 'id': 95, 'color': [255, 128, 0], 'type': '',
                 'swap': 'right_thumb4'},
            96: {'name': 'left_forefinger1', 'id': 96, 'color': [255, 153, 255], 'type': '',
                 'swap': 'right_forefinger1'},
            97: {'name': 'left_forefinger2', 'id': 97, 'color': [255, 153, 255], 'type': '',
                 'swap': 'right_forefinger2'},
            98: {'name': 'left_forefinger3', 'id': 98, 'color': [255, 153, 255], 'type': '',
                 'swap': 'right_forefinger3'},
            99: {'name': 'left_forefinger4', 'id': 99, 'color': [255, 153, 255], 'type': '',
                 'swap': 'right_forefinger4'},
            100: {'name': 'left_middle_finger1', 'id': 100, 'color': [102, 178, 255], 'type': '',
                  'swap': 'right_middle_finger1'},
            101: {'name': 'left_middle_finger2', 'id': 101, 'color': [102, 178, 255], 'type': '',
                  'swap': 'right_middle_finger2'},
            102: {'name': 'left_middle_finger3', 'id': 102, 'color': [102, 178, 255], 'type': '',
                  'swap': 'right_middle_finger3'},
            103: {'name': 'left_middle_finger4', 'id': 103, 'color': [102, 178, 255], 'type': '',
                  'swap': 'right_middle_finger4'},
            104: {'name': 'left_ring_finger1', 'id': 104, 'color': [255, 51, 51], 'type': '',
                  'swap': 'right_ring_finger1'},
            105: {'name': 'left_ring_finger2', 'id': 105, 'color': [255, 51, 51], 'type': '',
                  'swap': 'right_ring_finger2'},
            106: {'name': 'left_ring_finger3', 'id': 106, 'color': [255, 51, 51], 'type': '',
                  'swap': 'right_ring_finger3'},
            107: {'name': 'left_ring_finger4', 'id': 107, 'color': [255, 51, 51], 'type': '',
                  'swap': 'right_ring_finger4'},
            108: {'name': 'left_pinky_finger1', 'id': 108, 'color': [0, 255, 0], 'type': '',
                  'swap': 'right_pinky_finger1'},
            109: {'name': 'left_pinky_finger2', 'id': 109, 'color': [0, 255, 0], 'type': '',
                  'swap': 'right_pinky_finger2'},
            110: {'name': 'left_pinky_finger3', 'id': 110, 'color': [0, 255, 0], 'type': '',
                  'swap': 'right_pinky_finger3'},
            111: {'name': 'left_pinky_finger4', 'id': 111, 'color': [0, 255, 0], 'type': '',
                  'swap': 'right_pinky_finger4'},
            112: {'name': 'right_hand_root', 'id': 112, 'color': [255, 255, 255], 'type': '',
                  'swap': 'left_hand_root'},
            113: {'name': 'right_thumb1', 'id': 113, 'color': [255, 128, 0], 'type': '',
                  'swap': 'left_thumb1'},
            114: {'name': 'right_thumb2', 'id': 114, 'color': [255, 128, 0], 'type': '',
                  'swap': 'left_thumb2'},
            115: {'name': 'right_thumb3', 'id': 115, 'color': [255, 128, 0], 'type': '',
                  'swap': 'left_thumb3'},
            116: {'name': 'right_thumb4', 'id': 116, 'color': [255, 128, 0], 'type': '',
                  'swap': 'left_thumb4'},
            117: {'name': 'right_forefinger1', 'id': 117, 'color': [255, 153, 255], 'type': '',
                  'swap': 'left_forefinger1'},
            118: {'name': 'right_forefinger2', 'id': 118, 'color': [255, 153, 255], 'type': '',
                  'swap': 'left_forefinger2'},
            119: {'name': 'right_forefinger3', 'id': 119, 'color': [255, 153, 255], 'type': '',
                  'swap': 'left_forefinger3'},
            120: {'name': 'right_forefinger4', 'id': 120, 'color': [255, 153, 255], 'type': '',
                  'swap': 'left_forefinger4'},
            121: {'name': 'right_middle_finger1', 'id': 121, 'color': [102, 178, 255], 'type': '',
                  'swap': 'left_middle_finger1'},
            122: {'name': 'right_middle_finger2', 'id': 122, 'color': [102, 178, 255], 'type': '',
                  'swap': 'left_middle_finger2'},
            123: {'name': 'right_middle_finger3', 'id': 123, 'color': [102, 178, 255], 'type': '',
                  'swap': 'left_middle_finger3'},
            124: {'name': 'right_middle_finger4', 'id': 124, 'color': [102, 178, 255], 'type': '',
                  'swap': 'left_middle_finger4'},
            125: {'name': 'right_ring_finger1', 'id': 125, 'color': [255, 51, 51], 'type': '',
                  'swap': 'left_ring_finger1'},
            126: {'name': 'right_ring_finger2', 'id': 126, 'color': [255, 51, 51], 'type': '',
                  'swap': 'left_ring_finger2'},
            127: {'name': 'right_ring_finger3', 'id': 127, 'color': [255, 51, 51], 'type': '',
                  'swap': 'left_ring_finger3'},
            128: {'name': 'right_ring_finger4', 'id': 128, 'color': [255, 51, 51], 'type': '',
                  'swap': 'left_ring_finger4'},
            129: {'name': 'right_pinky_finger1', 'id': 129, 'color': [0, 255, 0], 'type': '',
                  'swap': 'left_pinky_finger1'},
            130: {'name': 'right_pinky_finger2', 'id': 130, 'color': [0, 255, 0], 'type': '',
                  'swap': 'left_pinky_finger2'},
            131: {'name': 'right_pinky_finger3', 'id': 131, 'color': [0, 255, 0], 'type': '',
                  'swap': 'left_pinky_finger3'},
            132: {'name': 'right_pinky_finger4', 'id': 132, 'color': [0, 255, 0], 'type': '',
                  'swap': 'left_pinky_finger4'}}

    hand = {
        0:
        dict(name='thumb4', id=0, color=[255, 128, 0], type='', swap=''),
        1:
        dict(name='thumb3', id=1, color=[255, 128, 0], type='', swap=''),
        2:
        dict(name='thumb2', id=2, color=[255, 128, 0], type='', swap=''),
        3:
        dict(name='thumb1', id=3, color=[255, 128, 0], type='', swap=''),
        4:
        dict(
            name='forefinger4', id=4, color=[255, 153, 255], type='', swap=''),
        5:
        dict(
            name='forefinger3', id=5, color=[255, 153, 255], type='', swap=''),
        6:
        dict(
            name='forefinger2', id=6, color=[255, 153, 255], type='', swap=''),
        7:
        dict(
            name='forefinger1', id=7, color=[255, 153, 255], type='', swap=''),
        8:
        dict(
            name='middle_finger4',
            id=8,
            color=[102, 178, 255],
            type='',
            swap=''),
        9:
        dict(
            name='middle_finger3',
            id=9,
            color=[102, 178, 255],
            type='',
            swap=''),
        10:
        dict(
            name='middle_finger2',
            id=10,
            color=[102, 178, 255],
            type='',
            swap=''),
        11:
        dict(
            name='middle_finger1',
            id=11,
            color=[102, 178, 255],
            type='',
            swap=''),
        12:
        dict(
            name='ring_finger4', id=12, color=[255, 51, 51], type='', swap=''),
        13:
        dict(
            name='ring_finger3', id=13, color=[255, 51, 51], type='', swap=''),
        14:
        dict(
            name='ring_finger2', id=14, color=[255, 51, 51], type='', swap=''),
        15:
        dict(
            name='ring_finger1', id=15, color=[255, 51, 51], type='', swap=''),
        16:
        dict(name='pinky_finger4', id=16, color=[0, 255, 0], type='', swap=''),
        17:
        dict(name='pinky_finger3', id=17, color=[0, 255, 0], type='', swap=''),
        18:
        dict(name='pinky_finger2', id=18, color=[0, 255, 0], type='', swap=''),
        19:
        dict(name='pinky_finger1', id=19, color=[0, 255, 0], type='', swap=''),
        20:
        dict(name='wrist', id=20, color=[255, 255, 255], type='', swap='')
    }

    face = {
        0:
        dict(
            name='kpt-0', id=0, color=[255, 255, 255], type='', swap='kpt-16'),
        1:
        dict(
            name='kpt-1', id=1, color=[255, 255, 255], type='', swap='kpt-15'),
        2:
        dict(
            name='kpt-2', id=2, color=[255, 255, 255], type='', swap='kpt-14'),
        3:
        dict(
            name='kpt-3', id=3, color=[255, 255, 255], type='', swap='kpt-13'),
        4:
        dict(
            name='kpt-4', id=4, color=[255, 255, 255], type='', swap='kpt-12'),
        5:
        dict(
            name='kpt-5', id=5, color=[255, 255, 255], type='', swap='kpt-11'),
        6:
        dict(
            name='kpt-6', id=6, color=[255, 255, 255], type='', swap='kpt-10'),
        7:
        dict(name='kpt-7', id=7, color=[255, 255, 255], type='', swap='kpt-9'),
        8:
        dict(name='kpt-8', id=8, color=[255, 255, 255], type='', swap=''),
        9:
        dict(name='kpt-9', id=9, color=[255, 255, 255], type='', swap='kpt-7'),
        10:
        dict(
            name='kpt-10', id=10, color=[255, 255, 255], type='',
            swap='kpt-6'),
        11:
        dict(
            name='kpt-11', id=11, color=[255, 255, 255], type='',
            swap='kpt-5'),
        12:
        dict(
            name='kpt-12', id=12, color=[255, 255, 255], type='',
            swap='kpt-4'),
        13:
        dict(
            name='kpt-13', id=13, color=[255, 255, 255], type='',
            swap='kpt-3'),
        14:
        dict(
            name='kpt-14', id=14, color=[255, 255, 255], type='',
            swap='kpt-2'),
        15:
        dict(
            name='kpt-15', id=15, color=[255, 255, 255], type='',
            swap='kpt-1'),
        16:
        dict(
            name='kpt-16', id=16, color=[255, 255, 255], type='',
            swap='kpt-0'),
        17:
        dict(
            name='kpt-17',
            id=17,
            color=[255, 255, 255],
            type='',
            swap='kpt-26'),
        18:
        dict(
            name='kpt-18',
            id=18,
            color=[255, 255, 255],
            type='',
            swap='kpt-25'),
        19:
        dict(
            name='kpt-19',
            id=19,
            color=[255, 255, 255],
            type='',
            swap='kpt-24'),
        20:
        dict(
            name='kpt-20',
            id=20,
            color=[255, 255, 255],
            type='',
            swap='kpt-23'),
        21:
        dict(
            name='kpt-21',
            id=21,
            color=[255, 255, 255],
            type='',
            swap='kpt-22'),
        22:
        dict(
            name='kpt-22',
            id=22,
            color=[255, 255, 255],
            type='',
            swap='kpt-21'),
        23:
        dict(
            name='kpt-23',
            id=23,
            color=[255, 255, 255],
            type='',
            swap='kpt-20'),
        24:
        dict(
            name='kpt-24',
            id=24,
            color=[255, 255, 255],
            type='',
            swap='kpt-19'),
        25:
        dict(
            name='kpt-25',
            id=25,
            color=[255, 255, 255],
            type='',
            swap='kpt-18'),
        26:
        dict(
            name='kpt-26',
            id=26,
            color=[255, 255, 255],
            type='',
            swap='kpt-17'),
        27:
        dict(name='kpt-27', id=27, color=[255, 255, 255], type='', swap=''),
        28:
        dict(name='kpt-28', id=28, color=[255, 255, 255], type='', swap=''),
        29:
        dict(name='kpt-29', id=29, color=[255, 255, 255], type='', swap=''),
        30:
        dict(name='kpt-30', id=30, color=[255, 255, 255], type='', swap=''),
        31:
        dict(
            name='kpt-31',
            id=31,
            color=[255, 255, 255],
            type='',
            swap='kpt-35'),
        32:
        dict(
            name='kpt-32',
            id=32,
            color=[255, 255, 255],
            type='',
            swap='kpt-34'),
        33:
        dict(name='kpt-33', id=33, color=[255, 255, 255], type='', swap=''),
        34:
        dict(
            name='kpt-34',
            id=34,
            color=[255, 255, 255],
            type='',
            swap='kpt-32'),
        35:
        dict(
            name='kpt-35',
            id=35,
            color=[255, 255, 255],
            type='',
            swap='kpt-31'),
        36:
        dict(
            name='kpt-36',
            id=36,
            color=[255, 255, 255],
            type='',
            swap='kpt-45'),
        37:
        dict(
            name='kpt-37',
            id=37,
            color=[255, 255, 255],
            type='',
            swap='kpt-44'),
        38:
        dict(
            name='kpt-38',
            id=38,
            color=[255, 255, 255],
            type='',
            swap='kpt-43'),
        39:
        dict(
            name='kpt-39',
            id=39,
            color=[255, 255, 255],
            type='',
            swap='kpt-42'),
        40:
        dict(
            name='kpt-40',
            id=40,
            color=[255, 255, 255],
            type='',
            swap='kpt-47'),
        41:
        dict(
            name='kpt-41',
            id=41,
            color=[255, 255, 255],
            type='',
            swap='kpt-46'),
        42:
        dict(
            name='kpt-42',
            id=42,
            color=[255, 255, 255],
            type='',
            swap='kpt-39'),
        43:
        dict(
            name='kpt-43',
            id=43,
            color=[255, 255, 255],
            type='',
            swap='kpt-38'),
        44:
        dict(
            name='kpt-44',
            id=44,
            color=[255, 255, 255],
            type='',
            swap='kpt-37'),
        45:
        dict(
            name='kpt-45',
            id=45,
            color=[255, 255, 255],
            type='',
            swap='kpt-36'),
        46:
        dict(
            name='kpt-46',
            id=46,
            color=[255, 255, 255],
            type='',
            swap='kpt-41'),
        47:
        dict(
            name='kpt-47',
            id=47,
            color=[255, 255, 255],
            type='',
            swap='kpt-40'),
        48:
        dict(
            name='kpt-48',
            id=48,
            color=[255, 255, 255],
            type='',
            swap='kpt-54'),
        49:
        dict(
            name='kpt-49',
            id=49,
            color=[255, 255, 255],
            type='',
            swap='kpt-53'),
        50:
        dict(
            name='kpt-50',
            id=50,
            color=[255, 255, 255],
            type='',
            swap='kpt-52'),
        51:
        dict(name='kpt-51', id=51, color=[255, 255, 255], type='', swap=''),
        52:
        dict(
            name='kpt-52',
            id=52,
            color=[255, 255, 255],
            type='',
            swap='kpt-50'),
        53:
        dict(
            name='kpt-53',
            id=53,
            color=[255, 255, 255],
            type='',
            swap='kpt-49'),
        54:
        dict(
            name='kpt-54',
            id=54,
            color=[255, 255, 255],
            type='',
            swap='kpt-48'),
        55:
        dict(
            name='kpt-55',
            id=55,
            color=[255, 255, 255],
            type='',
            swap='kpt-59'),
        56:
        dict(
            name='kpt-56',
            id=56,
            color=[255, 255, 255],
            type='',
            swap='kpt-58'),
        57:
        dict(name='kpt-57', id=57, color=[255, 255, 255], type='', swap=''),
        58:
        dict(
            name='kpt-58',
            id=58,
            color=[255, 255, 255],
            type='',
            swap='kpt-56'),
        59:
        dict(
            name='kpt-59',
            id=59,
            color=[255, 255, 255],
            type='',
            swap='kpt-55'),
        60:
        dict(
            name='kpt-60',
            id=60,
            color=[255, 255, 255],
            type='',
            swap='kpt-64'),
        61:
        dict(
            name='kpt-61',
            id=61,
            color=[255, 255, 255],
            type='',
            swap='kpt-63'),
        62:
        dict(name='kpt-62', id=62, color=[255, 255, 255], type='', swap=''),
        63:
        dict(
            name='kpt-63',
            id=63,
            color=[255, 255, 255],
            type='',
            swap='kpt-61'),
        64:
        dict(
            name='kpt-64',
            id=64,
            color=[255, 255, 255],
            type='',
            swap='kpt-60'),
        65:
        dict(
            name='kpt-65',
            id=65,
            color=[255, 255, 255],
            type='',
            swap='kpt-67'),
        66:
        dict(name='kpt-66', id=66, color=[255, 255, 255], type='', swap=''),
        67:
        dict(
            name='kpt-67',
            id=67,
            color=[255, 255, 255],
            type='',
            swap='kpt-65'),
    }

    parser = KeypointsSet(wholebody)
    print(parser.group_ids)
    # parser = KeypointsSet(hand)
    # print(parser.group_ids)
    parser = KeypointsSet(face)
    print(parser.group_ids)