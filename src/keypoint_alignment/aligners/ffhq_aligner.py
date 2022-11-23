import PIL.Image
import numpy as np
import scipy.ndimage


class FFHQAligner:
    def __init__(self, output_size=1024, transform_size=4096, enable_padding=True, x_scale=1, y_scale=1, em_scale=0.1, blur_padding=False):

        self.blur_padding = blur_padding
        self.output_size = output_size
        self.transform_size = transform_size
        self.enable_padding = enable_padding
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.em_scale = em_scale

    def align(self, img, keypoitns):
        eye_left = keypoitns['eye_left']
        eye_right = keypoitns['eye_right']
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = keypoitns['mouth_left']
        mouth_right = keypoitns['mouth_right']
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        x *= self.x_scale
        y = np.flipud(x) * [-self.y_scale, self.y_scale]
        c = eye_avg + eye_to_mouth * self.em_scale
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Visualization
        # img = np.uint8(img)
        # for point, color in zip([eye_left, eye_right, mouth_left, mouth_right],
        #                         [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]):
        #     img = cv2.circle(img, tuple(point.astype(np.int)), 5, color, lineType=cv2.LINE_4)
        # img = PIL.Image.fromarray(img, 'RGB')
        img = PIL.Image.fromarray(img, 'RGB')
        # Shrink.
        shrink = int(np.floor(qsize / self.output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if self.enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            if self.blur_padding:
                h, w, _ = img.shape
                y, x, _ = np.ogrid[:h, :w, :1]
                mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                                  1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
                blur = qsize * 0.02
                img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
                img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = np.uint8(np.clip(np.rint(img), 0, 255))

            img = PIL.Image.fromarray(img, 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((self.transform_size, self.transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                            PIL.Image.BILINEAR)
        if self.output_size < self.transform_size:
            img = img.resize((self.output_size, self.output_size), PIL.Image.ANTIALIAS)

        return np.uint8(img)
