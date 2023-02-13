import numpy as np
import torch

import src.third_party.ESRGAN.RRDBNet_arch as arch


class ESRGANUpsampler:
    def __init__(self, path_to_ckpt, device):
        self.model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        self.model.load_state_dict(torch.load(path_to_ckpt, map_location=device), strict=True)
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device

    def get_upsample_size(self):
        return 4

    def __call__(self, img):
        return self.upsample(img)

    def upsample(self, img):
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(self.device)

        with torch.inference_mode():
            output = self.model(img_LR).data.squeeze(0).float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        return output
