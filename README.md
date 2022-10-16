This project allows to capture image style transfer dataset from any sources (Video, image folders)
It splits dataset into three images group: Faces, persons, lanscapes, (Animals and etc??) 

This repo inspired by https://github.com/thomd/stylegan2-toonification and [insert Random Styler paper]

PREREQUISITES
1. Step 0. Install MMCV using MIM.
   1. pip install torch==1.9 torchvision
   2. pip install -U openmim
   3. mim install mmcv-full
2. Install mmpose for face detection and alignment
   1. mim install mmpose
About configs and detectors:
   2. Approaches are classified into two kinds: the
two-step framework (top-down approach) and the part-based
framework (bottom-up approach). While the two-step framework first incorporates a person detector and then estimates
the pose within each box independently, detecting all body
parts in the image and associating parts belonging to distinct
persons is conducted in the part-based framework. You  can read about this in https://arxiv.org/pdf/2202.02656.pdf
   3. Tips: If you  want to find something in from valid_field, skip /smth because its not worked. Also all requests works in 'or' method.
   4. In search you will see `config id` use it in `mim download mmpose --config <config id>  --dest .`
   5. To find current ckpt I use `mim search mmpose  --dataset coco-wholebody --sort face_ap` and take first config
   6. Run `mim download mmpose --config topdown_heatmap_hrnet_w48_coco_wholebody_384x288_dark_plus  --dest ckpt/`
   7. If you want to know which method you use. Find model type 'TopDown' or 'BottomUp' if configs.py files which was downloaded
   8. Also we need a detection model which you can find by ` mim search mmdet  --dataset coco --config faster_rcnn_r50_fpn_1x_coco`. I think coco dataset is the most important part, because in MMDetDetectionStrategy hardcoded class for persons. Also you can find by dataset where 'person' in CLASSES putted first 
   9. This corresponds to mmpos demo files from `https://github.com/open-mmlab/mmpose/blob/master/demo/docs/2d_wholebody_pose_demo.md`
   10. download by `mim download mmdet --config faster_rcnn_r50_fpn_1x_coco --dest ckpt/`
   11. Finaly Run `python vis_demo_top_down.py ckpt/faster_rcnn_r50_fpn_1x_coco.py ckpt/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth ckpt/topdown_heatmap_hrnet_w48_coco_wholebody_384x288_dark_plus.py  ckpt/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth  --img-root demo --img arcane.jpg  --show`