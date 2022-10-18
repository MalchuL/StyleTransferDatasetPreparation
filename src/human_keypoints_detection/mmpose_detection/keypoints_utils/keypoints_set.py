from src.human_keypoints_detection.mmpose_detection.keypoints_utils.keypoints_groups import KPS_GROUPS


class KeypointsSet:

    def __init__(self, keypoints_info):
        """

        :param keypoints_info: dict from pose_model.cfg.dataset_info.keypoint_info
        """
        self.id2name, self.name2id = self.parse_keypoints_names(keypoints_info)
        self.kps_groups = KPS_GROUPS

        self.group_ids = self.channel_group_to_id(self.kps_groups, self.name2id)

    def channel_group_to_id(self, kps_groups, name2id):
        group_ids = {}
        for kps_group, names in kps_groups.items():
            group_ids[kps_group] = [name2id[alias] for aliases in names for alias in aliases if alias in name2id]
        mapped_count = 0
        for group, ids in group_ids.items():
            mapped_count += len(ids)
        assert mapped_count == len(name2id), f'no mapped for {name2id}'
        return group_ids

    def parse_keypoints_names(self, keypoints_info):
        """
        Maps dataset info about channel and names into ids
        :param keypoints_info: dict from pose_model.cfg.dataset_info.keypoint_info
        :return:
        """
        keys = set([x['name'] for x in keypoints_info.values()])
        assert len(keys) == len(keypoints_info), 'Exists some keys with similar names'
        num_channels = len(keys)
        id2name = [None] *  num_channels
        for id, kp_info in keypoints_info.items():
            assert id == kp_info['id']
            id2name[id] = kp_info['name']
        assert all([x is not None for x in id2name])
        name2id = {kp_info['name']: id for id, kp_info in keypoints_info.items()}
        return id2name, name2id

