'''
    Adapted from https://github.com/yabufarha/ms-tcn
'''

import torch
import os
import numpy as np
import torch.nn.functional as F
import random
from grid_sampler import GridSampler, TimeWarpLayer

class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate,features_dim):
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.features_dim = features_dim 
        self.sample_rate = sample_rate

        self.timewarp_layer = TimeWarpLayer()
        
    def get_num_examples(self):
        """Returns the total number of videos in the dataset."""
        return len(self.list_of_examples)

    def reset(self):
        self.index = 0
        self.my_shuffle()

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        self.gts = [self.gt_path + vid for vid in self.list_of_examples]
        self.features = [self.features_path + vid.split('.')[0] + '.npy' for vid in self.list_of_examples]
        self.my_shuffle()

    def my_shuffle(self):
        # shuffle list_of_examples, gts, features with the same order
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(self.list_of_examples)
        random.seed(randnum)
        random.shuffle(self.gts)
        random.seed(randnum)
        random.shuffle(self.features)


    def warp_video(self, batch_input_tensor, batch_target_tensor):
        '''
        :param batch_input_tensor: (bs, C_in, L_in)
        :param batch_target_tensor: (bs, L_in)
        :return: warped input and target
        '''
        bs, _, T = batch_input_tensor.shape
        grid_sampler = GridSampler(T)
        grid = grid_sampler.sample(bs)
        grid = torch.from_numpy(grid).float()

        warped_batch_input_tensor = self.timewarp_layer(batch_input_tensor, grid, mode='bilinear')
        batch_target_tensor = batch_target_tensor.unsqueeze(1).float()
        warped_batch_target_tensor = self.timewarp_layer(batch_target_tensor, grid, mode='nearest')  # no bilinear for label!
        warped_batch_target_tensor = warped_batch_target_tensor.squeeze(1).long()  # obtain the same shape

        return warped_batch_input_tensor, warped_batch_target_tensor

    def merge(self, bg, suffix):
        '''
        merge two batch generator. I.E
        BatchGenerator a;
        BatchGenerator b;
        a.merge(b, suffix='@1')
        :param bg:
        :param suffix: identify the video
        :return:
        '''

        self.list_of_examples += [vid + suffix for vid in bg.list_of_examples]
        self.gts += bg.gts
        self.features += bg.features

        print('Merge! Dataset length:{}'.format(len(self.list_of_examples)))


    def next_batch(self, batch_size, if_warp=False):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        lengths = []

        for vid in batch:
            features = np.load(os.path.join(self.features_path, vid + '.npy'))  # shape [T, C] or [C, T]
            
            # Ensure features shape is [C, T] for the model
            if features.shape[0] == self.features_dim:  # If already [C, T], do nothing
                pass
            elif features.shape[1] == self.features_dim:  # If [T, C], transpose to [C, T]
                features = features.T
            else:
                raise ValueError(f"Unexpected feature shape {features.shape} for video {vid}")
            
            with open(os.path.join(self.gt_path, vid + '.txt'), 'r') as f:
                content = f.read().splitlines()
            classes = np.array([self.actions_dict[c] for c in content])
            
            # Apply sample rate (downsample frames)
            features = features[:, ::self.sample_rate]  # downsample time dimension
            classes = classes[::self.sample_rate]

            batch_input.append(torch.tensor(features, dtype=torch.float))  # [C, T]
            batch_target.append(torch.tensor(classes, dtype=torch.long))   # [T]
            lengths.append(features.shape[1])  # time length after downsampling

        max_len = max(lengths)
        padded_inputs = []
        padded_targets = []
        masks = torch.zeros((len(batch), 1, max_len), dtype=torch.float)

        for i in range(len(batch_input)):
            feat = batch_input[i]  # [C, T]
            tgt = batch_target[i]  # [T]
            T = feat.shape[1]

            # Pad features to [C, max_len]
            padded_feat = F.pad(feat, (0, max_len - T))
            padded_inputs.append(padded_feat)

            # Pad targets to [max_len] with ignore_index = -100 for loss masking
            padded_tgt = F.pad(tgt, (0, max_len - T), value=-100)
            padded_targets.append(padded_tgt)

            # Update mask: 1 for valid frames, 0 for padded frames
            masks[i, 0, :T] = 1

        return (
            torch.stack(padded_inputs),     # [B, C, max_len]
            torch.stack(padded_targets),   # [B, max_len]
            masks,                         # [B, 1, max_len]
            batch
        )



if __name__ == '__main__':
    pass