'''Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
'''

# Enable import from parent package
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import sdf_meshing
import configargparse
from os import listdir
from os.path import isfile, join

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=16384)
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--mode', type=str, default='mlp',
               help='Options are "mlp" or "nerf"')
p.add_argument('--resolution', type=int, default=1600)
p.add_argument('--filename', type=str, default='test')

p.add_argument('--checkpoints_path', default=None)
p.add_argument('--step', type=int, default=1,
               help='Options mean how many checkpoints will be skipped on each step while test_sdf')


opt = p.parse_args()

if opt.checkpoints_path and not opt.checkpoint_path:
    print("You must pass only one parameter --checkpoints_path or --checkpoint_path")

class SDFDecoder(torch.nn.Module):
    def __init__(self, checkpoint_path=opt.checkpoint_path):
        super().__init__()
        # Define the model.
        if opt.mode == 'mlp':
            self.model = modules.SingleBVPNet(type=opt.model_type, final_layer_factor=1, in_features=3)
        elif opt.mode == 'nerf':
            self.model = modules.SingleBVPNet(type='relu', mode='nerf', final_layer_factor=1, in_features=3)
        
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)['model_out']


if opt.checkpoints_path:
    mypath = opt.checkpoints_path
    files = sorted([f for f in listdir(mypath) if isfile(join(mypath, f)) and str(f).startswith('model_epoch')])
    step = opt.step
    for checkpoint_file in files[::step]:
        sdf_decoder = SDFDecoder(checkpoint_path=join(opt.checkpoints_path, checkpoint_file))
        
        root_path = os.path.join(opt.logging_root, opt.experiment_name, 'PLYs')
        utils.cond_mkdir(root_path)
        filename = str(checkpoint_file).split("model_")[1].split(".")[0] + "mesh"
        sdf_meshing.create_mesh(sdf_decoder, os.path.join(root_path, filename), N=opt.resolution)
        
         
    
else:
    sdf_decoder = SDFDecoder()
    root_path = os.path.join(opt.logging_root, opt.experiment_name, 'PLYs')
    utils.cond_mkdir(root_path)

    sdf_meshing.create_mesh(sdf_decoder, os.path.join(root_path, opt.filename), N=opt.resolution)
