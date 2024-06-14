from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from learning.amp_network_builder import AMPBuilder
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import copy
from phc.learning.unrealego import network
from easydict import EasyDict as edict
import timm
DISC_LOGIT_INIT_SCALE = 1.0


def remove_bn_from_resnet(model):
    # Collect names of batch norm layers
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append(name)

    # Replace batch norm layers with nn.Identity
    for name in bn_layers:
        parent_module = model
        components = name.split('.')

        # Navigate to the parent module of the batch norm layer
        for component in components[:-1]:
            parent_module = getattr(parent_module, component)

        # Replace the batch norm layer with nn.Identity
        setattr(parent_module, components[-1], nn.Identity())

    return model

def replace_bn_with_gn(model, num_groups=32):
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # Define the new GroupNorm layer
            new_layer = nn.GroupNorm(num_groups=num_groups, num_channels=child.num_features, eps=child.eps, affine=True)
            setattr(model, name, new_layer)
        else:
            # Recursively apply to children
            replace_bn_with_gn(child, num_groups=num_groups)
    return model

class AMPResNetBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPResNetBuilder.Network(self.params, **kwargs)
        return net

    class Network(AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            self.self_obs_size = kwargs['self_obs_size']
            self.task_obs_size = kwargs['task_obs_size']
            self.task_obs_size_detail = kwargs['task_obs_size_detail']
            self.img_size = self.task_obs_size_detail['img_size']
            self.target_size = self.task_obs_size_detail['target_size']
            self.use_unrealego = self.task_obs_size_detail.get("use_unrealego", False)
            self.use_convnext = self.task_obs_size_detail.get("use_convnext", False)
            self.use_resnet_no_bn = self.task_obs_size_detail.get("use_resnet_no_bn", False)
            self.use_resnet_gn = self.task_obs_size_detail.get("use_resnet_gn", False)
            self.use_siamese = self.task_obs_size_detail.get("use_siamese", False)
            self.use_visibility_branch = self.task_obs_size_detail.get("use_visibility_branch", False)
            self.num_joints = self.task_obs_size_detail.get("num_joints", 23)
            self.pretrain = self.task_obs_size_detail.get("pretrain", False)
            
            self.img_latent_dim = self.task_obs_size_detail.get("img_latent_dim", 512)
            self.img_size_flat = np.prod(self.img_size) 
            
            if len(self.img_size) > 1: # full images
                kwargs['input_shape'] = (kwargs['self_obs_size'] + self.target_size + self.img_latent_dim,)  # Task embedding size + self_obs
            else:
                kwargs['input_shape'] = (kwargs['self_obs_size'] + self.target_size + self.img_size_flat,)  # Task embedding size + self_obs
            
            super().__init__(params, **kwargs)
            self.running_mean = kwargs['mean_std'].running_mean
            self.running_var = kwargs['mean_std'].running_var
            
            ### build resnet
            if len(self.img_size) > 1: # full images
                if self.use_unrealego:
                    opt = edict()
                    opt.init_ImageNet = True
                    opt.num_heatmap = self.num_joints
                    opt.ae_hidden_size = 20
                    self.net_heatmap = network.HeatMap_UnrealEgo_Shared(opt=opt, model_name='resnet18')
                    
                    self.after_heatmap = nn.Sequential(
                        torch.nn.Conv2d(1024, 4, 1),
                        torch.nn.SiLU(), 
                    )
                    self.feat_mlp = torch.nn.Linear(32 * 40, 512)
                elif self.use_convnext:
                    convnext = timm.create_model("convnextv2_atto.fcmae_ft_in1k", pretrained=True)
                    convnext.stem[0] = torch.nn.Conv2d(self.img_size[0], 40, kernel_size=(4, 4), stride=(4, 4))
                    convnext.head = nn.Sequential(
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Identity()
                    )
                    self.resnet_embedding=convnext
                    
                else:
                    resnet = resnet18(pretrained=True)
                    if self.use_siamese:
                        resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                    else:
                        resnet.conv1 = torch.nn.Conv2d(self.img_size[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                    if self.use_resnet_no_bn:
                        resnet = remove_bn_from_resnet(resnet)
                        
                    if self.use_resnet_gn:
                        resnet = replace_bn_with_gn(resnet)
                        
                    modules=list(resnet.children())[:-1]
                    self.resnet_embedding=nn.Sequential(*modules)
                    
            if self.use_visibility_branch and self.use_siamese:
                vis_mlp_units = [1024, 512, 256]
                mlp_args = {'input_size': self.img_latent_dim//self.img_size[0], 'units': vis_mlp_units, 'activation': self.activation, 'dense_func': torch.nn.Linear}
                self.vis_mlp = self._build_mlp(**mlp_args)
                self.vis_mlp.append(nn.Linear(vis_mlp_units[-1], self.num_joints))
                
            # self.resnet_embedding_actor=nn.Sequential(*copy.deepcopy(modules))

        def forward(self, obs_dict):
            states = obs_dict.get('rnn_states', None)

            actor_outputs = self.eval_actor(obs_dict)
            value_outputs = self.eval_critic(obs_dict)

            if self.has_rnn:
                mu, sigma, a_states = actor_outputs
                value, c_states = value_outputs
                states = a_states + c_states
                output = mu, sigma, value, states
            else:
                output = actor_outputs + (value_outputs, states)

            return output
            
        def eval_critic(self, obs_dict):
            obs = obs_dict['obs']
            seq_length = obs_dict.get('seq_length', 1)
            states = obs_dict.get('rnn_states', None)
            B = obs.shape[0]
            c_out = self.critic_cnn(obs)  # This is empty
            c_out = c_out.contiguous().view(B, -1)
            if self.pretrain:
                other_obs = c_out
            else:
                other_obs = c_out[:, :-self.img_size_flat]
                img_obs = c_out[:, -self.img_size_flat:]
                
            if len(self.img_size) > 1: # full images
                if self.use_unrealego:
                    
                    heat_out, feat_out = self.net_heatmap.forward_feat_full(img_obs.view(B, *self.img_size))
                    feat_out = self.after_heatmap(feat_out)
                    img_out = self.feat_mlp(feat_out.view(feat_out.shape[0] * feat_out.shape[1], -1)).view(B, -1)
                else:
                    if self.pretrain:
                        img_out = torch.zeros((B, self.img_latent_dim)).to(c_out)
                    else:
                        if self.use_siamese:
                            images = img_obs.view(B, *self.img_size)
                            img_out = torch.cat([self.resnet_embedding(images[:, (idx):(idx+1)]).squeeze(-1).squeeze(-1) for idx in range(self.img_size[0])], dim = -1)
                        else:
                            img_out = self.resnet_embedding(img_obs.view(B, *self.img_size)).view(B, -1)
                    
            else:
                img_out = img_obs
            
            
            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    c_out_in = c_out
                    c_input = torch.cat([other_obs, img_out], dim=-1)
                    c_out = self.critic_mlp(c_input)

                    if self.rnn_concat_input:
                        c_out = torch.cat([c_out, c_out_in], dim=1)

                batch_size = c_out.size()[0]
                num_seqs = batch_size // seq_length
                c_out = c_out.reshape(num_seqs, seq_length, -1)

                if self.rnn_name == 'sru':
                    c_out = c_out.transpose(0, 1)
                ################# New RNN
                if len(states) == 2:
                    c_states = states[1].reshape(num_seqs, seq_length, -1)
                else:
                    c_states = states[2:].reshape(num_seqs, seq_length, -1)
                c_out, c_states = self.c_rnn(c_out, c_states[:, 0:1].transpose(0, 1).contiguous()) # ZL: only pass the first state, others are ignored. ???            
                
                ################# Old RNN
                # if len(states) == 2:	
                #     c_states = states[1]	
                # else:	
                #     c_states = states[2:]	
                # c_out, c_states = self.c_rnn(c_out, c_states)
                
                
                if self.rnn_name == 'sru':
                    c_out = c_out.transpose(0, 1)
                else:
                    if self.rnn_ln:
                        c_out = self.c_layer_norm(c_out)
                c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)

                if type(c_states) is not tuple:
                    c_states = (c_states,)

                if self.is_rnn_before_mlp:
                    c_out = self.critic_mlp(c_out)
                value = self.value_act(self.value(c_out))
                return value, c_states

            else:
                c_input = torch.cat([other_obs, img_out], dim=-1)

                c_out = self.critic_mlp(c_input)
                value = self.value_act(self.value(c_out))
                return value
            
        def compute_heatmap(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)
            B = obs.shape[0]
            a_out = self.actor_cnn(obs)  # This is empty
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            other_obs = obs[:, :-self.img_size_flat]
            img_obs = obs[:, -self.img_size_flat:].contiguous()
            
            if len(self.img_size) > 1: # full images
                if self.use_unrealego:
                    heat_out, feat_out = self.net_heatmap.forward_feat_full(img_obs.view(B, *self.img_size))
                    return heat_out
                else:
                    raise NotImplementedError
        
        def eval_visibility(self, img_out):
            return self.vis_mlp(img_out)

        def eval_actor(self, obs_dict, return_extra = False):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)
            B = obs.shape[0]
            a_out = self.actor_cnn(obs)  # This is empty
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            
            
            if self.pretrain:
                other_obs = a_out
            else:
                other_obs = a_out[:, :-self.img_size_flat]
                img_obs = a_out[:, -self.img_size_flat:]
                
            extra_dict = {}
            
            if len(self.img_size) > 1: # full images
                if self.use_unrealego:
                    heat_out, feat_out = self.net_heatmap.forward_feat_full(img_obs.view(B, *self.img_size))
                    feat_out = self.after_heatmap(feat_out)
                    
                    img_out = self.feat_mlp(feat_out.view(feat_out.shape[0] * feat_out.shape[1], -1)).view(B, -1)
                    # torch.cat([self.feat_mlp(feat_out[:, 0:1].view(B, -1)), self.feat_mlp(feat_out[:, 1:2].view(B, -1)), self.feat_mlp(feat_out[:, 2:3].view(B, -1)), self.feat_mlp(feat_out[:, 3:4].view(B, -1))], dim = -1)
                else:
                    if self.pretrain:
                        img_out = torch.zeros((B, self.img_latent_dim)).to(a_out)
                    else:
                        if self.use_siamese:
                            images = img_obs.view(B, *self.img_size)
                            img_out = torch.cat([self.resnet_embedding(images[:, (idx):(idx+1)]).squeeze(-1).squeeze(-1) for idx in range(self.img_size[0])], dim = -1)
                        else:
                            img_out = self.resnet_embedding(img_obs.view(B, *self.img_size)).view(B, -1)
                    
            else: # pretrained images. 
                img_out = img_obs
            
            # self.resnet_embedding(img_obs.view(B, *self.img_size))[0:1] - self.resnet_embedding(img_obs.view(B, *self.img_size)[0:1])
            
            
            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    a_out_in = a_out
                    actor_input = torch.cat([other_obs, img_out], dim=-1)
                    a_out = self.actor_mlp(actor_input)
                    
                    if self.rnn_concat_input:
                        a_out = torch.cat([a_out, a_out_in], dim=1)
                
                batch_size = a_out.size()[0]
                num_seqs = batch_size // seq_length
                a_out = a_out.reshape(num_seqs, seq_length, -1)

                if self.rnn_name == 'sru':
                    a_out = a_out.transpose(0, 1)

                ################# New RNN
                if len(states) == 2:
                    a_states = states[0].reshape(num_seqs, seq_length, -1)
                else:
                    a_states = states[:2].reshape(num_seqs, seq_length, -1)
                # a_states[:] = 0
                a_out, a_states = self.a_rnn(a_out, a_states[:, 0:1].transpose(0, 1).contiguous())
                ################ Old RNN
                # if len(states) == 2:	
                #     a_states = states[0]	
                # else:	
                #     a_states = states[:2]	
                # a_out, a_states = self.a_rnn(a_out, a_states)

                if self.rnn_name == 'sru':
                    a_out = a_out.transpose(0, 1)
                else:
                    if self.rnn_ln:
                        a_out = self.a_layer_norm(a_out)

                a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)

                if type(a_states) is not tuple:
                    a_states = (a_states,)

                if self.is_rnn_before_mlp:
                    a_out = self.actor_mlp(a_out)

                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, a_states

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits, a_states

                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    if self.space_config['fixed_sigma']:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))

                    return mu, sigma, a_states

            else:
                actor_input = torch.cat([other_obs, img_out], dim=-1)
                a_out = self.actor_mlp(actor_input)
                
                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits

                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    if self.space_config['fixed_sigma']:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))
                    if return_extra:
                        extra_dict['img_out'] = img_out
                        if self.use_unrealego:
                            extra_dict['heat_out'] = heat_out
                        return mu, sigma, extra_dict
                    else:
                        return mu, sigma
            return


        