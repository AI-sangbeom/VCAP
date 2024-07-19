import os
import numpy as np
import random
import torch
import torch.distributed as dist
import shutil

METHODS = ['FT', 'HT', 'VCAP', 'VP', 'VPT']

class io():
    def __init__(self):
        pass

    def line(self):
        return '================================================================================'

    def line_test(self):
        return '================================== model test =================================='

    def line_train(self):
        return '================================= model train =================================='

    def set_dir(self, env):
        cfg = env.cfg
        
        output_file = env.args.outdir

        if cfg.method in METHODS:
            dir0 = os.path.join(output_file, cfg.method)
            shutil.copytree('models', os.path.join(dir0, 'models'), dirs_exist_ok=True)
        else:
            raise Exception(f'There is not "{cfg.method}" in methods')

        dir1 = os.path.join(dir0, cfg.model.name)
        dir2 = os.path.join(dir1, cfg.data.dataset)

        self.output_dir = dir2
        self.base_path = os.path.join(self.output_dir, env.args.time)
        self.checkpoint = self.base_path+'_best.pth'
        self.log_path = self.base_path+'_log.txt'
        if env.p:
            if not os.path.exists(dir0): os.mkdir(dir0)
            if not os.path.exists(dir1): os.mkdir(dir1)
            if not os.path.exists(dir2): os.mkdir(dir2)


    def base_log(self, cfg, args):
        text =  self.line()+\
                f'\n config file : {args.cfg}\
                  \n method : {cfg.method}\
                  \n dataset : {cfg.data.dataset}\
                  \n model : {cfg.model.name}\
                  \n optimizer : {cfg.train.optimizer}\
                  \n learning rate : {cfg.train.lr}\
                  \n drop rate : {cfg.train.dropout}\
                  \n batch size : {cfg.train.batch}\n'+\
                self.line()+\
                f'\n\n model output will be saved in "{self.output_dir}"\n'
        self.log(text)
        
    def param_log(self, env, model):
        if env.p: 
            total = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
            p_classifier = sum(p.numel() for p in model.module.head.parameters() if p.requires_grad)
            p_prompter = total - p_classifier
            text_p = ' prompter : {0:,d}'.format(p_prompter)
            text_c = ' classifier : {0:,d}'.format(p_classifier)
            print(env.line())
            print('\n Number of train parameters\n')
            text_t = ' Total : {0:,d}\n'.format(total)
            text = f'{text_p}\n{text_c}\n{text_t}'
            self.log(text)

    def log(self, text):
        print(text)
        with open(self.log_path, 'a+') as f:
            f.write(text+'\n')

    def train_log(self, epoch, ta, tl, va, vl):
        text = '[Epoch %d] train : acc=%.2f, loss=%.2f | valid : acc=%.2f, loss=%.2f\n' %(epoch+1, ta, tl, va, vl)
        with open(self.log_path, 'a+') as f:
            f.write(text)


class Env(io):

    def __init__(self, cfg, args):
        super(Env, self).__init__()
        self.cfg = cfg
        self.args = args
        self.p = True
        self.device = None
        self.num_gpus = os.environ['NUM_GPUS']
        

    def set_seed(self, seed=42):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        os.environ['PYTHONHASHSEED']=str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)

    def set_ddp(self):
        dist.init_process_group(backend='nccl')
        self.rank = dist.get_rank()
        torch.cuda.set_device(self.rank)
        torch.distributed.barrier()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.device = torch.cuda.current_device()
        self.p = self.rank == 0        

    def set_cuda(self):
        self.rank = 1
        self.world_size = 1
        self.local_rank = 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.p = True

def init_env(env, seed=42):
    
    env.set_seed(seed)
    if env.num_gpus != '1':
        env.set_ddp()
    else:
        env.set_cuda()
    
    if env.p:
        print()
        print(env.line())
        print(' number of gpus :', env.num_gpus)

def init_log(env):
    env.set_dir(env)
    if env.p:
        env.base_log(env.cfg, env.args)
    