import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from configs.argparse import parse_args, update_config
from configs.default import C as cfg
from lib.utils.environment import Env, init_env, init_log
from lib.data.dataloader import dataLoader
from models.utils.get_model import get_model
from lib.utils.trainer import Trainer

def main(args):
    env = Env(cfg, args)
    init_env(env)
    model = get_model(env)
    dataloader = dataLoader(env)
    trainer = Trainer(env, model, dataloader)
    init_log(env)
    env.param_log(env, model)

    trainer.train()
    

if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)
    main(args)