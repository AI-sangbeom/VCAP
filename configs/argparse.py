import argparse
from datetime import datetime


def parse_args():
    now = datetime.now()
    
    parser = argparse.ArgumentParser(description="codes for ddp")
    parser.add_argument(
        '--cfg',
        help='decide which cfg to use',
        default='configs/config_files/basic.yaml',
        type=str        
    )
    parser.add_argument(
        '--outdir',
        help='decide output directory',
        default='output',
        type=str        
    )
    parser.add_argument(
        '--time',
        help='time setting please typing like YYMMDD_HHMM',
        default= now.strftime('%y%m%d_%H%M'),
        type=str
    )
        
    return parser.parse_known_args()[0]

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()