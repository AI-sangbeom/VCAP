from lib.utils.trainer_base import Base, optimizer, criterion, scheduler
from torch.cuda.amp import GradScaler

class Trainer(Base):

    def __init__(self, env, model, dataLoader):
        super(Trainer, self).__init__()
        cfg = env.cfg
        self.env = env
        self.model = model
        self.trainLoader = dataLoader[0]
        self.testLoader = dataLoader[1]
        self.optimizer = optimizer(cfg, self.model)
        self.criterion = criterion(cfg)
        self.scheduler = scheduler(cfg, self.optimizer)
        self.init_numerical_value()
        self.scaler = GradScaler()

    def train(self):
        t_cfg = self.env.cfg.train
        env = self.env        
        if env.p: env.log(env.line_train())
        for epoch in range(t_cfg.epoch):
            self.epoch = epoch
            self.model_train()
            self.model_valid()                
            self.save_best()
            if env.p: env.train_log(self.epoch, self.acc, self.loss, self.val_acc, self.val_loss)          
        if env.p: env.log(env.line())