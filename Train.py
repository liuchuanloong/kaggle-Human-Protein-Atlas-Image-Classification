from Dataset import DatasetReader
from contextlib import contextmanager
import time
from Model_c import model50A_DeepSupervion, model50A_RFClass, model50A_DenseASPP, model101A_DenseASPP


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

##############################
PATH_TO_IMAGES = './data/full_train/'
PATH_TO_META = './data/full_dev_train.csv'

# LOAD_PATHS = None
##############################
LOSS = 'bceloss'
OPTIMIZER = 'SGD'
PRETRAINED = True
N_EPOCH = 40
NET = model101A_DenseASPP
##############################
BATCH_SIZE = 8
NUM_WORKERS = 0
DEV_MODE = False
SEED = 666
SIZE = 512
SHUFFLE = True
N_SPLITS = 3
###########OPTIMIZER###########
LR = 1e-2
USE_SCHEDULER = 'CosineAnneling'
MILESTONES = [20, 40, 75]
GAMMA = 0.5
PATIENCE = 10
T_MAX = 18
T_MUL = 1
LR_MIN = 0
# ESP = 6
VAL_MODE = 'min'
##############################
COMMENT = 'SGDR (Tmax18, Tmul1), Lovasz, relu, pretrained'

# MEAN = [0.049116287, 0.050290816, 0.050428327, 0.050616704]
# STD = [0.0099924, 0.010046058, 0.0100394245, 0.010036305]

DatasetReader = DatasetReader(random_state=SEED, n_split=N_SPLITS, size=SIZE, dev_mode=DEV_MODE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=SHUFFLE)
loaders = DatasetReader.train_reader(train_path=PATH_TO_META, img_path=PATH_TO_IMAGES)


for i, (train_loader, val_loader, weights) in enumerate(loaders, 1):
    with timer('Fold {}'.format(i)):
        if i < 3:
            continue
        net = NET(lr=LR, pretrained=PRETRAINED, fold=i, comment=COMMENT, val_mode=VAL_MODE)
        net.define_criterion(LOSS, weights)
        net.create_optmizer(optimizer=OPTIMIZER, use_scheduler=USE_SCHEDULER, milestones=MILESTONES,
                            gamma=GAMMA, patience=PATIENCE, T_max=T_MAX, T_mul=T_MUL, lr_min=LR_MIN)

        # if LOAD_PATHS is not None:
        #     if LOAD_PATHS[i - 1] is not None:
        #         net.load_model(LOAD_PATHS[i - 1])

        net.train_network(train_loader, val_loader, n_epoch=N_EPOCH, earlystopping_patience=None)
        # net.plot_training_curve(show=True)



