import os
from Dataset import DatasetReader
import numpy as np
from Model_c import UNetResNet34_SE, model50A_DeepSupervion, model50A_DenseASPP, model101A_DenseASPP

# ******************************************************************************
def load_net_and_predict(net, submission_load, LOAD_PATH):

    best_th = net.load_model(LOAD_PATH)
    best_th = np.array(best_th) - 0.05
    print('Valid Best Threshold: %s' %best_th)
    preds = net.predict(submission_load, best_th)

    return preds


def make_submission_file(sample_submission_df, predictions):
    submissions = []
    for row in predictions:
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)

    sample_submission_df['Predicted'] = submissions
    # sample_submission_df.to_csv('submission.csv', index=None)

    return sample_submission_df

if __name__ == '__main__':
    # **********************
    PATH_TO_TEST_IMAGES = './data/test/'
    SAMPLE_SUBMI = './data/sample_submission.csv'
    # ********************************
    SEED = 666
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    SIZE = 512
    DEV_MODE = False
    SHUFFLE = True
    N_SPLITS = 5
    # **************************************
    net = model101A_DenseASPP()
    NET_NAME = type(net).__name__
    # **************************************

    # MEAN = [0.049116287, 0.050290816, 0.050428327, 0.050616704]
    # STD = [0.0099924, 0.010046058, 0.0100394245, 0.010036305]
    DatasetReader = DatasetReader(random_state=SEED, n_split=N_SPLITS, size=SIZE, dev_mode=DEV_MODE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=SHUFFLE)
    submission_load, df_submission = DatasetReader.test_reader(sub_path=SAMPLE_SUBMI, sub_img_path=PATH_TO_TEST_IMAGES)


    # *********************************
    # *************************************
    LOAD_PATH = '/home/loong/loong/PROTEIN/Saves/model101A_DenseASPP/2018-12-29 10:27_Fold3_Epoach13_reset0_val0.175'

    ################################################
    pred = load_net_and_predict(net,
                              submission_load,
                              LOAD_PATH
                              )
    submission_file = make_submission_file(sample_submission_df=df_submission,
                                           predictions=pred)
    submission_file.to_csv(os.path.join(
        './Saves',
        NET_NAME,
        '{}_122801.csv'.format(NET_NAME)),
        index=False)