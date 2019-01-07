import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from Augmentation import *
from sklearn.preprocessing import MultiLabelBinarizer
import pathlib
import torchvision.transforms as transforms
import torch
import PIL
from torch.utils.data import WeightedRandomSampler
LABEL_MAP = {
0: "Nucleoplasm" ,
1: "Nuclear membrane"   ,
2: "Nucleoli"   ,
3: "Nucleoli fibrillar center",
4: "Nuclear speckles"   ,
5: "Nuclear bodies"   ,
6: "Endoplasmic reticulum"   ,
7: "Golgi apparatus"  ,
8: "Peroxisomes"   ,
9:  "Endosomes"   ,
10: "Lysosomes"   ,
11: "Intermediate filaments"  ,
12: "Actin filaments"   ,
13: "Focal adhesion sites"  ,
14: "Microtubules"   ,
15: "Microtubule ends"   ,
16: "Cytokinetic bridge"   ,
17: "Mitotic spindle"  ,
18: "Microtubule organizing center",
19: "Centrosome",
20: "Lipid droplets"   ,
21: "Plasma membrane"  ,
22: "Cell junctions"   ,
23: "Mitochondria"   ,
24: "Aggresome"   ,
25: "Cytosol" ,
26: "Cytoplasmic bodies",
27: "Rods & rings"}


def augmentator(img):
    c = np.random.choice(2)
    if c == 0:
        pass
    if c == 1:
        c_ = np.random.choice(7)
        img = do_flip_transpose(img, type=c_)
    if c == 2:
        img = do_shift_scale_rotate(img, dx=0, dy=0, scale=1, angle=np.random.uniform(0, 1))  # 10
    if c == 3:
        img = do_elastic_transform(img, grid=10, distort=np.random.uniform(0, 0.1))  # 0.10
    return img

class DatasetReader():
    def __init__(self, random_state=1234, size=512, n_split=5, dev_mode=False, batch_size=32, num_workers=8, shuffle=True):
        self.SEED = random_state
        self.n_split = n_split
        self.DEV_MODE = dev_mode
        self.BATCH_SIZE = batch_size
        self.NUM_WORKERS = num_workers
        self.shuffle = shuffle
        self.size = size

    def image_transform(self, img):
        # img = self.normalize(img)
        mean = [10.281611, 10.265814, 10.265338, 10.269585]
        std = [24.66839, 24.687712, 24.719505, 24.727997]
        # mean = [9.591226, 9.568537, 9.563865, 9.567095]
        # std = [3.0389864, 3.046309, 3.052598, 3.0547194]
        img = cv2.resize(img, (self.size, self.size))
        img = np.array(img).transpose(2,0,1).astype('float32')
        img = torch.from_numpy(img)
        img = transforms.functional.normalize(img, mean=mean, std=std)
        return img

    def _normalize(self, im):
        max = np.max(im)
        min = np.min(im)
        if (max - min) > 0:
            im = (im - min) / (max - min)
        return im
    def normalize(self, img):
        img1, img2, img3, img4 = img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3]
        img1 = self._normalize(img1)
        img2 = self._normalize(img2)
        img3 = self._normalize(img3)
        img4 = self._normalize(img4)
        img = np.stack((img1, img2, img3, img4)).transpose(1,2,0)
        return img

    def train_reader(self, train_path, img_path):
        df = pd.read_csv(train_path)
        print('Train Sample total number: %d' % (df.shape[0]))
        trainlabels = df

        def fill_targets(row):
            tmp = np.array(row.Target.split(" ")).astype(np.int)
            for num in tmp:
                # name = LABEL_MAP[int(num)]
                row.loc[num] = 1
            return row

        def create_class_weight(labels_dict, mu=0.5):
            total = 0
            for k, v in labels_dict.items():
                total += v
            #     print(total)
            keys = labels_dict.keys()
            class_weight = dict()
            class_weight_log = dict()

            for key in keys:
                score = total / float(labels_dict[key])
                score_log = math.log(mu * total / float(labels_dict[key]))
                class_weight[key] = round(score, 2) if score > 1.0 else round(1.0, 2)
                class_weight_log[key] = round(score_log, 2) if score_log > 1.0 else round(1.0, 2)

            return class_weight_log

        for key in LABEL_MAP.keys():
            trainlabels[key] = 0
        trainlabels = trainlabels.apply(fill_targets, axis=1)

        train_test_split = MultilabelStratifiedKFold(n_splits=self.n_split, random_state=self.SEED)
        kfold_index = train_test_split.split(trainlabels.Id, trainlabels.iloc[:, 2:])
        loaders = []
        for i, (df_train_index, df_test_index) in enumerate(kfold_index, 1):
            df_train, df_test = trainlabels.iloc[df_train_index], trainlabels.iloc[df_test_index]
            if self.DEV_MODE:
                df_train = df_train[:200]
                df_test = df_test[:50]
            df_train_count = df_train.copy()
            target_counts = df_train_count.drop(["Id", "Target"], axis=1).sum(axis=0)
            weights = create_class_weight(dict(target_counts))
            weights = pd.DataFrame(weights, index=['num']).values
            weights = torch.FloatTensor(weights)
            # nsample = pd.value_counts(df_train_count.Target)
            # nweights = dict(1.0 / nsample)
            # nweights = create_class_weight(dict(nsample), mu=0.5)
            # nweights = pd.DataFrame(nweights, index=['weights']).T
            # samples_weight = np.array([nweights.loc[t].weights for t in df_train.Target])
            # samples_weight = torch.FloatTensor(samples_weight)
            # sampler = WeightedRandomSampler(samples_weight, samples_weight.shape[0])

            # Prepare datasets and loaders

            gtrain = MultiBandMultiLabelDataset(df_train, base_path=img_path, image_transform=self.image_transform, augmentator=augmentator)
            gtest = MultiBandMultiLabelDataset(df_test, base_path=img_path, image_transform=self.image_transform)

            train_load = DataLoader(gtrain, collate_fn=gtrain.collate_func, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS,
                                    shuffle=True)
            test_load = DataLoader(gtest, collate_fn=gtest.collate_func, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS)
            loaders.append((train_load, test_load, weights))
        return loaders

    def test_reader(self, sub_path, sub_img_path, TRAIN_MODE=False):
        df_submission = pd.read_csv(sub_path)
        print('Submission Sample total number: %d' %(df_submission.shape[0]))
        if self.DEV_MODE:
            df_submission = df_submission[:50]

        gsub = MultiBandMultiLabelDataset(df_submission, base_path=sub_img_path, train_mode=TRAIN_MODE,
                                          image_transform=self.image_transform)
        submission_load = DataLoader(gsub, collate_fn=gsub.collate_func, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS)
        return submission_load, df_submission

class MultiBandMultiLabelDataset(Dataset):
    BANDS_NAMES = ['_red.png', '_green.png', '_blue.png', '_yellow.png']

    def __len__(self):
        return len(self.images_df)

    def __init__(self, images_df,
                 base_path,
                 image_transform,
                 augmentator=None,
                 train_mode=True
                 ):
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)

        self.images_df = images_df.copy()
        self.image_transform = image_transform
        self.augmentator = augmentator
        self.images_df.Id = self.images_df.Id.apply(lambda x: base_path / x)
        self.mlb = MultiLabelBinarizer(classes=list(LABEL_MAP.keys()))
        self.train_mode = train_mode

    def __getitem__(self, index):
        y = None
        index = int(index)
        X = self._load_multiband_image(index)
        if self.train_mode:
            y = self._load_multilabel_target(index)

        # augmentator can be for instance imgaug augmentation object
        if self.augmentator is not None:
            X = self.augmentator(X)

        X = self.image_transform(X)

        return X, y

    def _load_multiband_image(self, index):
        row = self.images_df.iloc[index]
        image_bands = []
        for band_name in self.BANDS_NAMES:
            p = str(row.Id.absolute()) + band_name
            pil_channel = PIL.Image.open(p).convert('L')
            image_bands.append(pil_channel)

        # lets pretend its a RBGA image to support 4 channels
        band4image = PIL.Image.merge('RGBA', bands=image_bands)
        band4image = np.array(band4image)
        return band4image

    def _load_multilabel_target(self, index):
        return list(map(int, self.images_df.iloc[index].Target.split(' ')))

    def collate_func(self, batch):
        labels = None
        images = [x[0] for x in batch]

        if self.train_mode:
            labels = [x[1] for x in batch]
            labels_one_hot = self.mlb.fit_transform(labels)
            labels = torch.FloatTensor(labels_one_hot)

        return torch.stack(images), labels

    def visualize_sample(self, sample_size):
        samples = np.random.choice(self.df['id'].values, sample_size)
        self.df.set_index('id', inplace=True)
        fig, axs = plt.subplots(2, sample_size)
        for i in range(sample_size):
            im = cv2.imread(self.df.loc[samples[i], 'im_path'], cv2.IMREAD_COLOR)
            mask = cv2.imread(self.df.loc[samples[i], 'mask_path'], cv2.IMREAD_GRAYSCALE)
            print('Image shape: ', np.array(im).shape)
            print('Mask shape: ', np.array(mask).shape)
            axs[0, i].imshow(im)
            axs[1, i].imshow(mask)


if __name__ == '__main__':
    # Prepare dataframe files

    PATH_TO_IMAGES = './data/train/'
    PATH_TO_TEST_IMAGES = './data/test/'
    PATH_TO_META = './data/full_dev_train.csv'
    SAMPLE_SUBMI = './data/sample_submission.csv'

    SEED = 666
    DEV_MODE = False
    SIZE = 256
    SHUFFLE = True
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    N_SPLITS = 5
    SIZE = 256
    # MEAN = [0.049116287, 0.050290816, 0.050428327, 0.050616704]
    # STD = [0.0099924, 0.010046058, 0.0100394245, 0.010036305]

    DatasetReader = DatasetReader(random_state=SEED, n_split=N_SPLITS, size=SIZE, dev_mode=DEV_MODE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=SHUFFLE)
    loader = DatasetReader.train_reader(train_path=PATH_TO_META, img_path=PATH_TO_IMAGES)

    for i, lists in enumerate(loader, 1):
        if i > 1:
            break
        for j, imgs in enumerate(lists[0]):
            # img = imgs[0].data.numpy().transpose((1, 2, 0))
            img = imgs[0].numpy().transpose((0, 2, 3, 1))
            for img in img:
                plt.imshow(img[:,:,0], cmap='Greens')
                plt.show()
                plt.imshow(img[:,:,1], cmap='Reds')
                plt.show()
                plt.imshow(img[:,:,2], cmap='Oranges')
                plt.show()
                plt.imshow(img[:,:,3], cmap='Blues')
                plt.show()

    # DatasetReader = DatasetReader(random_state=SEED, n_split=N_SPLITS, size=SIZE, dev_mode=DEV_MODE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=SHUFFLE)
    # loader, df_sub= DatasetReader.test_reader(sub_path=SAMPLE_SUBMI, sub_img_path=PATH_TO_TEST_IMAGES)
    # for i, lists in enumerate(loader):
    #     if i > 1:
    #         break
    #     for j, imgs in enumerate(lists[0]):
    #         # img = imgs[0].data.numpy().transpose((1, 2, 0))
    #         img = imgs.numpy().transpose((1, 2, 0))
    #         plt.imshow(img[:,:,0], cmap='Greens')
    #         plt.show()
    #         plt.imshow(img[:,:,1], cmap='Reds')
    #         plt.show()
    #         plt.imshow(img[:,:,2], cmap='Blues')
    #         plt.show()
    #         plt.imshow(img[:,:,3], cmap='Oranges')
    #         plt.show()
