import argparse
import os
import json
import re
import readline
import subprocess as sb
import sys

sb.call([sys.executable, '-m', 'pip', 'install', '-U', 'pandas'])
sb.call([sys.executable, "-m", "pip", "install", "awscli"])

import mxnet as mx
from mxnet import gluon, image, nd, autograd
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
import pandas as pd

def img_prep(img):
    """prepare an RGB image NDArray for ImageNet pre-trained inference"""
    data = mx.image.resize_short(img, 256).astype('float32')
    data, _ = mx.image.center_crop(data, (224,224))
    data = data.transpose((2,0,1))
    return data


class ImageSeqDataset(gluon.data.Dataset):
    """
    Custom Dataset to handle the UTK image sequence dataset json file
    """

    def __init__(self, metadata, folder='RGB', downsample=1, framecount=6):
        """
        Parameters
        ---------
        folder: folder storing images. This is the folder containing the path mentioned in the metadata file
        metadata: action metadata file as structured above
        downsample: downsample factor
        framecount: how many frames to keep. Crop after that limit
        records: index of records to select. Use this for train-test split
        """
        self.folder = folder
        self.ds = downsample
        self.fct = framecount

        self.annotdf = pd.read_csv(metadata)

    def __getitem__(self, idx):
        """
        Parameters
        ---------
        idx: int, index requested

        Returns
        -------
        Tensor: nd.NDArray of seq of images
        label: np.NDArray bounding box labels of the form [[x1,y1, x2, y2, class], ...]
        """

        picdir = self.folder + '/' + self.annotdf.loc[idx]['path']

        # list available frames
        pathframes = [int(re.sub("[^0-9]", "", pic))
                      for pic in os.listdir(picdir)]

        allframes = [f for f in pathframes if (self.annotdf.iloc[idx]['fstart'] <= f
                                               and f <= self.annotdf.iloc[idx]['fstop'])]
        allframes.sort()

        frames = allframes[:: self.ds][:self.fct]
        pics = ['colorImg' + str(f) + '.jpg' for f in frames]

        # if not enough frames, repeating the last one
        while len(pics) < self.fct:
            pics = pics + [pics[-1]]

        # return a tensor with all prepared images concatenated
        tensor = nd.concat(*[nd.expand_dims(img_prep(image.imread(picdir + '/' + pic)), axis=0)
                             for pic in pics], dim=0)

        labelid = self.annotdf.iloc[idx]['labelid']

        return tensor, labelid

    def __len__(self):
        return len(self.annotdf)


class PoolingClassifier(gluon.HybridBlock):
    """this network runs a softmax on top of the average-pooled frame imagenet embeddings"""

    def __init__(self, num_classes, backbone, fc_width, ctx, dropout_p=0.3):
        super(PoolingClassifier, self).__init__()

        self.num_classes = num_classes
        self.backbone = backbone
        self.fc_width = fc_width
        self.dropout_p = dropout_p

        with self.name_scope():
            self.emb = models.get_model(name=self.backbone, ctx=ctx, pretrained=True).features
            self.dropout_1 = gluon.nn.Dropout(self.dropout_p)
            self.fc1 = gluon.nn.Dense(self.fc_width, activation='relu')
            self.fc2 = gluon.nn.Dense(self.num_classes)

    def hybrid_forward(self, F, x):
        emb = F.concat(*[F.max(self.emb(ts), axis=0).expand_dims(axis=0) for ts in x], dim=0)

        e1 = self.fc1(emb)
        e1 = self.dropout_1(e1)
        Y = self.fc2(e1)

        return Y
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.1)

    # input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()

    # ... load from args.train and args.test, train a model, write model to args.model_dir.