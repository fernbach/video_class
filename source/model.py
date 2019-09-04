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

    def __init__(self, metadata, folder, downsample=1, framecount=6):
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
        #emb = F.concat(*[F.max(self.emb(ts), axis=0).expand_dims(axis=0) for ts in x], dim=0)
        emb = gluon.rnn.LSTM(100)
        e1 = self.fc1(emb)
        e1 = self.dropout_1(e1)
        Y = self.fc2(e1)

        return Y
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--backbone', type=str, default='resnet18_v2')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--ttsplit', type=float, default=0.7)
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--downsample', type=int, default=2)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--loadworkers', type=int, default=6)
    parser.add_argument('--fc', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.3)

    # input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    parser.add_argument('--rgb', type=str, default=os.environ.get('SM_CHANNEL_RGB'))

    args, _ = parser.parse_known_args()

    # INSTANCIATE DATA PIPELINE ----------------
    framecount = args.frames

    # datasets
    trainset = ImageSeqDataset(folder=args.rgb,downsample=args.downsample, framecount=framecount, metadata=args.train+'/train_labels.csv')
    testset = ImageSeqDataset(folder=args.rgb,downsample=args.downsample, framecount=framecount, metadata=args.test+'/test_labels.csv')
    valset = ImageSeqDataset(folder=args.rgb,downsample=args.downsample, framecount=framecount, metadata=args.val+'/val_labels.csv')

    # dataloaders
    train_loader = gluon.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.loadworkers)

    val_loader = gluon.data.DataLoader(
        dataset=valset,
        batch_size=len(valset),
        last_batch='rollover',
        shuffle=False,
        num_workers=args.loadworkers)

    test_loader = gluon.data.DataLoader(
        dataset=testset,
        batch_size=len(testset),
        last_batch='rollover',
        shuffle=False,
        num_workers=args.loadworkers)

    net = PoolingClassifier(num_classes=10, backbone=args.backbone, ctx=mx.gpu(), fc_width=args.fc)

    ctx = mx.gpu()

    net.fc1.initialize(mx.init.Xavier(), ctx=ctx)
    net.fc2.initialize(mx.init.Xavier(), ctx=ctx)
    net.collect_params().reset_ctx(ctx)

    net.summary(mx.nd.random.uniform(shape=(1, framecount, 3, 224, 224)).as_in_context(ctx))

    trainer = gluon.Trainer(
        params=net.collect_params(),
        optimizer=mx.optimizer.create('adam', multi_precision=True, learning_rate=args.lr))

    metric = mx.metric.Accuracy()
    loss_function = gluon.loss.SoftmaxCrossEntropyLoss()


    def perf(loader, net):

        testmetric = mx.metric.Accuracy()
        for inputs, labels in loader:
            # Possibly copy inputs and labels to the GPU
            inputs = inputs.astype('float16').as_in_context(ctx)
            labels = labels.astype('float16').as_in_context(ctx)
            testmetric.update(labels, net(inputs))

        _, value = testmetric.get()
        return value


    net.cast('float16')

    num_epochs = args.epochs

    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            # Possibly copy inputs and labels to the GPU
            inputs = inputs.astype('float16').as_in_context(ctx)
            labels = labels.astype('float16').as_in_context(ctx)

            with autograd.record():
                outputs = net(inputs)
                loss = loss_function(outputs, labels)

            # Compute gradients by backpropagation and update the evaluation metric
            loss.backward()
            metric.update(labels, outputs)

            # Update the parameters by stepping the trainer; the batch size
            # is required to normalize the gradients by `1 / batch_size`.
            trainer.step(batch_size=inputs.shape[0])

        # Print the evaluation metric and reset it for the next epoch
        name, acc = metric.get()
        print('After {} epoch : {} = {}'.format(epoch + 1, name, acc))
        train_acc.append(acc)

        metric.reset()

        val_perf = perf(val_loader, net)
        print('Test: accuracy: {}'.format(val_perf))
        val_acc.append(val_perf)