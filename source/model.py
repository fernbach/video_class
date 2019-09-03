import mxnet as mx
from mxnet import gluon, image, nd, autograd
from mxnet.gluon.model_zoo import vision as models


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