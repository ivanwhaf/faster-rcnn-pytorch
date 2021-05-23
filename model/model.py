from torch import nn

from roi_pooling import RoIPolling
from rpn import RPN


class FasterRCNN(nn.Module):
    """FasterRCNN model structure
    FasterRCNN = feature-extractor + rpn + roi-pooling + fast-rcnn
    """

    def __init__(self, training, num_classes):
        super(FasterRCNN, self).__init__()
        self.training = training
        self.extractor = VGGNet()
        self.rpn = RPN(training=training)
        self.roi_pooling = RoIPolling()

    def forward(self, x):
        feats = self.extractor(x)  # feature maps
        rpn_proposals, rpn_loss_cls, rpn_loss_loc = self.rpn(feats)  # rpn outputs
        roi_outputs = self.roi_pooling(rpn_proposals)  # roi_pooling outputs

        if self.training:
            # backward loss
            pass

        return x


class VGGNet(nn.Module):
    """
    VGG16 net, default input shape: 227*227
    """

    def __init__(self, num_classes=10):
        super(VGGNet, self).__init__()

        # 800*600*3 -> 400*300*64
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        # 400*300*64 -> 200*150*128
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        # 200*150*128 -> 100*75*256
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        # 100*75*256 -> 50*37*512
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes)
        # )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x
