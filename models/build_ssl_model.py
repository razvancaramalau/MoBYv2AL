from .moby import MoBY, DINOHead, MultiCropWrapper
import models.resnet_o as resnet
from .builder import MoCo
from yacs.config import CfgNode as CN
from .lenet import LeNet5

def build_model(no_classes, model_type, model_encoder, batch_size, no_of_labelled):

    config = CN()
    # config.MODEL = model_encoder
    config.DATA_BATCH_SIZE = batch_size
    config.DATA_TRAINING_IMAGES = len(no_of_labelled)
    config.TRAIN_START_EPOCH = 0
    config.TRAIN_EPOCHS = 222

    if model_encoder == 'vgg16':
        enc = resnet.dnn_16enc
        clsf = resnet.VGGNet2C(512, no_classes)
    if model_encoder == 'wideresnet28':
        enc = resnet.Wide_ResNet28
        clsf = resnet.ResNetC(640, num_classes=no_classes)
    if model_encoder == 'resnet18':
        enc = resnet.ResNet18E
        clsf = resnet.ResNetC(512, num_classes=no_classes)
    if model_encoder == 'lenet5':
        enc = LeNet5
        clsf = resnet.VGGNet2C(400, no_classes)
    if model_type == 'moby':
        encoder = enc()      
        encoder_k = enc()
        model = MoBY(
            cfg=config,
            encoder=encoder,
            encoder_k=encoder_k,
            classifier=clsf,
            contrast_momentum=0.99,
            contrast_temperature=0.2,
            contrast_num_negative=4096,
            proj_num_layers=1,
            pred_num_layers=1,
        )
    elif model_type == 'moco':
        model = MoCo(base_encoder=enc, dim=512, K=65536)
    elif model_type == 'linear':
        model = enc(
            num_classes=no_classes,
        )
    else:
        raise NotImplementedError(f'--> Unknown model_type: {model_type}')

    return model
