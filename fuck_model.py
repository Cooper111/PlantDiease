from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19

from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation
from keras.models import Model
from config import img_height, img_width, num_channels, num_classes, FREEZE_LAYERS, dropout_rate
import os
import numpy as np
from keras import Input
from keras import layers

def get_best_model(list_name):
    import re
    pattern = 'model.(?P<epoch>\d+)-(?P<val_acc>[0-9]*\.?[0-9]*).hdf5'
    p = re.compile(pattern)
    files = [f for f in os.listdir(list_name) if p.match(f)]
    filename = None
    epoch = None
    if len(files) > 0:
        epoches = [p.match(f).groups()[0] for f in files]
        accs = [float(p.match(f).groups()[1]) for f in files]
        best_index = int(np.argmax(accs))
        filename = os.path.join(list_name, files[best_index])
        epoch = int(epoches[best_index])
        print('loading best model: {}'.format(filename))
    return filename, epoch

def build_model_VGG19():
    base_model = VGG19(input_shape=(img_height, img_width, num_channels), weights='imagenet',
                                   include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    for layer in model.layers:
        layer.trainable = False
    return model

def build_model_InceptionResNetV2():
    base_model = InceptionResNetV2(input_shape=(img_height, img_width, num_channels), weights='imagenet',
                                   include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    for layer in model.layers:
        layer.trainable = False

    return model

def build_model_Xception():
    base_model = Xception(input_shape=(img_height, img_width, num_channels), weights='imagenet',
                                   include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    for layer in model.layers:
        layer.trainable = False

    return model

def bulid_model():
    cnn_no_vary = False

    input_layer=Input(shape=(299,299,3))

    xception = build_model_Xception()
    vgg19 = build_model_VGG19()
    #inceptionresnetv2 = build_model_InceptionResNetV2()


#     for i,layer in enumerate(inceptionresnetv2.layers):
#         inceptionresnetv2.layer[i].trainable = False

#     for i,layer in enumerate(xception.layers):
#         xception.layers[i].trainable=False
#     for i,layer in enumerate(vgg19.layers):
#         vgg19.layers[i].trainable=False


    best_vgg19_weights, vgg19_epoch = get_best_model('/home/yjz/lhj/others_model/vgg/models')
    best_xception_weights, xception_epoch = get_best_model('/home/yjz/lhj/others_model/Crop-Disease-Detection/models')
    #best_inceptionresnetv2_weights, inceptionresnetv2_epoch = get_best_model('/home/yjz/lhj/others_model/Inception/models')

    vgg19.load_weights(best_vgg19_weights, by_name=True)
    xception.load_weights(best_xception_weights, by_name=True)
    #inceptionresnetv2.load_weights(best_inceptionresnetv2_weights, by_name=True)

    #inceptionresnetv2 = inceptionresnetv2(input_layer)
    xception = xception(input_layer)
    vgg19 = vgg19(input_layer)

    #t=layers.Concatenate(axis=-1)([xception,vgg19])#inceptionresnetv2,
    t=layers.Concatenate(axis=-1)([xception,vgg19])
    # x = Dense(512, name='Logits')(t)
    x = Dense(512, name='Logits')(t)
    x = Dropout(dropout_rate, name='Dropout')(x)
    top_model = Dense(num_classes, activation='softmax')(x)
    model=Model(inputs=input_layer,outputs=top_model)
    print(model.summary())
    return model