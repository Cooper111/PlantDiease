import os

img_width, img_height = 299, 299
num_channels = 3
train_data = 'F:\\Plant_disease\\ai_challenger_pdr2018_trainingset_20180905\\AgriculturalDisease_trainingset'
valid_data = 'F:\\Plant_disease\\ai_challenger_pdr2018_validationset_20180905\\AgriculturalDisease_validationset'
test_a_data = 'F:\\Plant_disease\\ai_challenger_pdr2018_testA_20180905\\AgriculturalDisease_testA\\test\\images'
train_annot = os.path.join(train_data, 'AgriculturalDisease_train_annotations.json')
valid_annot = os.path.join(valid_data, 'AgriculturalDisease_validation_annotations.json')
train_image_folder = os.path.join(train_data, 'images')
valid_image_folder = os.path.join(valid_data, 'images')
test_a_image_folder = test_a_data
num_classes = 61
num_train_samples = 32739
num_valid_samples = 4982
verbose = 1
batch_size = 16
num_epochs = 1000
patience = 50
FREEZE_LAYERS = 2
dropout_keep_prob = 0.8
dropout_rate = 0.2
