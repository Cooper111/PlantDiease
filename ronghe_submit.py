from keras import models
import os
from keras.preprocessing import image
import numpy as np
import json
import cv2
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation
from keras.models import Model
import config as cfg
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from fuck_model import bulid_model,get_best_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"



model = bulid_model()
best_model, epoch = get_best_model("/home/yjz/skw/ronghe_2/models")
model.load_weights(best_model)
model.summary()
sgd = optimizers.SGD(lr=2.5e-5, momentum=0.9, decay=1e-6, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


test_dir = '/home/yjz/lhj/DATA/test'
img_list = os.listdir(test_dir)
result = []
# test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
# test_generator = test_datagen.flow_from_directory(
#     "../DATA/",
#     target_size=(299, 299),
#     batch_size=1,
#     class_mode='binary'
#     )

i = 0

# for (img, label), image_name in zip(test_generator, test_generator.filenames):
#     image_name = image_name.split('/')[-1]
for img in img_list:
    #=====================Fun1===================
    img_path = os.path.join(test_dir, img)
    img_data = image.load_img(img_path, target_size=(299, 299))
    img_tensor = image.img_to_array(img_data)
    img_tensor = preprocess_input(img_tensor)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    #=====================Fun2=====================
    # img_path = os.path.join(test_dir, img)
    # image = cv.imread(img_path)
    # image = cv.resize(image, (cfg.img_height, cfg.img_width), cv.INTER_CUBIC)
    # rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # rgb_img = np.expand_dims(rgb_img, 0).astype(np.float32)
    # rgb_img = preprocess_input(rgb_img)

    i += 1
    if i % 100 == 0:
        print(i)
        #print(image_name)

    # if i == 4960:
    #     break

    preds = model.predict(img_tensor)
    #prob = np.max(preds)
    class_id = int(np.argmax(preds))
    # predictions = model.predict(img_tensor).tolist()
    # predictions = predictions[0]
        
    # fuck = predictions.copy()
    # predictions.sort(reverse=True)
    # ass = [fuck.index(x) for x in predictions]
    # ass = np.array(ass)

    #temp = {}
    #temp['image_id'] = img
    #temp['disease_class'] = int(np.argmax(ass))

    #result.append(temp)
    result.append({'image_id': img, 'disease_class': class_id})



with open('submit.json', 'w') as f:
    #f.write(str(result))
    json.dump(result, f,ensure_ascii=False)
    print('write result json, num is %d' % len(result))
       

