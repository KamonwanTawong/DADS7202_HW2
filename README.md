# DADS7202

## INTRODUCTION
ตราประจำจังหวัดของไทย มีพัฒนาการมาจากตราประจำตำแหน่งของเจ้าเมืองในสมัยสมบูรณาญาสิทธิราชย์ และตราประจำธงประจำกองลูกเสือ 14 มณฑล 
ในสมัยรัชกาลที่ 6-7 ในสมัยที่จอมพลแปลก พิบูลสงครามเป็นนายกรัฐมนตรีนั้น รัฐบาลได้กำหนดให้แต่ละจังหวัดมีตราประจำจังหวัดของตนเองใช้เมื่อ พ.ศ.2483 
โดยกรมศิลปากรเป็นผู้ออกแบบตราตามแนวคิดที่แต่ละจังหวัดกำหนดไว้ ในปัจจุบันเมื่อมีการตั้งจังหวัดขึ้นใหม่ ก็จะมีการออกแบบตราประจำจังหวัดด้วยเสมอ แต่ตราของบางจังหวัดที่ใช้อยู่นั้นบางตราก็ไม่ใช่ตราที่กรมศิลปากรเป็นผู้ออกแบบ บางจังหวัดก็เปลี่ยนไปใช้ตราประจำจังหวัดเป็นแบบอื่น บางที่ลักษณะของตราก็เพี้ยนไปจากลักษณะที่กรมศิลปากรออกแบบไว้ แต่ยังคงลักษณะหลัก ๆ ของตราเดิม

ประเทศไทยมีทั้งหมด 77 จังหวัด จึงได้เก็บภาพแยกเป็น Folder ละ 1 จังหวัด โดยเก็บรวบรวมภาพให้ได้มากที่สุดเท่าที่รวบรวมได้ ซึ่งมีทั้งภาพที่นำมาจากอินเทอร์เน็ต 
โดยใช้ Pre-Trained Models 3 โมเดล คือ VGG-16, ResNet50 และ EfficientNet 

## DATA
https://drive.google.com/drive/folders/1wjw-2dWe0zl6D040VFI2jUTjYxOe5RA1?usp=sharing

## PREPARE DATA
นำเข้าชุดข้อมูลจาก Google Drive

```
data_path = pathlib.Path(r"/content/drive/MyDrive/Colab Notebooks/CNN/data")

all_images = list(data_path.glob(r'*/*.jpg')) + list(data_path.glob(r'*/*.jpeg')) + list(data_path.glob(r'*/*.png'))

images = []
labels = []

for item in all_images:
    path = os.path.normpath(item)
    splits = path.split(os.sep)
    if 'GT' not in splits[-2]:
        images.append(item)
        label = splits[-2]
        labels.append(label)
```
สร้าง data frame ของชุดข้อมูลจาก Google Drive ที่ประกอบด้วย Path ของรูปภาพแต่ละรูปและติด label ตามชื่อ Folder ให้กับภาพทั้งหมดเพื่อระบุจังหวัด

```
image_pathes = pd.Series(images).astype(str)
labels = pd.Series(labels)
dataframe =pd.concat([image_pathes, labels], axis=1)
dataframe.columns = ['images', 'labels']
dataframe.head()
```
<img width="411" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 16 33 19" src="https://user-images.githubusercontent.com/107698198/196654392-bef1824a-e678-44ac-a91b-d3e7cd5a5b5e.png">

แสดงตัวอย่างรูปภาพ
```
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15,10), subplot_kw={'xticks':[], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(dataframe.images[i]))
    ax.set_title(dataframe.labels[i])
plt.show()
```
<img width="876" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 16 39 49" src="https://user-images.githubusercontent.com/107698198/196655995-e7598bda-6463-4c6c-881c-e41250ef74d5.png">

แบ่งชุดของมูล Train, Validation และ Test
```
all_train, test = train_test_split(shuffled_dataframe, test_size=0.2, random_state=42)
train, val = train_test_split(all_train, test_size=0.3, random_state=42)
```
ปรับแต่งชุดรูปภาพที่ใช้สำหรับ Train, Validation และ Test ให้มีคุณลักษณะที่เหมือนกัน 
```
training_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
training_generator = training_data_gen.flow_from_dataframe(dataframe=train,
                                                          x_col='images', y_col='labels',
                                                          target_size=(224, 224),
                                                          color_mode='rgb',
                                                          class_mode='categorical',
                                                          batch_size=64)

val_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)
validation_generator = val_data_gen.flow_from_dataframe(dataframe=val,
                                                       x_col='images', y_col='labels',
                                                       target_size=(224, 224),
                                                       color_mode='rgb',
                                                       class_mode='categorical',
                                                       batch_size=64)

test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)
test_generator = test_data_gen.flow_from_dataframe(dataframe=test,
                                                  x_col='images', y_col='labels',
                                                  target_size=(224, 224),
                                                  color_mode='rgb',
                                                  class_mode='categorical',
                                                  batch_size=64,
                                                  shuffle=False)
```                                                 
<img width="525" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 16 46 57" src="https://user-images.githubusercontent.com/107698198/196657629-d4585779-9741-4258-8ceb-70eb2b2a7738.png">

## MODEL(1) VGG16 
### Import Model and Finetuning 

- Optimizer the RMSprop algorithm. <br />
- Learning Rate = 0.0001 <br />
- Loss ใช้ categorical crossentropy <br />
- Output Dense Layer เป็น 77 ตามจำนวน class <br />
- Metric ใช้ accuracy
- Activation ใช้ softmax เพื่อให้ output แสดงผลออกมาเพียง class เดียว

```
from tensorflow.keras.applications.vgg16 import VGG16
base_model = VGG16(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')
```
```
x = layers.Flatten()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(77, activation='softmax')(x)

model = tf.keras.models.Model(base_model.input, x)
model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss = 'categorical_crossentropy',metrics = ['acc'])
```
ทำการ Train model ทั้งหมด 50 ครั้ง

```
vgghist = model.fit(training_generator, validation_data = validation_generator, epochs = 50)
```
<img width="989" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 16 59 19" src="https://user-images.githubusercontent.com/107698198/196660497-b737165b-f7f5-4ca1-bbb1-58059a35d39d.png">

ผลจากการ Train พบว่า ค่าความแม่นยำสูงที่สุดมีค่า 0.6810 


```
model_train_loss = vgghist.history['loss']
model_val_loss = vgghist.history['val_loss']

plt.plot(vgghist.epoch, model_train_loss, label='Trainnig Loss')
plt.plot(vgghist.epoch, model_val_loss, label='Validation Loss')
plt.grid(True)
plt.legend()
```
<img width="396" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 17 05 16" src="https://user-images.githubusercontent.com/107698198/196661886-811de5b7-dff3-45fc-baf3-98c5ef7ab143.png">

จากนั้นนำผลการ Train มาสร้างเป็นกราฟ ช่วยให้เราเห็นเทรนด์ ของการ Train Model ซึ่งสามารถ สรุปผลได้ว่า เมื่อเราทำการ Train Model มากครั้งขึ้นเรื่อย ๆ จะทำให้ความคลาดเคลื่อนของ Model ลดลง หรืออีกนัยหนึ่งคือ Model มีความแม่นยำมากยิ่งขึ้น

```
train_acc = vgghist.history['acc']
val_acc = vgghist.history['val_acc']

plt.plot(vgghist.epoch, train_acc, label='Trainnig Accuracy')
plt.plot(vgghist.epoch, val_acc, label='Validation Accuracy')
plt.grid(True)
plt.legend()
```
<img width="389" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 17 08 44" src="https://user-images.githubusercontent.com/107698198/196662618-053deeb2-f75a-41a3-b6a2-e09329bfc4ae.png">

ในส่วนของค่าความแม่นยำนั้น เมื่อเรานำผลการ Train ทั้ง 50 ครั้ง มาสร้างเป็นกราฟ ช่วยให้เราเห็นเทรนด์ของความแม่นยำจากการ Train Model ในลักษณะที่ผกผันกับความคลาดเคลื่อนที่กล่าวไปก่อนหน้า
<br /><br />
ประเมินความแม่นยำของ Model
```
model.evaluate(test_generator)
```
<img width="697" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 17 11 10" src="https://user-images.githubusercontent.com/107698198/196663103-8fbd3c6e-5414-4d75-8142-48c42319d998.png">
ค่าความแม่นยำของ Model อยู่ที่ 68.54%
<br /><br />
ทดสอบแบบจำลองหรือ Test Model 

```
test_dup = test.copy()
test_dup = test_dup.reset_index(drop=True)
test_dup["predict"] = pd.DataFrame(predicted_label_batch)
test_dup.head()
```
<img width="530" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 17 13 47" src="https://user-images.githubusercontent.com/107698198/196663643-5a46ef66-31c0-40e5-975c-d466545b5365.png">

ทำการหารูปที่ทายผิดพลาดของจังหวัด โดยที่ สีเขียว หมายถึง ทายถูก และสีแดง หมายถึง ทายผิด

```
fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(20,20), subplot_kw={'xticks':[], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(test_dup.images[i]))
    color = "green" if test_dup.predict[i] == test_dup.labels[i] else "red"
    ax.set_title(test_dup.labels[i],color=color)
plt.show()
```
<img width="972" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 17 18 59" src="https://user-images.githubusercontent.com/107698198/196664672-bffe0a80-7085-4085-afde-e8a5cd1186b9.png">

### Base Model
- Optimizer the RMSprop algorithm. <br />
- Learning Rate = 0.0001 <br />
- Loss ใช้ categorical crossentropy <br />
- Output Dense Layer เป็น 77 ตามจำนวน class <br />
- Metric ใช้ accuracy
- Activation ใช้ softmax เพื่อให้ output แสดงผลออกมาเพียง class เดียว

```
from tensorflow.keras.applications.vgg16 import VGG16
base_model = VGG16(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')
```

```
y = layers.Flatten()(base_model.output)
y = layers.Dense(77, activation='softmax')(y)
base_model = tf.keras.models.Model(base_model.input, y)
base_model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss = 'categorical_crossentropy',metrics = ['acc'])
```
ทำการ Train model ทั้งหมด 50 ครั้ง
```
basehist = base_model.fit(training_generator, validation_data = validation_generator, epochs = 50)
```
<img width="985" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 17 24 05" src="https://user-images.githubusercontent.com/107698198/196665767-2483b880-07c1-42b4-af10-82512bb8a645.png">

ผลจากการ Train พบว่า ค่าความแม่นยำสูงที่สุดมีค่า 0.7280
```
base_model_train_loss = basehist.history['loss']
base_model_val_loss = basehist.history['val_loss']

plt.plot(basehist.epoch, base_model_train_loss, label='Trainnig Loss')
plt.plot(basehist.epoch, base_model_val_loss, label='Validation Loss')
plt.grid(True)
plt.legend()
```
<img width="387" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 17 28 10" src="https://user-images.githubusercontent.com/107698198/196666661-23cc2d61-888c-47c8-b0e5-df219a57a7bd.png">

จากนั้นนำผลการ Train มาสร้างเป็นกราฟ ช่วยให้เราเห็นเทรนด์ ของการ Train Model ซึ่งสามารถ สรุปผลได้ว่า เมื่อเราทำการ Train Model มากครั้งขึ้นเรื่อย ๆ จะทำให้ความคลาดเคลื่อนของ Model ลดลง หรืออีกนัยหนึ่งคือ Model มีความแม่นยำมากยิ่งขึ้น

```
base_model_train_acc =basehist.history['acc']
base_model_val_acc = basehist.history['val_acc']

plt.plot(basehist.epoch, base_model_train_acc, label='Trainnig Accuracy')
plt.plot(basehist.epoch, base_model_val_acc, label='Validation Accuracy')
plt.grid(True)
plt.legend()
```

<img width="385" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 17 29 13" src="https://user-images.githubusercontent.com/107698198/196666871-ba79e1e7-51eb-45a2-a750-84e033c89ca0.png">
ในส่วนของค่าความแม่นยำนั้น เมื่อเรานำผลการ Train ทั้ง 50 ครั้ง มาสร้างเป็นกราฟ ช่วยให้เราเห็นเทรนด์ของความแม่นยำจากการ Train Model ในลักษณะที่ผกผันกับความคลาดเคลื่อนที่กล่าวไปก่อนหน้า
<br /><br />
ประเมินความแม่นยำของ Model

```
base_model.evaluate(test_generator)
```
<img width="696" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 17 31 20" src="https://user-images.githubusercontent.com/107698198/196667342-d5806e34-91d9-4122-a740-aed25b5e5cef.png">


ค่าความแม่นยำของ Model อยู่ที่ 73.47%
<br /><br />
ทดสอบแบบจำลองหรือ Test Model 

```
test2_dup = test.copy()
test2_dup = test2_dup.reset_index(drop=True)
test2_dup["predict"] = pd.DataFrame(predicted2_label_batch)
test2_dup.head()
```
<img width="527" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 17 33 22" src="https://user-images.githubusercontent.com/107698198/196667760-c90e2408-4a5e-43eb-b473-e25b98d3f0ae.png">

ทำการหารูปที่ทายผิดพลาดของจังหวัด โดยที่ สีเขียว หมายถึง ทายถูก และสีแดง หมายถึง ทายผิด

```
fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(20,20), subplot_kw={'xticks':[], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(test2_dup.images[i]))
    color = "green" if test2_dup.predict[i] == test2_dup.labels[i] else "red"
    ax.set_title(test2_dup.labels[i],color=color)
plt.show()
```
<img width="983" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 17 34 10" src="https://user-images.githubusercontent.com/107698198/196667937-0be60610-327b-42d4-8ade-1a0a42b6cb67.png">

## MODEL(2) ResNet50
### Model Finetuning 
<img width="387" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 19 18 51" src="https://user-images.githubusercontent.com/107698198/196688829-a58bca8f-d78c-41e6-96e3-63d646454de7.png">
<img width="387" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 19 19 30" src="https://user-images.githubusercontent.com/107698198/196688958-4c5108b3-ea79-4b2c-b81b-15fd96079bb9.png">

### Base Model
<img width="410" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 19 20 42" src="https://user-images.githubusercontent.com/107698198/196689259-b6e358da-d712-4e4a-9ac4-a764c1f7d536.png">
<img width="395" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 19 20 54" src="https://user-images.githubusercontent.com/107698198/196689267-7a43a9ec-662e-4de3-b066-20bff2579ebe.png">


## MODEL(3) EfficientNet
### Model Finetuning 
<img width="382" alt="ภาพถ่ายหน้าจอ 2565-10-20 เวลา 17 56 35" src="https://user-images.githubusercontent.com/107698198/196930684-35727cb0-e636-43e8-a6ba-b4b8d59a77b7.png">
จากนั้นนำผลการ Train มาสร้างเป็นกราฟ ช่วยให้เราเห็นเทรนด์ ของการ Train Model ซึ่งสามารถ สรุปผลได้ว่า เมื่อเราทำการ Train Model มากครั้งขึ้นเรื่อย ๆ จะทำให้ความคลาดเคลื่อนของ Model ลดลง หรืออีกนัยหนึ่งคือ Model มีความแม่นยำมากยิ่งขึ้น

<img width="391" alt="ภาพถ่ายหน้าจอ 2565-10-20 เวลา 17 56 47" src="https://user-images.githubusercontent.com/107698198/196930696-ed40ec78-2eb2-4eb9-9f36-649b3b7525b3.png">
ในส่วนของค่าความแม่นยำนั้น เมื่อเรานำผลการ Train ทั้ง 50 ครั้ง มาสร้างเป็นกราฟ ช่วยให้เราเห็นเทรนด์ของความแม่นยำจากการ Train Model ในลักษณะที่ผกผันกับความคลาดเคลื่อนที่กล่าวไปก่อนหน้า

### Base Model
<img width="390" alt="ภาพถ่ายหน้าจอ 2565-10-20 เวลา 17 57 00" src="https://user-images.githubusercontent.com/107698198/196930703-314c8821-9a7a-462d-9937-6e690d789dbc.png">
จากนั้นนำผลการ Train มาสร้างเป็นกราฟ ช่วยให้เราเห็นเทรนด์ ของการ Train Model ซึ่งสามารถ สรุปผลได้ว่า เมื่อเราทำการ Train Model มากครั้งขึ้นเรื่อย ๆ จะทำให้ความคลาดเคลื่อนของ Model ลดลง หรืออีกนัยหนึ่งคือ Model มีความแม่นยำมากยิ่งขึ้น

<img width="413" alt="ภาพถ่ายหน้าจอ 2565-10-20 เวลา 17 57 09" src="https://user-images.githubusercontent.com/107698198/196930706-cf3f19a1-2a45-4ca9-bd23-adabbf9ecfba.png">
ในส่วนของค่าความแม่นยำนั้น เมื่อเรานำผลการ Train ทั้ง 50 ครั้ง มาสร้างเป็นกราฟ ช่วยให้เราเห็นเทรนด์ของความแม่นยำจากการ Train Model ในลักษณะที่ผกผันกับความคลาดเคลื่อนที่กล่าวไปก่อนหน้า <br /><br />





ค่าที่ได้จาก model อื่น ๆ เป็นดังตารางสรุปผลด้านล่าง

| Pre-Trained Models  | val_acc train finetuning  | evaluate test finetuning  | val_acc train base model | evaluate test base model |
| :---:  | :---:  | :---:  | :---:  | :---:  |
| VGG-16  | 0.6810  | 68.54%  | 0.7280  | 73.47%  |
| ResNet50 | 0.0470  | 3.05%  | 0.4207  | 38.26%  |
| EfficientNet  | 0.7671  | 90.14%  | 0.9256  | 89.20%  |


## CONCLUSION
จากตารางสรุปผลจะเห็นได้ว่าค่าความแม่นยำของ Base model มีค่ามากกว่า Model finetuning แต่จากกราฟที่แสดงค่าความคลาดเคลื่อนและค่าความแม่นยำ ของ VGG-16, ResNet50 และ EfficientNet จะได้เห็นว่ากราฟ Model Finetuning จะมีความ smooth มากกว่า Base model และ Pre-Trained Models ที่ให้ค่าสูงที่สุด คือ EfficientNet ที่ค่าความแม่นยำ 0.7671

## REFERANCE
[1]	Purva Huilgol — Published On August 18, 2020 and Last Modified On June 15th, 2022 from https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/ <br />
[2]	(Sep 5, 2021). ‘DSE4 G.6’ from https://medium.com/botnoi-classroom/ส่งคุกกี้เสี่ยงทาย-ให้-ai-ทำนายสายพันธุ์มะม่วงกัน-d379b7fc6d6e <br />
[3]	‘ImageClassification’ from https://github.com/nicknochnack/ImageClassification/blob/main/Getting%20Started.ipynb <br />
[4]	'Build a Deep CNN Image Classifier with ANY Images' from https://youtu.be/jztwpsIzEGc


## END CREDIT
งานชิ้นนี้เป็นส่วนหนึ่งของวิชา DADS7202 Deep Learning หลักสูตรวิทยาศาสตร์มหาบัณฑิต คณะสถิติประยุกต์ สถาบันบัณฑิตพัฒนบริหารศาสตร์

กลุ่มปลาทองออกลูกเป็นไข่
- 6410412001
- 6410412008
- 6410412017
