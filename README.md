# DADS7202

## INTRODUCTION
ตราประจำจังหวัดของไทย มีพัฒนาการมาจากตราประจำตำแหน่งของเจ้าเมืองในสมัยสมบูรณาญาสิทธิราชย์ และตราประจำธงประจำกองลูกเสือ 14 มณฑล 
ในสมัยรัชกาลที่ 6-7 ในสมัยที่จอมพลแปลก พิบูลสงครามเป็นนายกรัฐมนตรีนั้น รัฐบาลได้กำหนดให้แต่ละจังหวัดมีตราประจำจังหวัดของตนเองใช้เมื่อ พ.ศ.2483 
โดยกรมศิลปากรเป็นผู้ออกแบบตราตามแนวคิดที่แต่ละจังหวัดกำหนดไว้ ในปัจจุบันเมื่อมีการตั้งจังหวัดขึ้นใหม่ ก็จะมีการออกแบบตราประจำจังหวัดด้วยเสมอ แต่ตราของบางจังหวัดที่ใช้อยู่นั้นบางตราก็ไม่ใช่ตราที่กรมศิลปากรเป็นผู้ออกแบบ บางจังหวัดก็เปลี่ยนไปใช้ตราประจำจังหวัดเป็นแบบอื่น บางที่ลักษณะของตราก็เพี้ยนไปจากลักษณะที่กรมศิลปากรออกแบบไว้ แต่ยังคงลักษณะหลัก ๆ ของตราเดิม

ประเทศไทยมีทั้งหมด 77 จังหวัด จึงได้เก็บภาพแยกเป็น Folder ละ 1 จังหวัด โดยเก็บรวบรวมภาพให้ได้มากที่สุดเท่าที่รวบรวมได้ ซึ่งมีทั้งภาพที่นำมาจากอินเทอร์เน็ต

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
ปรับแต่งชุดรูปภาพที่ใช้สำหรับ Train, Validation, Test ให้มีคุณลักษณะที่เหมือนกัน 
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

## Import Model and Finetuning
Model(1) VGG16 <br />
Optimizer that implements the RMSprop algorithm. <br />
Learning Rate = 0.0001 <br />
Computes the categorical crossentropy loss. <br />

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
Train model <br />
epochs = 50
```
vgghist = model.fit(training_generator, validation_data = validation_generator, epochs = 50)
```
<img width="989" alt="ภาพถ่ายหน้าจอ 2565-10-19 เวลา 16 59 19" src="https://user-images.githubusercontent.com/107698198/196660497-b737165b-f7f5-4ca1-bbb1-58059a35d39d.png">

ผลจากการ Train พบว่า ค่าความแม่นยำสูงที่สุดมีค่า 0.6810
