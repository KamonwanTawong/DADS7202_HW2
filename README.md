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

```
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15,10), subplot_kw={'xticks':[], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(dataframe.images[i]))
    ax.set_title(dataframe.labels[i])
plt.show()
```
