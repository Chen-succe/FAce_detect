import os
import cv2






# testing dataset
path='./data/widerface/val/images/'
testset_list = path[:-7] + "wider_val.txt"
print(testset_list)
with open(testset_list, 'r') as fr:
    test_dataset = fr.read().split()
num_images = len(test_dataset)

# print(num_images)


# testing begin
# stop=0
for i, img_name in enumerate(test_dataset):
    #
    # if stop==1:
    #     break
    # stop+=1
    path='./data/widerface/val/images'
    image_path = path + img_name
    # image_path = os.path.join(path,img_name)
    # print(img_name)
    # print(image_path)
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     path='/data/JCC/vampire/data/wider_val/'
    path='/home/kpl/ImageData/wider_valset/'
    if not os.path.exists(path):
        os.mkdir(path)
    # name=str(i).rjust(len(str(test_dataset))+1,'0') +'.jpg'
    # name=os.path.join(path,name)
    # cv2.imshow('',img_raw)
    # cv2.waitKey(0)


    name=path+str(i)+'.jpg'
    # print(name)

    cv2.imwrite(name,img_raw)

