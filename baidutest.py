from aip import AipImageClassify
from PIL import Image
import matplotlib.pyplot as plt

# 在百度云创建实例应用，获取的三个参数
app_ID='23862064'
app_KEY='VQ7hEVplyKIrtF9BBmhMO4Ov'
select_KEY='2QCvTvhdFVTO1PNjlitFWGnqIumTPPAN'
client=AipImageClassify(app_ID,app_KEY,select_KEY)


# 打印图片文件并读取二进制图片信息
def get_file_content(file_path):
    with open(file_path,'rb') as f:
        return f.read()

image=get_file_content('1.jpg')

# 调用client对象的cardetect方法
# {“top_num”：1} 表示返回车型中第一个
# print(client.carDetect(image,options={"top_num":1})["result"][0]['name'])
print(client.carDetect(image))
# 复习了PIL结合matplotlib显示图片

image1=Image.open('1.jpg')
a=plt.figure()
plt.imshow(image1)
plt.show()
# 第一次出错是我的三个参数出错，没有复制过去，是自己敲上的，IL有歧义。
