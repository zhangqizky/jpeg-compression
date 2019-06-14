### 1. 前言
&emsp;&emsp;最近在学习JPEG图像的压缩编码过程，想着可能有别的同学也对这个过程感兴趣，所以把自己的学习过程整理一下，顺便也再帮助自己理解这个过程。
### 2. 常见图像存储格式介绍 </br>
&emsp;&emsp;现代数字图像是由数码相机中的光电传感器将光信号转换为数字电信号，也就是数字信号获得的。如果将这种图像原始数字信号直接存储到文件中将会非常大，比如一个5000*5000的24位图，则其大小为5000*5000*3字节=71.5M（RGB三个通道，每个通道的像素值范围为0～255一个字节）其大小非常不适合存储和传输。所以，各种图像存储格式的出现是非常必要的。
<center>
![51397-20160327233326604-782826888.png](https://ata2-img.cn-hangzhou.oss-pub.aliyun-inc.com/65d12df6f76f13851b39f8f1b478b8fc.png)
图1. 图像文件存储与在内存中表示
</center>

&emsp;&emsp;我们都知道，JPEG/JPG是一种常见的图像存储格式，该格式是由Joint Photographic experts goup (联合图像专家组)开发的，并且是一种最为常见的图像存储格式。这种格式的优势是保证图像质量的同时最大可能的降低数据的冗余度，便于存储和传输。但是图像的存储也有其他的格式，并且不是所有的场合JPEG都是最佳的选择，其他常见的格式有：RAW(CMOS或CCD将光信号转换成的数字信号)，BMP(位图)，PNG，TIF，GIF等。但由于JPEG是目前市面上基本上全部数码设备的默认格式(注意，是默认格式，相机或手机都可以选择输出RAW原始图像数据，便于摄影发烧友后期修图)，所以对JPEG图像的特性研究非常有必要。本文主要就是对JPEG压缩过程进行详细的解说。

### 3. JPEG压缩过程详解  
JPEG静态图像压缩的基本算法分为四个步骤：
1. 按8*8像素将图像分块
2. 对每个小块进行DCT余弦变换
3. 量化
4. 编码
如果图片是彩色的，那么第一步之前需要先做色彩空间变换(RGB->YCrCb)，整个编码算法的流程图如下所示。
<center>
![13_47_46__06_11_2019.jpg](https://ata2-img.cn-hangzhou.oss-pub.aliyun-inc.com/a307d421f7129a9335abfa206b340244.jpg)
图2. JPEG编码流程图</center>
#### 3.1 分块及预处理
首先利用PIL的Image模块将未经压缩的位图读进内存：
```
    image = Image.open("lena.bmp")
    ycbcr = image.convert('YCbCr')
    npmat = np.array(ycbcr, dtype=np.uint8)
    npmat = npmat -128
```
然后进行DCT变换的预处理，因为dct的常用分块是8*8，如果图像宽和高不能被8整除，则进行补零操作。
```
   # dct分块：8*8
    if rows % 8 == cols % 8 == 0:
        blocks_count = rows // 8 * cols // 8
    else:
        #如果图像分辨率不能被8整除，将其补零成能被整除的
        dct_rows = rows + 8 - rows % 8
        dct_cols = cols + 8 - cols % 8
        diff_cols = dct_cols - cols
        diff_rows = dct_rows - rows
        npmat = np.pad(npmat,((0,diff_rows), (0,diff_cols),(0,0)), mode = 'mean')
        blocks_count = (dct_rows // 8)*(dct_cols // 8)
        # raise ValueError(("the width and height of the image should both be mutiples of 8"))
```
#### 3.2 对每个8*8分块进行DCT变换
&emsp;&emsp;变换的基本思想是找到一组新的基，让图像在这组基下能量分布更为集中，便于分离能量。适合图像的变换有很多种，比如KL离散变换，傅立叶变换(DFT)，离散余弦变换(DCT)。其中KL变换的基与输入数据相关不固定，反变换时还需要原始数据，这就很尴尬了。而DFT变换的系数会比DCT变换多，且重构误差比DCT高，因此，这里的变换选择DCT离散余弦变换，并且在实际的实验中，证明了离散余弦变换的分块为8×8的时候性能与重构误差达到了一个最好的trade-off。DCT变换过程是可逆的，主要目的是为了找到图像中比较稀疏的高频分量，在下一步的量化过程将其舍弃，达到信息压缩的目的。JPEG采用的是2-D DCT变换作为其核心，该变换的定义是：
<center>
![14_09_50__06_11_2019.jpg](https://ata2-img.cn-hangzhou.oss-pub.aliyun-inc.com/999feff79d48ade33e4a70106095c803.jpg)![14_10_41__06_11_2019.jpg](https://ata2-img.cn-hangzhou.oss-pub.aliyun-inc.com/2955eec54df6c76cd4cc436d98b36c8b.jpg)
</center>

代码的实现就简单啦,python的scipy包中有一个库叫做fftpack，了解一下。
```
def dct_2d(image):
    return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')
    
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            try:
                block_index += 1
            except NameError:
                block_index = 0

            for k in range(3):
                # 将像素值归一化在-127到128之间
                block = npmat[i:i+8, j:j+8, k] - 128

                dct_matrix = dct_2d(block)#分块进行dct变换
                quant_matrix = quantize(dct_matrix,'lum' if k == 0 else 'chrom')
                zz = block_to_zigzag(quant_matrix)

                dc[block_index, k] = zz[0]#直流系数为每一块的第一个
                ac[block_index, :, k] = zz[1:]#剩下的63个都是交流系数
```

#### 3.3 量化 
整个JPEG压缩编码过程中，只有这一步是不可逆的，不可逆的意味着能量的损失。64个DCT系数会用 8 × 8 的量化表进行均匀量化，量化表中的每个元素是 1 到 255 之间的整 数，表示对应的 DCT 系数的量化步长。量化的作用在于降低 DCT 系数的精度， 从而达到更好的压缩率。量化是多对一映射，因此是有损的，它是基于变换的编 码器中导致信息损失的主要步骤，也是用户惟一能参与控制压缩质量的步骤。量化的过程是将每个DCT系数除以对应的量化步长，并四舍五入为整数：
<center>
![14_23_43__06_11_2019.jpg](https://ata2-img.cn-hangzhou.oss-pub.aliyun-inc.com/28fd9e9422611fd64f1de90b7d8070e2.jpg)
</center>
量化表和量化因子是一一对应的，因子越高，量化表中的量化步长越小。并且理论上应该根据输入图像确定，但是JPEG标准中并没有规定或推荐使用哪一个量化表，通常亮度和色度分量个有一份量化表。亮度指的是Y通道，色度指的是Cr和Cb通道。一个量化表的例子如下代码中所示。
当然，这个量化的过程用python实现起来也很简单:

```
def load_quantization_table(component):
    # Quantization Table for: Photoshop - (Save For Web 080)
    # (http://www.impulseadventure.com/photo/jpeg-quantization.html)
    #亮度通道的量化表
    if component == 'lum':
        q = np.array([[2, 2, 2, 2, 3, 4, 5, 6],
                      [2, 2, 2, 2, 3, 4, 5, 6],
                      [2, 2, 2, 2, 4, 5, 7, 9],
                      [2, 2, 2, 4, 5, 7, 9, 12],
                      [3, 3, 4, 5, 8, 10, 12, 12],
                      [4, 4, 5, 7, 10, 12, 12, 12],
                      [5, 5, 7, 9, 12, 12, 12, 12],
                      [6, 6, 9, 12, 12, 12, 12, 12]])
    #色度通道的量化表
    elif component == 'chrom':
        q = np.array([[3, 3, 5, 9, 13, 15, 15, 15],
                      [3, 4, 6, 11, 14, 12, 12, 12],
                      [5, 6, 9, 14, 12, 12, 12, 12],
                      [9, 11, 14, 12, 12, 12, 12, 12],
                      [13, 14, 12, 12, 12, 12, 12, 12],
                      [15, 12, 12, 12, 12, 12, 12, 12],
                      [15, 12, 12, 12, 12, 12, 12, 12],
                      [15, 12, 12, 12, 12, 12, 12, 12]])
    else:
        raise ValueError((
            "component should be either 'lum' or 'chrom', "
            "but '{comp}' was found").format(comp=component))

    return q

def quantize(block, component):
    q = load_quantization_table(component)
    return (block / q).round().astype(np.int32)

```
#### 3.4 行程编码和熵编码   
&emsp;&emsp;JPEG 压缩的最后一步是对量化后的系数进行熵编码。这一步采用通用的无损数据压缩技术，对图像质量没有影响。在熵编码前，对63个交流系数先采用ZigZag排序，转变为一维向量。这样做的目的是为了将低频系数放在前面，高频系数放在后面，因为高频系数中有很多 0，为了 节约空间，所以交流系数的“中间符号”用零行程码 (Zero Run Length) 表示。
<center>
![14_22_00__06_11_2019.jpg](https://ata2-img.cn-hangzhou.oss-pub.aliyun-inc.com/d26dddb8edce620de81e5afdd2c6a245.jpg)
图3. DCT变换的直流系数和交流系数的编码
</center>
此段的代码为：
```
 #Huffman编码，其中交流系数还需要游程编码
    H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
    H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))
    H_AC_Y = HuffmanTree(flatten(run_length_encode(ac[i, :, 0])[0] for i in range(blocks_count)))
    H_AC_C = HuffmanTree(flatten(run_length_encode(ac[i, :, j])[0] for i in range(blocks_count) for j in [1, 2]))
```
然后再对直流系数和行程编码之后的交流系数进行huffman编码。Huffman编码是一种变长编码，符号出现的频率越高，码字越短。其实现的细节不是本文的重点，可以参考这里[Huffman编码详细解释](https://blog.csdn.net/FX677588/article/details/70767446)

#### 3.5 解码
解码的过程与编码完全相反，除去量化步骤不可逆，其余步骤都是可逆的。解码的流程图如下：
<center>
![image.png](https://ata2-img.cn-hangzhou.oss-pub.aliyun-inc.com/23d65dba367c62407df047820ea3438c.png)
图4. JPEG解码流程图
</center>


### 4. 总结
本文详细介绍了JPEG图像压缩过程，并且附上代码分析。如果有兴趣，完整代码请参考。


