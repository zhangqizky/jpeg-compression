# --------------------------------------
# @authur  = "tangxi.zq"
# @time    = "2019-06-11"
# @file     = "utils.py"
# Description :jpeg编码的一些辅助函数.
# --------------------------------------
import numpy as np


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


def zigzag_points(rows, cols):
    """根据行和列找到对应的zigzag排序的点
    
    Arguments:
        rows {[type]} -- [description]
        cols {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    # 枚举方向常量
    UP, DOWN, RIGHT, LEFT, UP_RIGHT, DOWN_LEFT = range(6)

    # zigzag排序中有六个方向，上下左右右上以及左下
    def move(direction, point):
        return {
            UP: lambda point: (point[0] - 1, point[1]),
            DOWN: lambda point: (point[0] + 1, point[1]),
            LEFT: lambda point: (point[0], point[1] - 1),
            RIGHT: lambda point: (point[0], point[1] + 1),
            UP_RIGHT: lambda point: move(UP, move(RIGHT, point)),
            DOWN_LEFT: lambda point: move(DOWN, move(LEFT, point))
        }[direction](point)

    # 判断点是否在边界内
    def inbounds(point):
        return 0 <= point[0] < rows and 0 <= point[1] < cols

    # 从左上角开始
    point = (0, 0)

    # 当move_up为True的时候往右上角移动，false的时候往左下角移动。
    move_up = True

    for i in range(rows * cols):
        #产生rows*cols的块内的所有点的zigzag排序向量
        yield point
        if move_up:
            if inbounds(move(UP_RIGHT, point)):
                point = move(UP_RIGHT, point)
            else:
                move_up = False #往右上已经到顶，该往右了。
                if inbounds(move(RIGHT, point)):
                    point = move(RIGHT, point)
                else:
                    point = move(DOWN, point)
        else:
            if inbounds(move(DOWN_LEFT, point)):
                point = move(DOWN_LEFT, point)
            else:
                move_up = True
                if inbounds(move(DOWN, point)):
                    point = move(DOWN, point)
                else:
                    point = move(RIGHT, point)


def bits_required(n):
    """[判断一个数有多少位]
    
    Arguments:
        n {[被判断的数]} -- [必须是正数，非正数取绝对值]
    
    Returns:
        [int] -- [n的位数]
    """
    n = abs(n)
    result = 0
    while n > 0:
        n>>= 1
        result += 1
    return result


def binstr_flip(binstr):
    # 判断一个字符串是否是二进制字符串,是的话将其0和1的位置反转
    if not set(binstr).issubset('01'):
        raise ValueError("binstr should have only '0's and '1's")
    return ''.join(map(lambda c: '0' if c == '1' else '1', binstr))


def uint_to_binstr(number, size):
    
    return bin(number)[2:][-size:].zfill(size)


def int_to_binstr(n):
    if n == 0:
        return ''

    binstr = bin(abs(n))[2:]

    # 如果n是负的，将其二进制反转
    return binstr if n > 0 else binstr_flip(binstr)


def flatten(lst):
    '''
    拉直lst
    '''
    return [item for sublist in lst for item in sublist]