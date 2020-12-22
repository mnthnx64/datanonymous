"""
A new age steganography algorithm that
can hide textual data.

:Author: Manthan C S
:GitHub: mnthnx64
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
pi = math.pi
import random

# def polygon(sides, radius=1, rotation=0, translation=None):
#     one_segment = math.pi * 2 / sides

#     points = [
#         (math.sin(one_segment * i + rotation) * radius,
#          math.cos(one_segment * i + rotation) * radius)
#         for i in range(sides)]

#     if translation:
#         points = [[sum(pair) for pair in zip(point, translation)]
#                   for point in points]
    
#     return points



# string = sorted("Hi! This is a test string to show encryption.")
# set_of_chars = set(string)
# print(len(set_of_chars))
# sides_num = len(set_of_chars)
# pts = polygon(sides_num, radius=sides_num)
# x = [m[0] for m in pts]
# y = [m[1] for m in pts]
# x.append(0)
# y.append(0)
# fig, ax = plt.subplots()
# ax.scatter(x,y)

# for i, txt in enumerate(set_of_chars):
#     ax.annotate(txt, (x[i], y[i]+1))

# plt.show()

class Coder:
    input_json = dict()
    circle = list()
    colors = list()
    initial_rgb = 53

    def __init__(self, input_json):
        self.input_json = input_json

    def getFormat(self):
        for key in self.input_json:
            code_inp = list()
            if type(self.input_json[key]) == list:
                code_inp.append(key)
                code_inp.append((' ').join(self.input_json[key]))
            else:
                code_inp.append(key)
                code_inp.append(self.input_json[key])
            self.code(code_inp)
        self.showimg()
    
    
    def code(self,input_list):
        self.circle.append(self.getCircle(len(input_list[1])+len(input_list[0])+1))
        joint_data = input_list[1] + " " + input_list[0]
        for char in joint_data:
            ascii_value = ord(char)
            rgb_val = '#%02x%02x%02x' % (self.initial_rgb+ascii_value,random.randint(0,255),random.randint(0,255))
            self.colors.append(rgb_val)


        

    @staticmethod
    def getCircle(length):
        r = length
        n = r
        pts = [(math.cos(2*pi/n*x)*r, math.sin(2*pi/n*x)*r) for x in range(1,n+1)]
        x = [m[0] for m in pts]
        y = [m[1] for m in pts]

        return x,y

    def showimg(self):
        big_x = list()
        big_y = list()
        x = [m[0] for m in self.circle]
        y = [m[1] for m in self.circle]
        for exes in x:
            for pts in exes:
                big_x.append(pts)
        for exes in y:
            for pts in exes:
                big_y.append(pts)
        
        big_x.append(0)
        big_y.append(0)
        self.colors.append('#FF00FF')
        # print(len(big_x),len(big_y),len(self.colors))
        for X,Y,color in zip(big_x,big_y,self.colors):
            plt.scatter(X,Y,color=color)
        # fig,ax = plt.subplots()
        # ax.scatter(big_x,big_y,color=)
        plt.show()
        pass


if __name__ == '__main__':
    x = Coder({'Persia': 'asia', 'Armenia': 'Afghanistan'})
    x.getFormat()