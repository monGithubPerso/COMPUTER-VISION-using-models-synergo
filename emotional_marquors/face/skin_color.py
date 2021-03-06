"""
the following script implements the "RGB-H-CbCr Skin Colour Model for Human Face Detection" 
algorithm for skin detection/extraction based on the paper published by
Nusirwan Anwar bin Abdul Rahman, Kit Chong Wei and John See 
Faculty of Information Technology, Multimedia University
johnsee@mmu.edu.my 
link : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.718.1964&rep=rep1&type=pdf
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
	


class Skin_Detect():
    def __init__(self):
    #Constractor that does nothing
            pass

    #RGB bounding rule
    def Rule_A(self,BGR_Frame,plot=False):
        '''this function implements the RGB bounding rule algorithm
        --inputs: 
        BGR_Frame: BGR components of an image
        plot: Bool type variable,if set to True draw the output of the algorithm
        --return a anumpy array of type bool like the following:
        [[False False False True]
        [False False False True]
        .
        .
        .
        [False False False True]]
        2d numpy array
        So in order to plot this matrix, we need to convert it to numbers like:
        255 for True values(white)
        0 for False(black)
        '''
        B_Frame, G_Frame, R_Frame =  [BGR_Frame[...,BGR] for BGR in range(3)]# [...] is the same as [:,:]
        #you can use the split built-in method in cv2 library to get the b,g,r components
        #B_Frame, G_Frame, R_Frame  = cv2.split(BGR_Frame)
        #i am using reduce built in method to get the maximum of a 3 given matrices
        BRG_Max = np.maximum.reduce([B_Frame, G_Frame, R_Frame])
        BRG_Min = np.minimum.reduce([B_Frame, G_Frame, R_Frame])
        #at uniform daylight, The skin colour illumination's rule is defined by the following equation :
        Rule_1 = np.logical_and.reduce([R_Frame > 95, G_Frame > 40, B_Frame > 20 ,
                                 BRG_Max - BRG_Min > 15,abs(R_Frame - G_Frame) > 15, 
                                 R_Frame > G_Frame, R_Frame > B_Frame])
        #the skin colour under flashlight or daylight lateral illumination rule is defined by the following equation :
        Rule_2 = np.logical_and.reduce([R_Frame > 220, G_Frame > 210, B_Frame > 170,
                         abs(R_Frame - G_Frame) <= 15, R_Frame > B_Frame, G_Frame > B_Frame])
        #Rule_1 U Rule_2
        RGB_Rule = np.logical_or(Rule_1, Rule_2)
        #return the RGB mask
        return RGB_Rule

    def lines(self,axis):
        '''return a list of lines for a give axis'''
        #equation(3)
        line1 = 1.5862  * axis + 20
        #equation(4)
        line2 = 0.3448  * axis + 76.2069
        #equation(5)
        #the slope of this equation is not correct Cr ??? -4.5652 ?? Cb + 234.5652
        #it should be around -1  
        line3 = -1.005 * axis + 234.5652
        #equation(6)
        line4 = -1.15   * axis + 301.75
        #equation(7)
        line5 = -2.2857 * axis + 432.85
        return [line1,line2,line3,line4,line5]
        #The five bounding rules of Cr-Cb

    def Rule_B(self,YCrCb_Frame,plot=False):
        '''this function implements the five bounding rules of Cr-Cb components
        --inputs: 
        YCrCb_Frame: YCrCb components of an image
        plot: Bool type variable,if set to True draw the output of the algorithm
        --return a anumpy array of type bool like the following:
        [[False False False True]
        [False False False True]
        .
        .
        .
        [False False False True]]
        2d numpy array
        So in order to plot this matrix, we need to convert it to numbers like:
        255 for True values(white)
        0 for False(black)
        '''
        Y_Frame,Cr_Frame, Cb_Frame = [YCrCb_Frame[...,YCrCb] for YCrCb in range(3)]
        line1,line2,line3,line4,line5 = self.lines(Cb_Frame)
        YCrCb_Rule = np.logical_and.reduce([line1 - Cr_Frame >= 0,
                                            line2 - Cr_Frame <= 0,
                                            line3 - Cr_Frame <= 0,
                                            line4 - Cr_Frame >= 0,
                                            line5 - Cr_Frame >= 0])
        
        return YCrCb_Rule



    def Rule_C(self, HSV_Frame):
        '''this function implements the five bounding rules of Cr-Cb components
        --inputs: 
        HSV_Frame: Hue,saturation and value components of a given image
        plot: Bool type variable,if set to True draw the output of the algorithm
        --return a anumpy array of type bool like the following:
        [[False False False True]
        [False False False True]
        .
        .
        .
        [False False False True]]
        2d numpy array
        So in order to plot this matrix, we need to convert it to numbers like:
        255 for True values(white)
        0 for False(black)
        '''
        Hue,Sat,Val = [HSV_Frame[...,i] for i in range(3)]
        #i changed the value of the paper 50 instead of 25 and 150 instead of 230 based on my plots
        HSV_ = np.logical_or(Hue < 50, Hue > 150)

        return HSV_

    def RGB_H_CbCr(self, Frame_, plot, percent):
        '''this function implements the RGB_H_CbCr bounding rule
        --inputs: 
        Frame_: BGR image
        plot: Bool type variable,if set to True draw the output of the algorithm
        --return a anumpy array of type integer like the following:
        [[0 0 1 0]
        [1 0 1 0]
        .
        .
        .
        [0 0 1 1]]
        2d numpy array
        '''
        Ycbcr_Frame = cv2.cvtColor(Frame_, cv2.COLOR_BGR2YCrCb)
        HSV_Frame = cv2.cvtColor(Frame_, cv2.COLOR_BGR2HSV)
        #Rule A ??? Rule B ??? Rule C
        skin_ = np.logical_and.reduce([self.Rule_A(Frame_),
                                       self.Rule_B(Ycbcr_Frame), self.Rule_C(HSV_Frame)])

        there_is_black_pixel = False
        if plot == True:
                skin_bw= skin_.astype(np.uint8) 
                skin_bw*=255
                #RGB original image
                RGB_Frame = cv2.cvtColor(Frame_, cv2.COLOR_BGR2RGB)
                seg = cv2.bitwise_and(Frame_,Frame_,mask=skin_bw)

                black_pixeks = np.sum(seg == np.array([0, 0, 0])) / 3
                shaping = seg.shape[0] * seg.shape[1]
                #print(black_pixeks)
                there_is_black_pixel = black_pixeks < (shaping * percent) // 100

                #print(there_is_black_pixel, black_pixeks, shaping)
                #cv2.imshow("aaa", seg)

        return there_is_black_pixel



def skin_color_main_function(image, percent_black_pixel_autorized):

    test = Skin_Detect()
    YCrCb_Frames = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    HSV_Frames = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    test.Rule_A(image)
    test.Rule_B(YCrCb_Frames)
    test.Rule_C(HSV_Frames)
    there_is_black_pixel = test.RGB_H_CbCr(image, True, percent_black_pixel_autorized)

    return there_is_black_pixel





#if __name__ == "__main__":

#	image = r"C:\Users\jeanbaptiste\Desktop\hardcoding\aa.png"

#	img = np.array(cv2.imread(image), dtype=np.uint8)

#	test = Skin_Detect()
#	YCrCb_Frames = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#	HSV_Frames = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#	test.Rule_A(img)
#	test.Rule_B(YCrCb_Frames)
#	test.Rule_C(HSV_Frames)
#	test.RGB_H_CbCr(img, True, 10)

#	cv2.waitKey(0)


