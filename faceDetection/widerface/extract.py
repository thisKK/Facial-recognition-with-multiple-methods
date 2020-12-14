import os
import numpy as np

def extract():
    count = 0
    xml_folder = "/home/thiskk/Desktop/WiderFace-Evaluation/wider/result_xml/WIDER_prediction_DSFD_confi0.5"
    output_folder = "/home/thiskk/Desktop/WiderFace-Evaluation/wider/result_xml/temp"
    for dir in os.listdir(xml_folder):
        print(dir)
        for fn in os.listdir(os.path.join(xml_folder, dir)):
            count += 1
            os.popen('cp ' + os.path.join(xml_folder, dir, fn) + ' ' +os.path.join(output_folder,fn))
    print('extract ', count, ' files')

def sortFile():
    count = 0
    text_folder = "/home/thiskk/Desktop/WiderFace-Evaluation/wider/result_text/ensumble/affirmative"
    output_folder = "/home/thiskk/Desktop/WiderFace-Evaluation/wider/result_text/WIDER_Ensumbel/affirmative"
    for dir in os.listdir(text_folder):
        number = dir.split('_')
        number = int(number[0])
        for outdir in os.listdir(output_folder):
            outdirX = outdir.split('-')
            outdirX = int(outdirX[0])
            if outdirX != number:
                continue
            os.popen('cp ' + os.path.join(text_folder, dir) + ' ' + os.path.join(output_folder,outdir,dir))
            count+=1
            # print('cp ' + os.path.join(text_folder, dir) + ' ' + os.path.join(output_folder,outdir,dir))
        print('sorted ', count, ' files')

def genarateDirWider():
    OUTPUT = "/home/thiskk/Desktop/WiderFace-Evaluation/wider/result_text/WIDER_Ensumbel/affirmative"
    path = '../widerface/wider/WIDER_val/images'
    for dir in os.listdir(path):
        folder_name = dir
        os.mkdir(os.path.join(OUTPUT, folder_name))
        print(folder_name)

if __name__ == '__main__':
     extract()
    # genarateDirWider()
    # sortFile()

