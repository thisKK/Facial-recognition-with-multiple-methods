import os
import numpy as np

def extract(xml_folder: str, output_folder: str):
    for dir in os.listdir(xml_folder):
        print(dir)
        for fn in os.listdir(os.path.join(xml_folder, dir)):
            os.popen('cp ' + os.path.join(xml_folder, dir, fn) + ' /home/thiskk/Desktop/wider/result_xml/temp/' + fn)

def sortFile(text_folder: str , output_folder: str, ):
    for dir in os.listdir(text_folder):
        number = dir.split('_')
        number = int(number[0])
        for outdir in os.listdir(output_folder):
            outdirX = outdir.split('-')
            outdirX = int(outdirX[0])
            if outdirX != number:
                continue
            os.popen('cp ' + os.path.join(text_folder, dir) + ' ' + os.path.join(output_folder,outdir,dir))
            # print('cp ' + os.path.join(text_folder, dir) + ' ' + os.path.join(output_folder,outdir,dir))


if __name__ == '__main__':
    # XML_FOLDER = "/home/thiskk/Desktop/wider/result_xml/WIDER_prediction_yolo_confi0.5"
    # OUTPUT_FOLDER = "/home/thiskk/Desktop/wider/result_xml/temp/"
    path = '../widerface/wider/WIDER_val/images'

    INPUT = "/home/thiskk/Desktop/wider/result_text/ensumble/consensus"
    OUTPUT = "/home/thiskk/Desktop/wider/result_text/WIDER_Ensumbel/consensus"
    # extract(XML_FOLDER, OUTPUT_FOLDER)
    # for dir in os.listdir(path):
    #     folder_name = dir
    #     os.mkdir(os.path.join(OUTPUT, folder_name))
    #     print(folder_name)
    sortFile(INPUT, OUTPUT)

