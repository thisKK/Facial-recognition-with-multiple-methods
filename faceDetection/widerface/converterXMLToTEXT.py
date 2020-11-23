from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict


class XMLHandler:
    def __init__(self, xml_path: str or Path):
        self.xml_path = Path(xml_path)
        self.root = self.__open()

    def __open(self):
        with self.xml_path.open() as opened_xml_file:
            self.tree = ET.parse(opened_xml_file)
            return self.tree.getroot()

    def return_boxes_class_as_dict(self) -> Dict[int, Dict]:
        """
        Returns Dict with class name and bounding boxes.
        Key number is box number

        :return:
        """

        boxes_dict = {}
        for index, sg_box in enumerate(self.root.iter('object')):
            boxes_dict[index] = {"name": sg_box.find("name").text,
                                 "confidence": float(sg_box.find('confidence').text),
                                 "x": int(sg_box.find("bndbox").find("xmin").text),
                                 "y": int(sg_box.find("bndbox").find("ymin").text),
                                 "w": int(sg_box.find("bndbox").find("xmax").text) - int(sg_box.find("bndbox").find("xmin").text),
                                 "h": int(sg_box.find("bndbox").find("ymax").text) - int(sg_box.find("bndbox").find("ymin").text),
                                 }

        return boxes_dict


def converter(xml_files: str, output_folder: str) -> None:
    """
    Function converts pascal voc formatted files into ODM-File format

    :param xml_files: Path to folder with xml files
    :param output_folder: Path where results files should be written
    :return:
    """
    xml_files = sorted(list(Path(xml_files).rglob("*.xml")))
    for xml_index, xml in enumerate(xml_files, start=1):
        file_name = xml.stem
        filename = f"{file_name}.txt"
        filename_path = Path(output_folder) / filename
        xml_content = XMLHandler(xml)
        boxes = xml_content.return_boxes_class_as_dict()

        with open(filename_path, "a") as file:
            for box_index in boxes:
                box = boxes[box_index]
                box_content = f"{box['name']} {box['confidence']} {box['x']} {box['y']} {box['w']} {box['h']}\n"
                file.write(box_content)

    print(f"Converted {len(xml_files)} files!")


if __name__ == '__main__':
    XML_FOLDER = "/home/thiskk/Desktop/ResultXMLFormat/ensemble_unanimous_yolo_resnet50_DSFD_Confidence0.1"
    OUTPUT_FOLDER = "/home/thiskk/Desktop/ResultTextFormat/ensemble_unanimous_yolo_resnet50_DSFD_Confidence0.1"

    converter(XML_FOLDER, OUTPUT_FOLDER)