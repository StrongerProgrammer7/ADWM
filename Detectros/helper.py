import os
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from PIL import Image


def create_annotation_xml(image_path, annotation_folder, output_folder):

    if not os.path.exists(image_path):
        return
    image = Image.open(image_path)
    width, height = image.size
    #print(width,height)

    annotation = Element("annotation")

    # Добавление элементов в XML
    folder = SubElement(annotation, "folder")
    folder.text = annotation_folder

    filename = SubElement(annotation, "filename")
    filename.text = os.path.basename(image_path)

    size = SubElement(annotation, "size")
    width_element = SubElement(size, "width")
    width_element.text = str(width)
    height_element = SubElement(size, "height")
    height_element.text = str(height)
    depth = SubElement(size, "depth")
    depth.text = "3"

    # Добавление информации об объекте (в данном случае, рука)
    object_element = SubElement(annotation, "object")
    name = SubElement(object_element, "name")
    name.text = "hand"

    bndbox = SubElement(object_element, "bndbox")
    xmin = SubElement(bndbox, "xmin")
    xmin.text = "10"  # Ваше значение xmin

    ymin = SubElement(bndbox, "ymin")
    ymin.text = "10"  # Ваше значение ymin

    xmax = SubElement(bndbox, "xmax")
    xmax.text = str(width)  # Используем ширину изображения

    ymax = SubElement(bndbox, "ymax")
    ymax.text = str(height)  # Используем высоту изображения

    # Преобразование XML в строку и сохранение в файл
    xml_string = tostring(annotation)
    dom = parseString(xml_string)
    xml_filepath = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + ".xml")

    with open(xml_filepath, "w") as xml_file:
        dom.writexml(xml_file)

def xmlCreate():
    # Hand_0000002
    # Hand_0011744
    for i in range(2, 11744):
        number = '000000' + str(i)
        if (i >= 10 and i < 100):
            number = f'00000{i}'
        if (i >= 100 and i < 1000):
            number = f'0000{i}'
        if (i >= 1000 and i < 10000):
            number = f'000{i}'
        if (i >= 10000):
            number = f'00{i}'
        image_path = f"./Hands/Hand_{number}.jpg"
        annotation_folder = "Hands"
        output_folder = "./annotation/"
        create_annotation_xml(image_path, annotation_folder, output_folder)

def createDatFile(images_folder):
    if not os.path.exists(images_folder):
        return

    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(images_folder, filename)
            image = Image.open(image_path)
            width, height = image.size
            dat_line = f"{image_path} 1 0 0 {width} {height}\n"

            # Открыть файл "dat" для дозаписи
            with open("Good.dat", "a") as dat_file:
                # Запись строки в файл "dat"
                dat_file.write(dat_line)

createDatFile("Hands")

