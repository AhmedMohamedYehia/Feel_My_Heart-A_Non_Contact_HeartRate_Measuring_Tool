
import xml.etree.ElementTree as ET

def read_from_xml_file_x_sliding_y(file_path,window_size =180, window_slide = 30):
    tree = ET.parse(file_path)
    root = tree.getroot()
    size = len(root)
    hr = []
    hr_180 = []
    sum_432 = 0
    count = 0
    for i in range(size):
        hr.append(int(root[i][1].text))
        
    for i in range(0,int(size-(window_size*2.4)),int(window_slide*2.4)):
        hr_180.append(int(sum(hr[i:int(i+(window_size*2.4))])/(window_size*2.4)))
    return hr_180