import sys
import xml.etree.ElementTree as ET
 
if len(sys.argv) < 4:
    raise ValueError('Please provide window size and input xml name.')

 
window_size = sys.argv[1]
window_slide = sys.argv[2]
input_file_name = sys.argv[3]
 

def read_from_xml_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    size = len(root)
    hr = []
    hr_72 = []
    sum_72 = 0
    count = 0
    for i in range(size):
        hr.append(int(root[i][1].text))
        
        sum_72 += int(root[i][1].text)
        count+=1
        if count == 72:
            hr_72.append(int(sum_72/72))
            count = 0
            sum_72 = 0
    return hr,hr_72

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



heart_rates = read_from_xml_file_x_sliding_y(input_file_name,window_size=int(window_size),window_slide=int(window_slide))


filename =  "./HR_Ground_Truth/HR_Ground_Truth.csv"
file = open(filename,"w")
file.close()


a_file = open(filename, "a")
a_file.write("Heart_Rates\n")
for hr in heart_rates:
    a_file.write(str(hr)+"\n")
a_file.close()

print("Heart Rate processed successfully!")