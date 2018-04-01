import os
import shutil
IEMOCAP_path = "d:\\Temp\\IEMOCAP\\IEMOCAP_full_release";
sorted_data_path =  IEMOCAP_path + "\\Sorted data"

class_path = IEMOCAP_path + "\\Session5\\dialog\\EmoEvaluation"
sentence_path = IEMOCAP_path + "\\Session5\\sentences\\wav"
files = os.listdir(class_path)
print (files)

for file in files:
    print('processing: ', file)
    if file.__contains__('txt') and not file.startswith('.'):
        with open(class_path+"\\"+file, "r") as ins:
            array = []
            for line in ins:
                array.append(line)
            prev_line_is_new_line = False
            for line in array:
                if prev_line_is_new_line:
                    words = line.split('\t')
                    file_name = words[1]
                    file_val = words[2]
                    shutil.copy2(sentence_path+"\\"+file.rstrip(('.txt'))+"\\"+file_name+".wav",sorted_data_path+"\\"+file_val)
                if line == '\n':
                    prev_line_is_new_line = True
                else:
                    prev_line_is_new_line = False
