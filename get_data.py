import re
import csv

def extract_class_from_fasta(fasta_file):
    classes = []
    pos_num = 0
    neg_num = 0
    with open(fasta_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('>'):
                match = re.search(r'class:(\d+)', line)
                if match:
                    class_number = match.group(1)
                    if class_number=='0':
                        neg_num=neg_num+1
                    else: pos_num=pos_num+1
                    classes.append([class_number])
    print('pos:' + str(pos_num))
    print('neg:' + str(neg_num))
    return classes

def write_classes_to_csv(classes, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(classes)

fasta_file_path = './datasets/clip/2_PARCLIP_AGO2MNASE_hg19/30000/training_sample_0/sequences.fasta'
csv_output_file_path = 'E:/Desktop/Chenhuixian/本科毕设\datasets_labels/1_train_label.csv'

classes = extract_class_from_fasta(fasta_file_path)
#write_classes_to_csv(classes, csv_output_file_path)
