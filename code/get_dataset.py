# -*- coding: utf-8 -*-
import os
import gzip
import math
import numpy as np
import random

def dataset(file_path,kmer,seq_kernel,sturc_kernel):
    print('construct sequence feature...')
    seq_file = file_path + '/sequences.fa.gz'
    seq_list, label = read_seq_file(seq_file)
    seq_alphabet = ['A', 'C', 'G', 'T']
    seq_fea = []
    for seq in seq_list:
        seq_fea.append(get_kmer_fea(seq, kmer,seq_kernel, seq_alphabet)) # 64 * 109
    print('construct struct feature...')
    stru_file = file_path + '/structure.gz'
    stru_list = read_stru_file(stru_file)
    stru_alphabet = ['F', 'T', 'I', 'H', 'M', 'S']
    stru_fea = []
    for stru in stru_list:
        stru_fea.append(get_kmer_fea(stru, kmer,sturc_kernel, stru_alphabet)) # 216 * 109
        
    datasets = [[np.array(seq_fea),np.array(stru_fea)], np.array(label)]
    return datasets


def read_seq_file(seq_file):
    seq_list = []
    label_list = []
    neg_num=0
    pos_num=0
    seq = ''
    with gzip.open(seq_file, 'rt') as fp:
        for line in fp:
            if line[0] == '>':
                name = line.split(';')[0][2:]
                label = line[-2]
                if label=='0':
                    neg_num=neg_num+1
                else: pos_num=pos_num+1
                label_list.append(label)
                if len(seq):
                    seq_list.append(seq)              
                seq = ''
            else:
                seq = seq + str(line[:-1])
        if len(seq):
            seq_list.append(seq)
    print('neg:'+str(neg_num))
    print('pos:'+str(pos_num))
    return seq_list, label_list


def read_stru_file(stru_file):
    stru_list = []
    stru = ''
    with gzip.open(stru_file, 'rt') as fp:
        for line in fp:
            if line[0] != '>':
                stru = line[:-1]
                stru_list.append(stru)
    return stru_list


def get_kmer_fea(seq,k,j,alphabet):
    encoder = buildmapper(alphabet,k)
    seq_data = GetSeqDegree(seq,k,j)
    return embed(seq_data,encoder)


def buildmapper(alphabet,degree):
    length = degree
    #alphabet = ['A', 'C', 'G', 'T']
    #alphabet = ['F', 'T', 'I', 'H', 'M', 'S']
    mapper = ['']
    while length > 0:
        mapper_len = len(mapper)
        temp = mapper
        for base in range(len(temp)):
            for letter in alphabet:
                mapper.append(temp[base] + letter)
        while mapper_len > 0:
            mapper.pop(0)
            mapper_len -= 1

        length -= 1

    code = np.eye(len(mapper), dtype=int)
    encoder = {}
    for i in range(len(mapper)):
        encoder[mapper[i]] = list(code[i, :])

    number = int(math.pow(len(alphabet), degree))
    encoder['N'] = [1.0 / number] * number
    return encoder


def GetSeqDegree(seq, degree,motif_len):
    half_len = int(motif_len/2)  #half_len=2
    length = len(seq)  #101
    row = (length + motif_len - degree + 1)  #101+4-2+1=104
    seqdata = []
    for i in range (half_len):
        multinucleotide = 'N'
        seqdata.append(multinucleotide)

    for i in range(length - degree + 1):
        multinucleotide = seq[i:i + degree]
        seqdata.append(multinucleotide)

    for i in range (row-half_len,row):
        multinucleotide = 'N'
        seqdata.append(multinucleotide)

    return seqdata


def embed(seq, mapper):
    mat = []
    for element in seq:
        if element in mapper:
            mat.append(mapper.get(element))
        elif "N" in element:
            mat.append(mapper.get("N"))
        else:
            print ("wrong")
    return mat


def split_training_validation(classes, validation_size = 0.2):
    num_samples=len(classes)
    classes=np.array(classes)
    classes_unique=np.unique(classes)
    num_classes=len(classes_unique)
    indices=np.arange(num_samples)
    #indices_folds=np.zeros([num_samples],dtype=int)
    training_indice = []
    training_label = []
    validation_indice = []
    validation_label = []
    for cl in classes_unique:
        indices_cl=indices[classes==cl]
        num_samples_cl=len(indices_cl)

        num_samples_each_split=int(num_samples_cl*validation_size)
        res=num_samples_cl - num_samples_each_split
        
        training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
        training_label = training_label + [cl] * res
        
        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
        validation_label = validation_label + [cl]*num_samples_each_split

    training_index = np.arange(len(training_label))
    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]
    
    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]    
    
    return training_indice, training_label, validation_indice, validation_label