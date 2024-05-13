from EDeN.eden.converter.fasta import sequence_to_eden
from EDeN.eden.modifier.rna.annotate_rna_structure import annotate_single
import subprocess as sp

def run_rnashape(sequence):
    #
    cmd = 'echo "%s" | RNAshapes -t %d -c %d -# %d' % (sequence,5, 10, 1)
    out = sp.check_output(cmd, shell=True)
    text = out.strip().decode(encoding="utf-8").split('\r\n')

    if 'configured to print' in text[-1]:
        struct_text = text[-2]
    else:
        struct_text = text[1]

    structur = struct_text.split()[1]
    graph = sequence_to_eden([("ID", sequence)]).__next__()
    graph.graph['structure']=structur

    annotate_single(graph)
    encode_struct = ''.join([ x["entity_short"].upper() for x in graph.node.values() ])
    return encode_struct
    #pdb.set_trace()

def read_structure(seq_file):
    fw = open(seq_file + '.structure', 'w')
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line
                if len(seq):
                    fw.write(name)
                    seq = seq.replace('U', 'T')
                    struc_en = run_rnashape(seq)
                    fw.write(struc_en + '\n')
                old_name = name              
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            fw.write(old_name)
            seq = seq.replace('U', 'T')
            struc_en = run_rnashape(seq)
            fw.write(struc_en + '\n') 
    fw.close()


if __name__ == "__main__":  
#    run_predict_graphprot_data()  
    sequence = 'TGGAAACATTCCTCAGGTGGTTCATCCAAGGCCCTTTCCACTCTTTCAGCTCACAGCACAGTGGTCCTTTTGTTCTTTGGTCCACCCATGTTTGTGTATAC'
    encode_struct = run_rnashape(sequence)
    print (encode_struct)
