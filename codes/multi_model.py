# -*- coding: utf-8 -*-
from keras.layers import Input,Conv1D,MaxPooling1D,Bidirectional,Flatten
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.merge import concatenate
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras import backend as K
from attention import attention_3d_block2,attention_3d_block
#K.set_image_dim_ordering('th')
K.image_data_format() == 'channels_first'


def merged_BLSTM_CNN_1mer(seq_shape,struc_shape,seq_kernel, struc_kernel):
    # input
    nbfilter = 16
    seq_input =  Input(shape = (seq_shape,4))
    structure_input = Input(shape = (struc_shape,6))
    
    #generate seq structure model
    seq_model = Conv1D(filters=nbfilter,kernel_size=seq_kernel)(seq_input)
    seq_model_act = Activation("relu")(seq_model)
    seq_model_pool = MaxPooling1D(pool_size=3)(seq_model_act)
    seq_model_pool_D = Dropout(0.3)(seq_model_pool)
    seq_model_LSTM = Bidirectional(LSTM(128, return_sequences=True))(seq_model_pool_D)
    seq_model_LSTM_D = Dropout(0.3)(seq_model_LSTM)
    seq_attention = attention_3d_block2(seq_model_LSTM_D)
    seq_attention = Flatten()(seq_attention)
    structure_model = Conv1D(filters=nbfilter, kernel_size=struc_kernel)(structure_input)
    structure_model_act = Activation("relu")(structure_model)
    structure_model_pool = MaxPooling1D(pool_size=3)(structure_model_act)
    structure_model_pool_D = Dropout(0.3)(structure_model_pool)
    structure_model_LSTM = Bidirectional(LSTM(128, return_sequences=True))(structure_model_pool_D)
    structure_model_LSTM_D = Dropout(0.3)(structure_model_LSTM)
    structure_attention = attention_3d_block2(structure_model_LSTM_D)
    structure_attention = Flatten()(structure_attention)


    #merge
    merged = concatenate([seq_attention,structure_attention],axis = -1)
    merged_Des = Dense(136)(merged)
    merged_act = Activation("relu")(merged_Des)
    merged_Dro = Dropout(0.4)(merged_act)
    
    merged_out = Dense(2)(merged_Dro)
    output = Activation("softmax")(merged_out)
    
    model = Model(inputs=[seq_input,structure_input],outputs= output)
    
    model.summary()
    
    return model


def merged_BLSTM_CNN_2mer(seq_shape,struc_shape,seq_kernel, struc_kernel):
    nbfilter = 16 #卷积层滤波器数量
    #generate seq structure model
    seq_input =  Input(shape = (seq_shape,16))
    seq_model = Conv1D(filters=nbfilter,kernel_size=seq_kernel)(seq_input)
    seq_model_act = Activation("relu")(seq_model)
    seq_model_pool = MaxPooling1D(pool_size=3)(seq_model_act)
    seq_model_pool_D = Dropout(0.3)(seq_model_pool)
    seq_model_LSTM = Bidirectional(LSTM(68, return_sequences=True))(seq_model_pool_D)
    seq_model_LSTM_D = Dropout(0.3)(seq_model_LSTM)
    seq_attention = attention_3d_block2(seq_model_LSTM_D)
    seq_attention = Flatten()(seq_attention)

    structure_input = Input(shape = (struc_shape,36))
    structure_model = Conv1D(filters=16,kernel_size=struc_kernel)(structure_input)
    structure_model_act = Activation("relu")(structure_model)
    structure_model_pool = MaxPooling1D(pool_size=3)(structure_model_act)
    structure_model_pool_D = Dropout(0.3)(structure_model_pool)
    structure_model_LSTM = Bidirectional(LSTM(68, return_sequences=True))(structure_model_pool_D)
    structure_model_LSTM_D = Dropout(0.3)(structure_model_LSTM)
    structure_attention = attention_3d_block2(structure_model_LSTM_D)
    structure_attention = Flatten()(structure_attention)

    #merge
    merged = concatenate([seq_attention,structure_attention],axis = -1)
    merged_Des = Dense(136)(merged)
    merged_act = Activation("relu")(merged_Des)
    merged_Dro = Dropout(0.4)(merged_act)
    
    merged_out = Dense(2)(merged_Dro)
    output = Activation("softmax")(merged_out)
    
    model = Model(inputs=[seq_input,structure_input],outputs= output)
    
    model.summary()
    
    return model


def merged_BLSTM_CNN_3mer(seq_shape,struc_shape,seq_kernel, struc_kernel):
    nbfilter = 16
    #generate seq structure model
    seq_input =  Input(shape = (seq_shape,64))
    seq_model = Conv1D(filters=nbfilter,kernel_size=seq_kernel)(seq_input)
    seq_model_act = Activation("relu")(seq_model)
    seq_model_pool = MaxPooling1D(pool_size=3)(seq_model_act)
    seq_model_pool_D = Dropout(0.3)(seq_model_pool)
    seq_model_LSTM = Bidirectional(LSTM(68))(seq_model_pool_D)
    seq_model_LSTM_D = Dropout(0.3)(seq_model_LSTM)
    
    structure_input = Input(shape = (struc_shape,216))
    structure_model = Conv1D(filters=nbfilter,kernel_size=struc_kernel)(structure_input)
    structure_model_act = Activation("relu")(structure_model)
    structure_model_pool = MaxPooling1D(pool_size=3)(structure_model_act)
    structure_model_pool_D = Dropout(0.3)(structure_model_pool)
    structure_model_LSTM = Bidirectional(LSTM(68))(structure_model_pool_D)
    structure_model_LSTM_D = Dropout(0.3)(structure_model_LSTM)
    
    #merge
    merged = concatenate([seq_model_LSTM_D,structure_model_LSTM_D],axis = -1)
    merged_Des = Dense(136)(merged)
    merged_act = Activation("relu")(merged_Des)
    merged_Dro = Dropout(0.4)(merged_act)
    
    merged_out = Dense(2)(merged_Dro)
    output = Activation("softmax")(merged_out)
    
    model = Model(inputs=[seq_input,structure_input],outputs= output)
    
    model.summary()
    
    return model


def only_BLSTM_CNN_2mer(seq_shape, struc_shape, seq_kernel, struc_kernel):
    nbfilter = 16  # 卷积层滤波器数量
    # generate seq structure model
    seq_input = Input(shape=(seq_shape, 16))
    seq_model = Conv1D(filters=nbfilter, kernel_size=seq_kernel)(seq_input)
    seq_model_act = Activation("relu")(seq_model)
    seq_model_pool = MaxPooling1D(pool_size=3)(seq_model_act)
    seq_model_pool_D = Dropout(0.3)(seq_model_pool)
    seq_model_LSTM = Bidirectional(LSTM(68))(seq_model_pool_D)
    seq_model_LSTM_D = Dropout(0.3)(seq_model_LSTM)


    structure_input = Input(shape=(struc_shape, 36))
    structure_model = Conv1D(filters=16, kernel_size=struc_kernel)(structure_input)
    structure_model_act = Activation("relu")(structure_model)
    structure_model_pool = MaxPooling1D(pool_size=3)(structure_model_act)
    structure_model_pool_D = Dropout(0.3)(structure_model_pool)
    structure_model_LSTM = Bidirectional(LSTM(68))(structure_model_pool_D)
    structure_model_LSTM_D = Dropout(0.3)(structure_model_LSTM)

    # merge
    merged = concatenate([seq_model_LSTM_D, structure_model_LSTM_D], axis=-1)
    merged_Des = Dense(136)(merged)
    merged_act = Activation("relu")(merged_Des)
    merged_Dro = Dropout(0.4)(merged_act)

    merged_out = Dense(2)(merged_Dro)
    output = Activation("softmax")(merged_out)

    model = Model(inputs=[seq_input, structure_input], outputs=output)

    model.summary()

    return model


def onlyseq_model(seq_shape, seq_kernel):
    nbfilter = 16  # 卷积层滤波器数量
    # generate seq structure model
    seq_input = Input(shape=(seq_shape, 16))
    seq_model = Conv1D(filters=nbfilter, kernel_size=seq_kernel)(seq_input)
    seq_model_act = Activation("relu")(seq_model)
    seq_model_pool = MaxPooling1D(pool_size=3)(seq_model_act)
    seq_model_pool_D = Dropout(0.3)(seq_model_pool)
    seq_model_LSTM = Bidirectional(LSTM(68, return_sequences=True))(seq_model_pool_D)
    seq_model_LSTM_D = Dropout(0.3)(seq_model_LSTM)
    seq_attention = attention_3d_block2(seq_model_LSTM_D)
    seq_attention = Flatten()(seq_attention)
    # merge
    #merged = concatenate([seq_attention, structure_attention], axis=-1)
    merged_Des = Dense(136)(seq_attention)
    merged_act = Activation("relu")(merged_Des)
    merged_Dro = Dropout(0.4)(merged_act)

    merged_out = Dense(2)(merged_Dro)
    output = Activation("softmax")(merged_out)

    model = Model(inputs=seq_input, outputs=output)

    model.summary()

    return model