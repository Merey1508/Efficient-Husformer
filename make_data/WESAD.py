import numpy as np
import pickle
import csv

def data_ready(pkl2):
    modality1 = pkl2[b'signal'][b'chest'][b'GSR']
    modality2 = pkl2[b'signal'][b'wrist'][b'BVP']
    modality3 = pkl2[b'signal'][b'chest'][b'EMG']
    modality4 = pkl2[b'signal'][b'chest'][b'ECG']
    modality5 = pkl2[b'signal'][b'chest'][b'Resp']
    modality6 = pkl2[b'signal'][b'wrist'][b'GSR']

    label = pkl2[b'label']

    modality11 = []
    modality21 = []
    modality31 = []
    modality41 = []
    modality41 = []
    modality51 = []
    modality61 = []
    label1 = []

    modality12 = []
    modality22 = []
    modality32 = []
    modality42 = []
    modality52 = []
    modality62 = []
    label2 = []

    for i in range(len(label)):
        if label[i] == 0:
            label[i] =0
        elif label[i] == 1:
            label[i] =2
        elif label[i] == 2:
            label[i] =-1
        elif label[i] == 3:
            label[i] =1
        elif label[i] == 4:
            label[i] =0
        elif label[i] == 5:
            label[i] =0
        elif label[i] == 6:
            label[i] =0
        elif label[i] == 7:
            label[i] =0
    
    for j in range(0,modality1.shape[0],700):
        modality11.append(modality1[j:j+700].reshape(50,14))
        modality31.append(modality3[j:j+700].reshape(50,14))
        modality41.append(modality3[j:j+700].reshape(50,14))
        modality51.append(modality3[j:j+700].reshape(50,14))
        label1.append(label[j:j+700])

    for j in range(0,modality2.shape[0],64):
        modality21.append(modality2[j:j+64].reshape(16,4))

    for j in range(0,modality4.shape[0],4):
        modality61.append(modality4[j:j+4].reshape(1,4))

    invalid_index = []
    for k in range(len(label1)):
        b = list(set(label1[k]))
        if len(b) != 1:
            invalid_index.append(k)

    invalid_index.reverse()

    for x in invalid_index:
        modality11.pop(x)
        modality21.pop(x)
        modality31.pop(x)
        modality41.pop(x)
        label1.pop(x)
    label_new = []
    zeros = []

    for y in range(len(label1)):
        label_new.append([label1[y][0]])
        
    #print(label_new)

    for z in range(len(label_new)):
        if label_new[z] != [0]:
            zeros.append(z)

    for x in zeros:
        modality12.append(modality11[x])
        modality22.append(modality21[x])
        modality32.append(modality31[x])
        modality42.append(modality41[x])
        modality52.append(modality51[x])
        modality62.append(modality61[x])
        label2.append(label_new[x])
    index = len(modality12)
    return modality12,modality22,modality32,modality42, modality52, modality62, label2,index

def pkl_make(modality1,modality2,modality3,modality4, modality5, modality6, label,train_id,val_id,test_id,pkl,epoch):
    print('data over'+ str(epoch))
    modality1_train =  np.array(modality1)[train_id]
    modality1_val = np.array(modality1)[val_id]
    modality1_test = np.array(modality1)[test_id]

    modality2_train = np.array(modality2)[train_id]
    modality2_val = np.array(modality2)[val_id]
    modality2_test = np.array(modality2)[test_id]

    modality3_train = np.array(modality3)[train_id]
    modality3_val = np.array(modality3)[val_id]
    modality3_test = np.array(modality3)[test_id]

    modality4_train = np.array(modality4)[train_id]
    modality4_val = np.array(modality4)[val_id]
    modality4_test = np.array(modality4)[test_id]

    modality5_train = np.array(modality5)[train_id]
    modality5_val = np.array(modality5)[val_id]
    modality5_test = np.array(modality5)[test_id]

    modality6_train = np.array(modality6)[train_id]
    modality6_val = np.array(modality6)[val_id]
    modality6_test = np.array(modality6)[test_id]


    id_train = np.arange(train_id.shape[0]).reshape(train_id.shape[0],1,1)
    id_val = np.arange(val_id.shape[0]).reshape(val_id.shape[0],1,1)
    id_test = np.arange(test_id.shape[0]).reshape(test_id.shape[0],1,1)

    label_train = np.array(label)[train_id].reshape(train_id.shape[0],1,1)
    label_val = np.array(label)[val_id].reshape(val_id.shape[0],1,1)
    label_test = np.array(label)[test_id].reshape(test_id.shape[0],1,1)
    print('array over'+ str(epoch))
    pkl1 = {}
    train = {}
    test = {}
    valid ={}

    train['id'] = id_train
    train['modality1'] = modality1_train
    train['modality2'] = modality2_train
    train['modality3'] = modality3_train
    train['modality4'] = modality4_train
    train['modality5'] = modality5_train
    train['modality6'] = modality6_train
    train['label'] = label_train
    
    valid['id'] = id_val
    valid['modality1'] = modality1_val
    valid['modality2'] = modality2_val
    valid['modality3'] = modality3_val
    valid['modality4'] = modality4_val
    valid['modality5'] = modality5_val
    valid['modality6'] = modality6_val
    valid['label'] = label_val

    test['id'] = id_test
    test['modality1'] = modality1_test
    test['modality2'] = modality2_test
    test['modality3'] = modality3_test
    test['modality4'] = modality4_test
    test['modality5'] = modality5_test
    test['modality6'] = modality6_test
    test['label'] = label_test

    pkl1['train'] = train
    pkl1['valid'] = valid
    pkl1['test'] = test

    pickle.dump(pkl1,pkl)
    print('done'+ str(epoch))
    return

def WESAD (array,lenth,modality11,modality21,modality31,modality41, modality51,modality61,label1):
    for i in range(1):
        train1 = []
        val_start = int(i*lenth/10)
        val_end = test_start = int((i+1)*lenth/10)
        test_end = int((i+2)*lenth/10)
        final_test = int(0.1*lenth)
        if i < 9:
            val = array[val_start:val_end]
            test = array[test_start:test_end]
        else:
            val = array[val_start:val_end]
            test = array[:final_test]

        for k in array:
            if k not in np.append(val,test):
                train1.append(k)
        train = np.array(train1)
        pkl1 = open(str(i)+'_3_classes.pkl','wb')
        pkl_make(modality11,modality21,modality31,modality41,modality51,modality61,label1,train,val,test,pkl1,i)
    return 




def WESAD_merge(array, lenth, modality11, modality21, modality31, modality41, modality51, modality61, label1):
    lenth = len(array)
    
    # Define split indices
    train_end = int(0.8 * lenth)  # 80% for training
    val_end = int(0.9 * lenth)    # Next 10% for validation
    
    # Split dataset
    train = np.array(array[:train_end])
    val = np.array(array[train_end:val_end])
    test = np.array(array[val_end:])  # Last 10% for testing

    # Save the merged dataset
    pkl1 = open(str(i)+'merge_dataset.pkl','wb')
    pkl_make(modality11, modality21, modality31, modality41, modality51, modality61, label1, train, val, test, pkl1, 0)

    print(f"Dataset merged and split into Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}. Saved as 'merged_dataset.pkl'.")
    return




if __name__ == '__main__':
    txt1 = open('/content/drive/MyDrive/Multimodal/Husformer/WESAD_list.txt','r').readlines()
    modality11 = []
    modality21 = []
    modality31 = []
    modality41 = []
    modality51 = []
    modality61 = []
    label1 = []

    for i in txt1:
        k = i.rstrip('\n')
        print(k)
        pkl1 = open(k,'rb')
        pkl2 = pickle.load(pkl1,encoding = 'bytes')

        modality12,modality22,modality32,modality42, modality52,modality62,label2,index2 = data_ready(pkl2)
        modality11.extend(modality12)
        modality21.extend(modality22)
        modality31.extend(modality32)
        modality41.extend(modality42)
        modality51.extend(modality52)
        modality61.extend(modality62)
        label1.extend(label2)

    indices = np.arange(len(modality11))
    np.random.shuffle(indices)
    WESAD_merge(indices,indices.shape[0],modality11,modality21,modality31,modality41,modality51,modality61,label1)
