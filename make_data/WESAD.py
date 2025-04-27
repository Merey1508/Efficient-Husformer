import pickle
import numpy as np
from sklearn.model_selection import train_test_split

def data_ready(pkl2):
    # Load signals
    modality1 = pkl2[b'signal'][b'chest'][b'GSR']
    modality2 = pkl2[b'signal'][b'wrist'][b'BVP']
    modality3 = pkl2[b'signal'][b'chest'][b'EMG']
    modality4 = pkl2[b'signal'][b'chest'][b'ECG']
    modality5 = pkl2[b'signal'][b'chest'][b'Resp']
    modality6 = pkl2[b'signal'][b'wrist'][b'GSR']
    label = pkl2[b'label']
    subjects = np.array(pkl2[b'subject'])  # <-- Subject IDs

    # Initialize arrays
    modality11, modality21, modality31, modality41, modality51, modality61, label1 = [], [], [], [], [], [], []
    subjects1 = []

    modality12, modality22, modality32, modality42, modality52, modality62, label2 = [], [], [], [], [], [], []
    subjects2 = []

    # Adjust labels
    for i in range(len(label)):
        if label[i] == 0: label[i] = 0
        elif label[i] == 1: label[i] = 2
        elif label[i] == 2: label[i] = -1
        elif label[i] == 3: label[i] = 1
        elif label[i] == 4: label[i] = 0
        elif label[i] == 5: label[i] = 0
        elif label[i] == 6: label[i] = 0
        elif label[i] == 7: label[i] = 0

    # Chunk signals
    for j in range(0, modality1.shape[0], 700):
        modality11.append(modality1[j:j+700].reshape(50,14))
        modality31.append(modality3[j:j+700].reshape(50,14))
        modality41.append(modality4[j:j+700].reshape(50,14))
        modality51.append(modality5[j:j+700].reshape(50,14))
        label1.append(label[j:j+700])
        subjects1.append(subjects[j])

    for j in range(0, modality2.shape[0], 64):
        modality21.append(modality2[j:j+64].reshape(16,4))

    for j in range(0, modality6.shape[0], 4):
        modality61.append(modality6[j:j+4].reshape(1,4))

    # Remove inconsistent labels
    invalid_index = []
    for k in range(len(label1)):
        if len(set(label1[k])) != 1:
            invalid_index.append(k)
    invalid_index.reverse()
    for x in invalid_index:
        modality11.pop(x)
        modality21.pop(x)
        modality31.pop(x)
        modality41.pop(x)
        modality51.pop(x)
        modality61.pop(x)
        label1.pop(x)
        subjects1.pop(x)

    # Keep only non-zero labels
    label_new = [[l[0]] for l in label1]
    zeros = [idx for idx, l in enumerate(label_new) if l != [0]]

    for x in zeros:
        modality12.append(modality11[x])
        modality22.append(modality21[x])
        modality32.append(modality31[x])
        modality42.append(modality41[x])
        modality52.append(modality51[x])
        modality62.append(modality61[x])
        label2.append(label_new[x])
        subjects2.append(subjects1[x])

    index = len(modality12)

    return modality12, modality22, modality32, modality42, modality52, modality62, label2, subjects2, index

def pkl_make(modality1, modality2, modality3, modality4, modality5, modality6, label, train_idx, val_idx, test_idx, pkl1, fold_idx):
    # Create dictionary to save
    data = {
        'train': {
            'modality1': [modality1[i] for i in train_idx],
            'modality2': [modality2[i] for i in train_idx],
            'modality3': [modality3[i] for i in train_idx],
            'modality4': [modality4[i] for i in train_idx],
            'modality5': [modality5[i] for i in train_idx],
            'modality6': [modality6[i] for i in train_idx],
            'label': [label[i] for i in train_idx]
        },
        'val': {
            'modality1': [modality1[i] for i in val_idx],
            'modality2': [modality2[i] for i in val_idx],
            'modality3': [modality3[i] for i in val_idx],
            'modality4': [modality4[i] for i in val_idx],
            'modality5': [modality5[i] for i in val_idx],
            'modality6': [modality6[i] for i in val_idx],
            'label': [label[i] for i in val_idx]
        },
        'test': {
            'modality1': [modality1[i] for i in test_idx],
            'modality2': [modality2[i] for i in test_idx],
            'modality3': [modality3[i] for i in test_idx],
            'modality4': [modality4[i] for i in test_idx],
            'modality5': [modality5[i] for i in test_idx],
            'modality6': [modality6[i] for i in test_idx],
            'label': [label[i] for i in test_idx]
        },
        'fold': fold_idx
    }
    pickle.dump(data, pkl1)
    pkl1.close()

def WESAD(array_len, modality1, modality2, modality3, modality4, modality5, modality6, label, subjects):
    unique_subjects = np.unique(subjects)
    
    for i in range(10):  # 10 random folds
        # Split subjects
        train_subjects, temp_subjects = train_test_split(unique_subjects, test_size=0.3, random_state=i)
        val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.66, random_state=i)
        
        # Collect sample indices based on subject split
        train_idx = [idx for idx, subj in enumerate(subjects) if subj in train_subjects]
        val_idx = [idx for idx, subj in enumerate(subjects) if subj in val_subjects]
        test_idx = [idx for idx, subj in enumerate(subjects) if subj in test_subjects]

        # Save each fold
        pkl1 = open(f'{i}_3_classes_subject_independent.pkl','wb')
        pkl_make(modality1, modality2, modality3, modality4, modality5, modality6, label, 
                 np.array(train_idx), np.array(val_idx), np.array(test_idx), pkl1, i)
    return

def WESAD_merge(array, modality11, modality21, modality31, modality41, modality51, modality61, label1, subjects):
    unique_subjects = np.unique(subjects)
    
    # Split subjects
    train_subjects, temp_subjects = train_test_split(unique_subjects, test_size=0.3, random_state=42)
    val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.66, random_state=42)
    
    train_idx = [idx for idx, subj in enumerate(subjects) if subj in train_subjects]
    val_idx = [idx for idx, subj in enumerate(subjects) if subj in val_subjects]
    test_idx = [idx for idx, subj in enumerate(subjects) if subj in test_subjects]

    pkl1 = open('merged_dataset.pkl','wb')
    pkl_make(modality11, modality21, modality31, modality41, modality51, modality61, label1, train_idx, val_idx, test_idx, pkl1, 0)

    print(f"Dataset merged and split into Train: {len(train_idx)}, Validation: {len(val_idx)}, Test: {len(test_idx)}. Saved as 'merged_dataset.pkl'.")
    return


if __name__ == '__main__':
    with open('path_to_your_WESAD_data.pkl', 'rb') as f:
        pkl2 = pickle.load(f)

    modality1, modality2, modality3, modality4, modality5, modality6, label, subjects, array_len = data_ready(pkl2)
    WESAD_merge(array_len, modality1, modality2, modality3, modality4, modality5, modality6, label, subjects)
