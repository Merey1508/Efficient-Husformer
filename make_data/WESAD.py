import pickle
import numpy as np
from sklearn.model_selection import KFold, train_test_split

def data_ready(pkl2):
    modality1 = pkl2[b'signal'][b'chest'][b'GSR']
    modality2 = pkl2[b'signal'][b'wrist'][b'BVP']
    modality3 = pkl2[b'signal'][b'chest'][b'EMG']
    modality4 = pkl2[b'signal'][b'chest'][b'ECG']
    modality5 = pkl2[b'signal'][b'chest'][b'Resp']
    modality6 = pkl2[b'signal'][b'wrist'][b'GSR']
    label = pkl2[b'label']
    subjects = np.array(pkl2[b'subject'])

    modality11, modality21, modality31, modality41, modality51, modality61, label1 = [], [], [], [], [], [], []
    subjects1 = []

    modality12, modality22, modality32, modality42, modality52, modality62, label2 = [], [], [], [], [], [], []
    subjects2 = []

    for i in range(len(label)):
        if label[i] == 0: label[i] = 0
        elif label[i] == 1: label[i] = 2
        elif label[i] == 2: label[i] = -1
        elif label[i] == 3: label[i] = 1
        elif label[i] == 4: label[i] = 0
        elif label[i] == 5: label[i] = 0
        elif label[i] == 6: label[i] = 0
        elif label[i] == 7: label[i] = 0

    # Iterate over unique subjects
    unique_subjects = np.unique(subjects)
    for subj in unique_subjects:
        subj_idx = np.where(subjects == subj)[0]
        mod1_subj = modality1[subj_idx]
        mod3_subj = modality3[subj_idx]
        mod4_subj = modality4[subj_idx]
        mod5_subj = modality5[subj_idx]
        mod6_subj = modality6[subj_idx]
        labels_subj = label[subj_idx]

        # Split into 700-sample chunks (or adjust based on your dataset structure)
        for j in range(0, mod1_subj.shape[0], 700):
            modality11.append(mod1_subj[j:j+700].reshape(50, 14))
            modality31.append(mod3_subj[j:j+700].reshape(50, 14))
            modality41.append(mod4_subj[j:j+700].reshape(50, 14))
            modality51.append(mod5_subj[j:j+700].reshape(50, 14))
            modality61.append(mod6_subj[j:j+700].reshape(50, 14))
            label1.append(labels_subj[j:j+700])
            subjects1.append(subj)

    # Remove any chunks with multiple labels (they have mixed labels)
    invalid_index = [k for k in range(len(label1)) if len(set(label1[k])) != 1]
    for x in reversed(invalid_index):
        for arr in [modality11, modality21, modality31, modality41, modality51, modality61, label1, subjects1]:
            arr.pop(x)

    # Create clean sets of data (no mixed labels)
    label_new = [[l[0]] for l in label1]
    nonzero_idx = [idx for idx, l in enumerate(label_new) if l != [0]]

    for x in nonzero_idx:
        modality12.append(modality11[x])
        modality22.append(modality21[x])
        modality32.append(modality31[x])
        modality42.append(modality41[x])
        modality52.append(modality51[x])
        modality62.append(modality61[x])
        label2.append(label_new[x])
        subjects2.append(subjects1[x])

    return modality12, modality22, modality32, modality42, modality52, modality62, label2, subjects2, len(modality12)


def pkl_make(modality1, modality2, modality3, modality4, modality5, modality6, label, train_idx, val_idx, test_idx, pkl1, fold_idx):
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


def WESAD_10fold_cv(modality1, modality2, modality3, modality4, modality5, modality6, label, subjects):
    unique_subjects = np.unique(subjects)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(unique_subjects)):
        train_val_subjects = unique_subjects[train_val_idx]
        test_subjects = unique_subjects[test_idx]

        train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.2, random_state=fold_idx)

        train_idx = [i for i, s in enumerate(subjects) if s in train_subjects]
        val_idx = [i for i, s in enumerate(subjects) if s in val_subjects]
        test_idx = [i for i, s in enumerate(subjects) if s in test_subjects]

        pkl1 = open(f'fold_{fold_idx}_3_classes_subject_independent.pkl', 'wb')
        pkl_make(modality1, modality2, modality3, modality4, modality5, modality6, label, train_idx, val_idx, test_idx, pkl1, fold_idx)

        print(f"Saved fold {fold_idx}: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")


if __name__ == '__main__':
    with open('/content/drive/MyDrive/Multimodal/Husformer/WESAD_list.txt', 'r') as f:
        pkl2 = pickle.load(f)

    modality1, modality2, modality3, modality4, modality5, modality6, label, subjects, _ = data_ready(pkl2)
    WESAD_10fold_cv(modality1, modality2, modality3, modality4, modality5, modality6, label, subjects)
