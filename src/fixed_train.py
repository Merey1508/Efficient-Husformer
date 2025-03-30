import torch
from torch import nn
import sys
from src import fixed_models
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *
#from focal_loss.focal_loss import FocalLoss
import time
import wandb
wandb.login()

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    # Initialize a new W&B run
    wandb.init(    # set the wandb project where this run will be logged
        project="Husformer",
        # track hyperparameters and run metadata
    )
    wandb.config.update(hyp_params)
    model = getattr(fixed_models, hyp_params.model+'Model')(hyp_params)
    if hyp_params.use_cuda:
        model = model.cuda()
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    weights = torch.FloatTensor([0.4,0.3,0.3]).to('cuda:0')
    criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler,}
    s = time.time()
    results = train_model(settings, hyp_params, train_loader, valid_loader, test_loader)
    print(f'Train time: {round(time.time()-s, 2)} sec.')
    wandb.finish()
    return results

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    
    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        mae_train2 = 0
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
            sample_ind, m1,m2,m3,m4, m5, m6 = batch_X
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            model.zero_grad()
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    m1,m2,m3,m4, m5, m6,eval_attr = m1.cuda(),m2.cuda(),m3.cuda(),m4.cuda(),m5.cuda(),m6.cuda(),eval_attr.cuda()
            batch_size = m1.size(0)
            batch_chunk = hyp_params.batch_chunk

            combined_loss = 0
            net = nn.DataParallel(model) if batch_size > 10 else model
            if batch_chunk > 1:
                raw_loss = combined_loss = 0
                m1_chunks = m1.chunk(batch_chunk, dim=0)
                m2_chunks = m2.chunk(batch_chunk, dim=0)
                m3_chunks = m3.chunk(batch_chunk, dim=0)
                m4_chunks = m4.chunk(batch_chunk, dim=0)
                m5_chunks = m5.chunk(batch_chunk, dim=0)
                m6_chunks = m6.chunk(batch_chunk, dim=0)
                eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)
                
                for i in range(batch_chunk):
                    m1_i, m2_i, m3_i, m4_i, m5_i, m6_i = m1_chunks[i],m2_chunks[i],m3_chunks[i],m4_chunks[i], m5_chunks[i], m6_chunks[i]
                    eval_attr_i = eval_attr_chunks[i]
                    #print(f"Train batch: m_1.shape={m1_i.shape}")

                    preds_i, hiddens_i = net(m1_i, m2_i, m3_i, m4_i, m5_i , m6_i)
                    
                    raw_loss_i = criterion(preds_i, eval_attr_i) / batch_chunk
                    raw_loss += raw_loss_i
                    raw_loss_i.backward()
                combined_loss = raw_loss 
            else:
                #print(f"Train batch: m_1.shape={m1.shape}")
                preds, hiddens = net(m1,m2,m3,m4,m5,m6)
                eval_attr = torch.where(eval_attr == -1, torch.tensor(0), eval_attr)  # -1 → 0
                eval_attr = torch.where(eval_attr == 1, torch.tensor(1), eval_attr)   # 1 → 1
                eval_attr = torch.where(eval_attr == 2, torch.tensor(2), eval_attr)   # 2 → 2
                eval_attr = eval_attr.squeeze().long()  # Ensure correct shape and integer type
                # Convert to (batch_size,) and integer type
                raw_loss = criterion(preds, eval_attr)
                combined_loss = raw_loss 

                combined_loss.backward()
                preds_class = torch.argmax(preds, dim=1)  # Get predicted class (1024,)
                mae_train1 = mae1(preds_class, eval_attr)  # Ensure same shape

            mae_train2 += mae_train1
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            # if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
            #     avg_loss = proc_loss / proc_size
            #     elapsed_time = time.time() - start_time
            #     memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            #     #print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f} | memory_used {:5.4f} MB'.
            #           format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss,memory_used))
            #     proc_loss, proc_size = 0, 0
            #     start_time = time.time()
                
        mae_train = mae_train2 / num_batches
        print('train_mae:',mae_train)
        wandb.log({"train_loss": epoch_loss / len(train_loader), "train_mae": mae_train})

        return epoch_loss / hyp_params.n_train, mae_train

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                sample_ind, m1, m2, m3, m4, m5, m6 = batch_X
                eval_attr = batch_Y.squeeze(dim=-1)  # if num of labels is 1

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        m1, m2, m3, m4, m5, m6, eval_attr = m1.cuda(), m2.cuda(), m3.cuda(), m4.cuda(), m5.cuda(), m6.cuda(), eval_attr.cuda()
                
                batch_size = m1.size(0)
                net = nn.DataParallel(model) if batch_size > 10 else model
                preds, _ = net(m1, m2, m3, m4, m5, m6)

                eval_attr = torch.where(eval_attr == -1, torch.tensor(0), eval_attr)  # -1 → 0
                eval_attr = torch.where(eval_attr == 1, torch.tensor(1), eval_attr)   # 1 → 1
                eval_attr = torch.where(eval_attr == 2, torch.tensor(2), eval_attr)   # 2 → 2
                eval_attr = eval_attr.squeeze().long()  # Ensure correct shape and integer type

                total_loss += criterion(preds, eval_attr) * batch_size

                # Collect the results
                preds_class = torch.argmax(preds, dim=1)
                results.append(preds_class)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)


        results1 = torch.cat(results)
        truths2 = torch.cat(truths)

        # Compute MAE
        mae = mae1(results1, truths2)

        results3 = torch.cat(results).cpu().numpy()
        truths4 = torch.cat(truths).cpu().numpy()
        # Compute confusion matrix for class-wise metrics
        cm = confusion_matrix(truths4, results3, labels=[0, 1, 2])

        # Compute class-wise accuracy
        class_accuracies = []
        for i in range(3):  # Three classes: 0, 1, 2
            TP = cm[i, i]
            s = sum(cm[i, :])
            FN = s - TP  # FN = actual positives - correctly predicted ones
            
            acc_i = TP / (TP + FN) if (TP + FN) > 0 else 0  # Avoid division by zero
            class_accuracies.append(acc_i)
            print(f'Class {i} : sum = {s}')

        print(f'acc_1 = {round(class_accuracies[0], 2)}, acc_2 = {round(class_accuracies[1], 2)}, acc_3 = {round(class_accuracies[2], 2)}')
        wandb.log({"acc_1": round(class_accuracies[0], 2), "acc_2": round(class_accuracies[1], 2), "acc_3": round(class_accuracies[2], 2)})
        acc = sum(class_accuracies) / 3  # Average over three classes
        f1 = f1_score(truths4, results3, average='macro')  # Macro-averaged F1-score

        test_log = f"Evaluation {'Test' if test else 'Validation'} - Loss: {avg_loss:.4f}, MAE: {mae:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}"
        print(test_log)
        
        return avg_loss, results1, truths2, mae, acc, test_log, f1

    mae_train1 = []
    mae_valid1 = []
    mae_test1 = []
    acc_val1 = []
    acc_test1 = []
    best_valid = 1e8
    test_logs = []
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        _,mae_train = train(model, optimizer, criterion)
        val_loss, _, _,mae_valid, val_acc, _, val_f1 = evaluate(model,criterion, test=False)
        test_loss, _, _ ,mae_test, test_acc, test_log, test_f1 = evaluate(model,criterion, test=True)
        mae_train1.append(mae_train)
        mae_valid1.append(mae_valid)
        mae_test1.append(mae_test)

        acc_val1.append(val_acc)
        acc_test1.append(test_acc)
        test_logs.append(test_log)

        wandb.log({"val_acc": val_acc, "test_acc": test_acc})
        wandb.log({"val_mae": mae_valid, "test_mae": mae_test})
        wandb.log({"val_loss": val_loss, "test_loss": test_loss})
        wandb.log({"test_f1": test_f1, "val_f1": val_f1})



        end = time.time()
        duration = end-start
        scheduler.step(val_loss)    # Decay learning rate by validation loss
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | MAE-valid{:5.4f} | Test Loss {:5.4f} | MAE-test{:5.4f} | memory_used{:5.4f} MB'.format(epoch, duration, val_loss, mae_valid,test_loss,mae_test,memory_used))
        print("-"*50)
        n_parameters = sum(p.numel() for p in model.parameters())
        print('n_parameters:',n_parameters)
        # if val_loss < best_valid:
        #     print(f"Saved model at output/{hyp_params.name}.pt!")
        #     save_model(hyp_params, model, name=hyp_params.name)
        #     best_valid = val_loss

    n_parameters = sum(p.numel() for p in model.parameters())

    max_acc_test = max(acc_test1)
    ind = acc_test1.index(max_acc_test)
    print('-'*50)
    print(f'Cross-modal: layers={hyp_params.c_layers}, heads={hyp_params.c_heads}')
    print(f'Self Attentions: layers={hyp_params.s_layers}, heads={hyp_params.s_heads}')
    print(f'd_m = {hyp_params.d_m}')
    print(test_logs[ind])
    print(f'Memory used: {round(memory_used, 2)}')
    print(f'Number of params: {n_parameters}')



    #sys.stdout.flush()
    #input('[Press Any Key to start another run]')