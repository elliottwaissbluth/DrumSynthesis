import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import argparse
import time
import json
import os

from model import OrchMatchNet
from process import show_all_class_num, show_all_instru_num, stat_test_db, decode
from dataset import OrchDataSet


epoch_num = 100
batch_size = 64
model_path = './model'
# db = stat_test_db()
db = {'BTb': 1349, 'TpC': 1305, 'Hn': 1370, 'Tbn': 1270, 'Va': 1357, 'Vn': 1358,
      'Vc': 1321, 'Cb': 1328, 'Ob': 1301, 'Fl': 1306, 'Bn': 1332, 'ClBb': 1403}
# db = {'BTb': 24, 'TpC': 17, 'Hn': 25, 'Tbn': 20, 'Va': 37, 'Vn': 27,
#       'Vc': 26, 'Cb': 33, 'Ob': 20, 'Fl': 24, 'Bn': 28, 'ClBb': 26}


def arg_parse():
    parser = argparse.ArgumentParser(description='combination of orch')
    parser.add_argument('--model', default='cnn')
    parser.add_argument('--is_resume', default='False',
                        choices=['True', 'False'])

    return parser.parse_args()


def main():
    print("Starting parsing -----")
    arg = arg_parse()
    print("End parsing")

    # dataset
    print("Start loading data -----")

    trainset = OrchDataSet('./data/trainset.pkl', transforms.ToTensor())
    testset = OrchDataSet('./data/testset.pkl', transforms.ToTensor())

    # load data
    train_load = torch.utils.data.DataLoader(dataset=trainset,
                                             batch_size=batch_size,
                                             shuffle=True)

    test_load = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=1,
                                            shuffle=False)
    print("End loading data")

    # model construction
    out_num, class_div = show_all_class_num()
    model = OrchMatchNet(out_num)
    start_epoch = 0
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # model load
    if arg.is_resume == 'True':
        ckpt = os.listdir(model_path)
        ckpts = []
        for x in ckpt:
            if x.endswith('.pth'):
                ckpts.append(int(x.split('.')[0].split('_')[-1]))

        if len(ckpt) != 0:
            model_resume_path = 'model_epoch_'+str(max(ckpts))+'.pth'
            state = torch.load(model_path+'/'+model_resume_path)
            model.load_state_dict(state['state_dict'])
            optimizer = optimizer.load_state_dict(state['optimizer'])
            start_epoch = state['epoch']
            print("Load %s successfully! " % model_resume_path)
        else:
            print("[Error] no checkpoint ")

    # train model
    if arg.model == 'cnn':
        train(model, optimizer, train_load, test_load, start_epoch)


def train(model, optimizer, train_load, test_load, start_epoch):
    print("Starting training")

    device = torch.device('gpu: 0' if torch.cuda.is_available() else 'cpu')

    out_num, class_div = show_all_class_num()

    # criterion = nn.CrossEntropyLoss().to(device)
    # criterion = nn.BCEWithLogitsLoss().to(device)
    sm = F.softmax
    criterion = F.binary_cross_entropy

    best_acc = 0
    best_epoch = None
    acc_sets = []
    stat_result = {}

    for epoch in range(start_epoch, epoch_num):
        model.train()
        for i, (trains, labels) in enumerate(train_load):
            trains = trains.to(device)
            labels = labels.to(device)

            outputs = model(trains)
            weights = np.ones(labels.shape, dtype=np.float32)
            weights = torch.tensor(weights)

            labels[labels == -1] = 0
            output = sm(outputs, dim=1)
            loss = criterion(output, labels/2,
                             weight=Variable(weights))*out_num

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print('Epoch:[{}/{}], Step:[{}/{}], Loss:{:.4f} '.format(
                    epoch + 1, epoch_num, i+1, len(train_load), loss))

        # save cuurent model
        model_name = model_path+'/model_epoch_'+str(epoch+1)+'.pth'
        state = {
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(),
        }

        torch.save(state, model_name)
        print('model_epoch_'+str(epoch+1)+' saved')

        model.eval()
        for key in class_div.keys():
            stat_result[key] = 0

        total_num = 0
        total_time = 0.0
        correct_num = 0
        correct_num_single = 0

        for tests, labels in test_load:
            tests = tests.to(device)
            labels = labels.to(device)

            start = time.time()
            outputs = model(tests)
            predicts = get_pred(outputs)
            end = time.time()

            total_num += labels.size(0)
            total_time += float(end-start)
            labels = decode(labels)
            correct_list, stat_result = count_correct(
                predicts, labels, class_div, stat_result)

            correct_num += correct_list[0]
            correct_num_single += correct_list[1]

        total_acc = 100*float(correct_num)/total_num
        single_acc = 100*float(correct_num_single)/total_num

        avg_time = total_time/float(len(test_load))
        print("Average Time: {:2.3f} ms".format(1000*avg_time))
        print("Total Accuracy: {:.3f}%, single Acc: {:.3f}% ".format(
            total_acc, single_acc))
        acc_result = {}
        for key in stat_result.keys():
            acc_result[key] = 100*float(stat_result[key])/float(db[key])
            print("{}: {}/{} = {:.4f} % ".format(key,
                                                 stat_result[key], db[key], acc_result[key]))

        acc_sets.append([epoch, total_acc, single_acc])

        if total_acc > best_acc:
            best_acc = total_acc
            best_epoch = epoch+1
            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
            }
            torch.save(state, "epoch_best.pth")

        f = open('acc.csv', 'w')
        for acc in acc_sets:
            for a in acc:
                if a != acc[len(acc)-1]:
                    f.write(str(a)+',')
                else:
                    f.write(str(a))
            f.write('\n')
        f.close()

        f = open('specific_acc.json', 'w')
        json.dump(acc_result, f)

    print("Best test accuracy: {} at epoch: {}".format(best_acc, best_epoch))


def count_correct(predicts, labels, class_div, stat_result):
    assert predicts.size(0) == labels.size(0)
    correct = 0
    correct_single = 0

    for one, two in zip(predicts, labels):
        if torch.equal(one, two) or one[0] == two[1] and one[1] == two[0]:
            correct += 1
            for key in class_div.keys():
                if one[0] >= class_div[key][0] and one[0] <= class_div[key][1]:
                    stat_result[key] += 1
                elif one[1] >= class_div[key][0] and one[1] <= class_div[key][1]:
                    stat_result[key] += 1

        elif one[0] == two[0] or one[0] == two[1]:
            correct_single += 1
            for key in class_div.keys():
                if one[0] >= class_div[key][0] and one[0] <= class_div[key][1]:
                    stat_result[key] += 1
                    break

        elif one[1] == two[1] or one[1] == two[0]:
            correct_single += 1
            for key in class_div.keys():
                if one[1] >= class_div[key][0] and one[1] <= class_div[key][1]:
                    stat_result[key] += 1
                    break

    return [correct, correct_single], stat_result


def get_pred(output):
    idx_1 = output.max(1, keepdim=True)[1]
    output[0][idx_1] = -1
    idx_2 = output.max(1, keepdim=True)[1]
    output[0][idx_1] = float(1)

    pred = torch.cat((idx_1, idx_2), dim=1)

    return pred


def test():
    device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

    testset = OrchDataSet('./data/testset.pkl', transforms.ToTensor())
    test_load = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=1,
                                            shuffle=False)

    model = OrchMatchNet(12)

    acc_sets = []
    model_paths = []
    _, class_div = show_all_class_num()
    stat_result = {}

    print("Start testing")
    for x in os.listdir('./model'):
        if x.endswith('.pth'):
            x = x.split('.')[0].split('_')[-1]
            model_paths.append(int(x))
            model_paths.sort()

    for model_path in model_paths:
        for key in class_div.keys():
            stat_result[key] = 0
        state = torch.load('./model/model_epoch_'+str(model_path)+'.pth')
        # state = torch.load('./epoch_best.pth')
        model.load_state_dict(state['state_dict'])

        model.eval()

        total_num = len(test_load)
        total_time = 0.0
        correct_num = 0
        correct_num_single = 0

        for tests, labels in test_load:
            tests = tests.to(device)
            labels = labels.to(device)

            start = time.time()
            outputs = model(tests)
            predicts = get_pred(outputs)
            end = time.time()

            total_time += float(end-start)
            labels = decode(labels)
            correct_list, stat_result = count_correct(
                predicts, labels, class_div, stat_result)

            correct_num += correct_list[0]
            correct_num_single += correct_list[1]

        total_acc = 100*float(correct_num)/total_num
        single_acc = 100*float(correct_num_single)/total_num

        avg_time = total_time/float(len(test_load))
        print("Average Time: {:2.3f} ms".format(1000*avg_time))
        print("Total Accuracy: {:.3f}%, single Acc: {:.3f}% ".format(
            total_acc, single_acc))
        acc_result = {}
        for key in stat_result.keys():
            acc_result[key] = 100*float(stat_result[key])/float(db[key])
            print("{}: {}/{} = {:.4f} % ".format(key,
                                                 stat_result[key], db[key], acc_result[key]))

        acc_sets.append([epoch, total_acc, single_acc])

        f = open('acc.csv', 'w')
        for acc in acc_sets:
            for a in acc:
                if a != acc[len(acc)-1]:
                    f.write(str(a)+',')
                else:
                    f.write(str(a))
            f.write('\n')
        f.close()

        f = open('specific_acc.json', 'w')
        json.dump(acc_result, f)


if __name__ == "__main__":
    main()
