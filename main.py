import tensorflow as tf
import tqdm, math, sys, os
from collections import defaultdict
from functools import reduce

A, B, count, pi = defaultdict(dict), {}, {}, {}
state_list = ['B', 'M', 'E', 'S']
model_path = './model_pku_training'
train_file, test_file = './icwb2-data/training/pku_training.utf8', './icwb2-data/testing/pku_test.utf8'
result_output = 'model_output.txt'

"""
    A   转移矩阵
    B   混淆矩阵
    pi  计算i=0时的分布
    count   状态频率计算
    'B'：词开始
    'M'：词内部
    'E'：词结束
    'S'：单字词
"""


# 读入训练集
def load_dataset(file_path):
    wordlines, states = [], []
    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in tqdm.tqdm(fin.readlines()):
            words = line.strip().split('  ')
            ws = [w for w in ''.join(words)]
            # word = reduce(lambda a, b: a.extend(b), [ws for ws in w for w in words])
            state = []
            for w in words:
                if len(w) == 1:
                    state.append('S')
                else:
                    state.extend(['B'] + ['M'] * (len(w) - 2) + ['E'])
            wordlines.append(ws)
            states.append(state)

    return wordlines, states


def main(file_path, save_model_path=''):
    ws, ss = load_dataset(file_path)

    for x1 in state_list:
        for x2 in state_list:
            A[x1][x2] = 0

        count[x1] = 0
        B[x1] = {}
        pi[x1] = 0.0

    for index, s in enumerate(ss):
        if len(s) != len(ws[index]):
            print('error')
            continue
        for i in range(len(s)):
            count[s[i]] += 1
            if i == 0:
                pi[s[i]] += 1
            else:
                A[s[i - 1]][s[i]] += 1
            B[s[i]][ws[index][i]] = B[s[i]].get(ws[index][i], 0) + 1

    for state in state_list:
        pi[state] = math.log(pi[state] / len(ws)) if pi[state] else 0
        for x2 in A[state]:
            A[state][x2] = math.log(A[state][x2] / count[state]) if A[state][x2] else 0
        for w in B[state]:
            B[state][w] = math.log(B[state][w] / count[state])
    # save model
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    print(dict(A), file=open(os.path.join(save_model_path, 'A'), 'w'))
    print(dict(B), file=open(os.path.join(save_model_path, 'B'), 'w'))
    print(dict(pi), file=open(os.path.join(save_model_path, 'pi'), 'w'))

    return A, B, pi


def laod_model(save_model_path):
    A = eval(open(os.path.join(save_model_path, 'A'), 'r').read())
    B = eval(open(os.path.join(save_model_path, 'B'), 'r').read())
    pi = eval(open(os.path.join(save_model_path, 'pi'), 'r').read())

    return A, B, pi


def HMM_Viterbi(sentence, A, B, pi):
    V, path = [{}], {}
    for state in state_list:
        V[0][state] = pi[state] * B[state].get(sentence[0], 0)
        path[state] = [state]
    for i in range(1, len(sentence)):
        V.append({})
        newpath = {}
        for statenow in state_list:
            (prob, statelast) = max(
                [(V[i - 1][state1] * A[state1].get(statenow, 0) * B[statenow].get(sentence[i], 0), state1) for state1 in
                 state_list if V[i - 1][state1] > 0])
            V[i][statenow] = prob
            newpath[statenow] = path[statelast] + [statenow]
        path = newpath

    (prob, state) = max([(V[len(sentence) - 1][state], path[state]) for state in state_list])

    wd_list, buffer = [], ''
    for i, st in enumerate(state):
        if st == 'S':
            wd_list.append(sentence[i])
        elif st == 'B' or st == 'M':
            buffer += sentence[i]
        else:
            wd_list.append(buffer + sentence[i])
            buffer = ''

    return prob, wd_list


if __name__ == '__main__':
    if os.path.exists(model_path):
        A, B, pi = laod_model(model_path)
    else:
        A, B, pi = main(train_file, save_model_path='./model_pku_training')

    with open(test_file, 'r', encoding='utf-8') as fin, open(result_output, 'w', encoding='utf-8') as fout:
        for line in tqdm.tqdm(fin.readlines()):
            print(line)
            try:
                prob, wd_list = HMM_Viterbi(line.strip(), A, B, pi)
                result = '  '.join(wd_list)
            except:
                print('error')
                result = ''
            fout.write(result + '\n')
