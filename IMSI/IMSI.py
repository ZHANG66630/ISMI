# -*- coding: utf-8 -*-

import os
import itertools
import gzip
import numpy as np
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from embedding.models import IMSI
from kgetools import KgDataset
import sys
sys.path.append("..")
from kgetools.metrics.ranking import precision_at_k, average_precision
from kgetools.metrics.classification import auc_pr, auc_roc


def main():
    seed = 1234
    nb_epochs_then_check = None
    data_name = "pse"
    kg_dp_path = "./data/SL_KG"

    print("Importing dataset files ... ")
    benchmark_train_fd = open(os.path.join(kg_dp_path, "train.txt"), "rt")
    benchmark_valid_fd = open(os.path.join(kg_dp_path, "valid.txt"), "rt")
    benchmark_test_fd = open(os.path.join(kg_dp_path, "test.txt"), "rt")

    benchmark_train = np.array([l.strip().split() for l in benchmark_train_fd.readlines()])
    benchmark_valid = np.array([l.strip().split() for l in benchmark_valid_fd.readlines()])
    benchmark_test = np.array([l.strip().split() for l in benchmark_test_fd.readlines()])

    #benchmark_triples的意思是把训练集，测试集，验证集三个表聚合在一起
    benchmark_triples = np.array([[d1, se, d2] for d1, se, d2 in
                                  np.concatenate([benchmark_train, benchmark_valid, benchmark_test])])

    #pse_drugs把三元组中的药物取出来
    pse_drugs = list(set(list(np.concatenate([benchmark_triples[:, 0], benchmark_triples[:, 2]]))))
    #pse_list把副作用取出来
    pse_list = set(list(benchmark_triples[:, 1]))

    #so_sort这个函数的意思是要求三元组（s,p,o）中s必须大于o，把满足的存储到tri = []
    def so_sort(benchmark_triples):
        tri = []
        for s, p, o in benchmark_triples:
            if s > o:
                tri.append((s, p, o))
            else:
                tri.append((o, p, s))
        return np.array(tri)

    benchmark_train = so_sort(benchmark_train)
    benchmark_valid = so_sort(benchmark_valid)
    benchmark_test = so_sort(benchmark_test)

    rel_dict = dict()#rel_dict这个字典的作用是存储副作用关系
    for s, p, o in benchmark_triples:
        if p not in rel_dict:
            rel_dict[p] = 1
        else:
            rel_dict[p] += 1

    pair_dict = dict()#pair_dict字典的作用是存储药物对，并且按照前一个药物的序号大于后一个
    for s, p, o in benchmark_triples:
        if s > o:
            pair = (s, o)
        else:
            pair = (o, s)
        if pair not in pair_dict:
            pair_dict[pair] = 1
        else:
            pair_dict[pair] += 1
    #drug_combinations数组是把三个数据集中的药物进行随机组合成不同的药物对
    drug_combinations = np.array([[d1, d2] for d1, d2 in list(itertools.product(pse_drugs, pse_drugs)) if d1 != d2])

    print("Processing dataset files to generate a knowledge graph ... ")
    # delete raw polypharmacy data，删除原始多药数据
    del benchmark_triples

    #dataset中使用字典存储了实体和关系，字典中的value数值表示着一个唯一的实体或者关系。此外存储了data数据，即把药物三元组用数字转换
    dataset = KgDataset(name=data_name)
    dataset.load_triples(benchmark_train, tag="bench_train")
    dataset.load_triples(benchmark_valid, tag="bench_valid")
    dataset.load_triples(benchmark_test, tag="bench_test")

    del benchmark_train
    del benchmark_valid
    del benchmark_test

    nb_entities = dataset.get_ents_count()#实体的个数
    nb_relations = dataset.get_rels_count()

    pse_indices = dataset.get_rel_indices(list(pse_list)) #对于每一个副作用关系对应一个索引
    # drug_combinations数组是把三个数据集中的药物进行随机组合成不同的药物对
    # 把drug_combinations中每一列的药物对取出来，然后找出这个药物对应的索引值
    d1 = np.array(dataset.get_ent_indices(list(drug_combinations[:, 0]))).reshape([-1, 1])#把drug_combinations中第一列的药物对取出来，然后找出这个药物对应的索引值
    d2 = np.array(dataset.get_ent_indices(list(drug_combinations[:, 1]))).reshape([-1, 1])
    drug_combinations = np.concatenate([d1, d2], axis=1) #这一步就是相等于再把d1，d2重新按照药物对的形式组合起来
    del d1
    del d2


    # grouping side effect information by the side effect type
    #以下的三个数据集就相当于把数据集中的字符串形式转换为了数字形式的三元组
    train_data = dataset.data["bench_train"]
    valid_data = dataset.data["bench_valid"]
    test_data = dataset.data["bench_test"]


    bench_idx_data = np.concatenate([train_data, valid_data, test_data])#聚合所有的数据
    se_facts_full_dict = {se: set() for se in pse_indices}
    #se_facts_full_dict是把每个关系对应的三元组使用集合聚合在一起。
    for s, p, o in bench_idx_data:
        se_facts_full_dict[p].add((s, p, o))

    print("Initializing the knowledge graph embedding model... ")
    # model pipeline definition
    model = IMSI(seed=seed, verbose=2)
    pipe_model = Pipeline([('kge_model', model)])

    # set model parameters
    model_params = {
        'kge_model__em_size': 200,
        'kge_model__lr': 0.01,
        'kge_model__optimiser': "AMSgrad",
        'kge_model__log_interval': 10,
        'kge_model__nb_epochs': 100,
        'kge_model__batch_size': 7000,
        'kge_model__initialiser': 'xavier_uniform',
        'kge_model__nb_ents': nb_entities,
        'kge_model__nb_rels': nb_relations
    }

    # add parameters to the model then call fit method
    pipe_model.set_params(**model_params)


    print("Training ... ")
    pipe_model.fit(X=train_data, y=None)

    metrics_per_se = {se_idx: {"ap": .0, "auc-roc": .0, "auc-pr": .0, "Ap@50": .0} for se_idx in pse_indices}#pse_indices=134 代表每个副作用的索引

    se_ap_list = []
    se_auc_roc_list = []
    se_auc_pr_list = []
    se_p50_list = []

    print("================================================================================")
    for se in tqdm(pse_indices, desc="Evaluating test data for each side-effect"):  #tqdm表示进度条
        print(se)
        # print(se_facts_full_dict)
        se_name = dataset.get_rel_labels([se])[0]
        se_all_facts_set = se_facts_full_dict[se]  #se_facts_full_dict是把每个关系对应的三元组使用集合聚合在一起。这里是把关系的索引等于1的三元组取出来
        se_test_facts_pos = np.array([[s, p, o] for s, p, o in test_data if p == se])
        #添加
        if (len(se_test_facts_pos)) ==0:
            continue

        se_test_facts_pos_size = len(se_test_facts_pos) #总的是（398*3）=SL中是43*3

        se_test_facts_neg = np.array([[d1, se, d2] for d1, d2 in drug_combinations
                                      if (d1, se, d2) not in se_all_facts_set
                                      and (d2, se, d1) not in se_all_facts_set])  #（401202，3）

        # shuffle and keep negatives with size equal to positive instances so positive to negative ratio is 1.txt:1.txt （一对一采样）
        np.random.shuffle(se_test_facts_neg)
        se_test_facts_neg = se_test_facts_neg[:se_test_facts_pos_size, :] #这一步就是从所有的负样本中取出跟正样本一样的数目（398，3

        # ）
        print(se_test_facts_pos)
        print(se_test_facts_neg)

        set_test_facts_all = np.concatenate([se_test_facts_pos, se_test_facts_neg]) #这一步就是聚合，把正负样本合成一个 size=（796，3）  size（86*3）
        se_test_facts_labels = np.concatenate([np.ones([len(se_test_facts_pos)]), np.zeros([len(se_test_facts_neg)])])#size=796
        se_test_facts_scores = model.predict(set_test_facts_all)

        se_ap = average_precision(se_test_facts_labels, se_test_facts_scores)
        #把50这个先去掉
        # se_p50 = precision_at_k(se_test_facts_labels, se_test_facts_scores, k=50)
        se_auc_pr = auc_pr(se_test_facts_labels, se_test_facts_scores)
        se_auc_roc = auc_roc(se_test_facts_labels, se_test_facts_scores)

        se_ap_list.append(se_ap)
        se_auc_roc_list.append(se_auc_roc)
        se_auc_pr_list.append(se_auc_pr)
        # se_p50_list.append(se_p50)  ，先去掉

        se_code = se_name.replace("SE:", "")
        metrics_per_se[se] = {"ap": se_ap, "auc-roc": se_auc_roc, "auc-pr": se_auc_pr}
        print("AP: %1.4f - AUC-ROC: %1.4f - AUC-PR: %1.4f " %
              (se_ap, se_auc_roc, se_auc_pr), flush=True) #先去掉se_p50,  se_code暂时去掉

    se_ap_list_avg = np.average(se_ap_list)
    print(se_ap_list_avg)
    se_auc_roc_list_avg = np.average(se_auc_roc_list)
    print( se_auc_roc_list_avg)
    se_auc_pr_list_avg = np.average(se_auc_pr_list)
    print(se_auc_pr_list_avg)
    # se_p50_list_avg = np.average(se_p50_list)#这个暂时不输出

    print("================================================================================")
    print("[AVERAGE] AP: %1.4f - AUC-ROC: %1.4f - AUC-PR: %1.4f" %
          (se_ap_list_avg, se_auc_roc_list_avg, se_auc_pr_list_avg), flush=True)
    print("================================================================================")


if __name__ == '__main__':
    main()

