import tensorflow as tf
import numpy as np
import math
from model import KGCN


def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, test_data = data[4], data[5]
    adj_entity, adj_relation = data[6], data[7]

    model = KGCN(args, n_user, n_entity, n_relation, adj_entity, adj_relation)

    # top-K evaluation settings
    user_list, train_record, test_record, train_weights, test_weights, item_set, k = topk_settings(
        show_topk, train_data, test_data, n_item)
    # 权重初始化
    train_data = weights_init(train_data)
    print('data loaded.')
    writer = open('../src/model/classifier_weight.txt', 'w', encoding='utf-8')
    for round_number in range(args.rounds):
        print('round {} training stage ...'.format(round_number + 1))
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            for step in range(args.n_epochs):
                # training
                np.random.shuffle(train_data)
                start = 0
                # skip the last incomplete minibatch if its size < batch size
                while start + args.batch_size <= train_data.shape[0]:
                    _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                    start += args.batch_size
                    if show_loss:
                        print('\r', 'epoch {}： {}/{}, loss: {:.5f}'.format(step, start, train_data.shape[0], loss), end='')

            # CTR evaluation
            #     train_auc, train_f1 = ctr_eval(sess, model, train_data, args.batch_size)
                # eval_auc, eval_f1 = ctr_eval(sess, model, eval_data, args.batch_size)
                # test_auc, test_f1 = ctr_eval(sess, model, test_data, args.batch_size)

                # print('epoch %d    train auc: %.4f  f1: %.4f' % (step, train_auc, train_f1))
            # save weights in model
            print()
            saver.save(sess, 'model/round' + str(round_number) + '.pth')

            # top-K evaluation
            print('top-10 evaluation stage ...')
            if show_topk:
                recall = topk_eval(
                    sess, model, user_list, train_record, test_record, train_weights, test_weights,
                    item_set, k, args.batch_size)
                # print('precision: ', end='')
                # for i in precision:
                # print('%.4f\t' % precision)
                # print()
                print('recall: ', end='')
                # for i in recall:
                print('%.4f\t' % recall)

            train_data, classifier_weight = update_weights(sess, model, train_data, args.batch_size)
            writer.write('{}\n'.format(classifier_weight))
        # classifier_weight_list.append(classifier_weight)
        print('done with round {}.'.format(round_number + 1))



def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 2000
        k = 10
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        train_weights = get_user_weight(train_data)
        test_weights = get_user_weight(test_data)
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, train_weights, test_weights, item_set, k
    else:
        return [None] * 5


def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2],
                 model.data_weights: data[start:end, 3]}
    return feed_dict


def ctr_eval(sess, model, data, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1 = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
        auc_list.append(auc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list))


def topk_eval(sess, model, user_list, train_record, test_record, train_weights, test_weights, item_set, k, batch_size):
    # precision_list = []
    recall_list = 0
    # ground_truth_num = 0
    np.random.seed(555)
    x = 0
    for user in user_list:
        x += 1
        print('\r', '{}/{}'.format(x, len(user_list)), end='')
        # selective_item_list = list(item_set - train_record[user] - test_record[user])
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size],
                                                    model.data_weights: [train_weights[user]] * batch_size})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start),
                       model.data_weights: [train_weights[user]] * batch_size})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        hit_num = len(set(item_sorted[:k]) & test_record[user])
        # precision_list[k].append(hit_num / k)
        recall = hit_num / len(test_record[user])
        recall_list += recall

    recall = recall_list / len(user_list)
    # recall = recall_hit_num / ground_truth_num
    print()

    return recall


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict


def get_user_weight(data):
    user_history_weight = dict()
    for interaction in data:
        user = interaction[0]
        weight = interaction[3]
        if user not in user_history_weight:
            user_history_weight[user] = weight
    return user_history_weight


def update_weights(sess, model, data, batch_size):
    print('updating weights ...')
    item_record = get_user_record(data, False)
    user_list = set(data[:, 0])
    user_ranking_record = {}
    item_list = data[:, 1]
    print("getting users' ranking list...")
    m = 0
    for user in user_list:
        print('\r', '{}/{}'.format(m, len(user_list)), end='')
        m += 1
        ranking_list = get_user_ranking_list(sess, model, user, batch_size, item_list)
        user_ranking_record[user] = ranking_list
    beta_withexp_sum = 0
    print('\n')
    print('calculating ...')
    auc_list = []
    for i in range(data.shape[0]):
        print('\r', '{}/{}'.format(i, data.shape[0]), end='')
        user = data[i][0]
        item = data[i][1]
        # if user not in user_ranking_record:
        #     # sess, model, user, batch_size, item_list
        #     ranking_list = get_user_ranking_list(sess, model, user, batch_size, item_list)
        #     # get_user_ranking_list(sess, model, user, batch_size, item_list):
        #     user_ranking_record[user] = ranking_list
        ranking_list = user_ranking_record[user]
        hits = 0
        for x in item_record[user]:
            if x in ranking_list[0: ranking_list.index(int(x))]:
                hits += 1
        auc = len(ranking_list) - ranking_list.index(int(item)) - (len(item_record[user]) - hits - 1)
        auc = auc / (len(ranking_list) - len(item_record[user]))
        auc_list.append(auc)
        beta_withexp_sum += (math.exp(-auc) / len(item_record[user]))
    print()
    print('assigning ...')
    updated_data = []
    classifier_weight_numerator = 0
    classifier_weight_denominator = 0
    for i in range(data.shape[0]):
        print('\r', '{}/{}'.format(i, data.shape[0]), end='')
        user = data[i][0]
        item = data[i][1]
        label = data[i][2]
        weight = (math.exp(-auc_list[i]) / beta_withexp_sum / len(item_record[user]))
        updated_data.append([user, item, label, weight])
        classifier_weight_numerator += (weight * (1 + auc_list[i]))
        classifier_weight_denominator += (weight * (1- auc_list[i]))
    print()
    print('done.')
    classifier_weight = 0.5 * math.log(classifier_weight_numerator / classifier_weight_denominator)
    return updated_data, classifier_weight


def weights_init(data):
    print('initializing the weights of train data ...')
    train_record_init = {}
    for idx in range(data.shape[0]):
        user = data[idx][0]
        item = data[idx][1]
        label = data[idx][2]
        weight = data[idx][3]
        if label == 1:
            if user not in train_record_init:
                train_record_init[user] = set()
            train_record_init[user].add(item)
    init_data = []
    print('initializing weights ...')
    for idx in range(data.shape[0]):
        user = data[idx][0]
        item = data[idx][1]
        label = data[idx][2]
        # weight = data[idx][3]
        betau = 1 / len(train_record_init[user])
        weight = betau / len(train_record_init)
        init_data.append([user, item, label, weight])
    init_data = np.array(init_data)
    return init_data


def get_user_ranking_list(sess, model, user, batch_size, item_list):
    start = 0
    item_score_map = {}
    item_list = list(item_list)
    while start + batch_size <= len(item_list):
        items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                model.item_indices: item_list[start:start + batch_size],
                                                model.data_weights: [1] * batch_size})
        for item, score in zip(items, scores):
            item_score_map[item] = score
        start += batch_size
    # padding the last incomplete minibatch if exists
    if start < len(item_list):
        items, scores = model.get_scores(sess,
                                         {model.user_indices: [user] * batch_size,
                                          model.item_indices: item_list[start:] + [item_list[-1]] * (batch_size - len(item_list) + start),
                                          model.data_weights: [1] * batch_size})
        for item, score in zip(items, scores):
            item_score_map[item] = score
    item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
    item_sorted = [i[0] for i in item_score_pair_sorted]
    return item_sorted
