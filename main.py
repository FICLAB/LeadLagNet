import pandas as pd
from loss_fun import *
from Models import *
import pickle
from evaluator import rank_evaluator
import random
from torch import optim
import gc
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--max-epoch', type=int, default='100',
                    help='Training max epoch')
parser.add_argument('--weight-constraint', type=float, default='0',
                    help='L2 weight constraint')
parser.add_argument('--tau', type=float, default='1.0',
                    help='gumble distribution sharpness')
parser.add_argument('--wait-epoch', type=int, default='30',
                    help='Training min epoch')
parser.add_argument('--eta', type=float, default='1e-4',
                    help='Early stopping')
parser.add_argument('--lr', type=float, default='5e-4',
                    help='Learning rate ')
parser.add_argument('--device', type=str, default='1',
                    help='GPU to use')
parser.add_argument('--heads-att', type=int, default='8',
                    help='attention heads')
parser.add_argument('--hidn-att', type=int, default='64',
                    help='attention hidden nodes')
parser.add_argument('--hidn-rnn', type=int, default='64',
                    help='rnn hidden nodes')
parser.add_argument('--ln-rnn', type=int, default='4',
                    help='rnn layer num')
parser.add_argument('--k', type=int, default='5',
                    help='number of messaging mechanism')
parser.add_argument('--rnn-length', type=int, default='24',
                    help='rnn length')
parser.add_argument('--sqmode', type=int, default='0',
                    help='1:lstm only 2:use_mrs, 3:use_scd')
parser.add_argument('--dropout', type=float, default='0.5',
                    help='dropout rate')
parser.add_argument('--clip', type=float, default='0.25',
                    help='rnn clip')
parser.add_argument('--task', type=str, default='reg',
                    help='rank:rank, reg:regression, cls:classification')
parser.add_argument('--period', type=str, default='bull',
                    help='bull, mixed, bear')
parser.add_argument('--log', type=bool, default=True,
                    help='write_logs')
parser.add_argument('--alpha', type=float, default='3',
                    help='init slope anealing')
parser.add_argument('--save', type=bool, default=True,
                    help='save models')
parser.add_argument('--batch-size', type=int, default=5,
                    help='train per batch')
parser.add_argument('--seednum', type=int, default=0,
                    help='start seed num')


def read_data(task, period):

    if period == "bull":
        data_path = "./data/bull_dataset.csv"
        supply_relation_path = "./data/bull_relation.pkl"
    elif period == "bear":
        data_path = "./data/bear_dataset.csv"
        supply_relation_path = "./data/bear_relation.pkl"
    elif period == "mixed":
        data_path = "./data/mixed_dataset.csv"
        supply_relation_path = "./data/mixed_relation.pkl"
    else:
        print("period error")

    with open(supply_relation_path, 'rb') as handle:
        supply_relation = pickle.load(handle)
    # supply_relation = torch.tensor(supply_relation, device=DEVICE)

    market_data = pd.read_csv(data_path)
    num_stock = len(market_data.STOCK_ID.unique())
    num_timestep = len(market_data.date.unique())

    x_col = ['x_earning_rate', 'x_BIDLO_rate', 'x_ASKHI_rate', 'x_turnover',
             'x_SMA5_rate', 'x_SMA15_rate', 'x_SMA30_rate', 'x_MIDPRICE_rate',
             'ADX', 'MACD','AROONOSC', 'PPO','ATR', 'NATR','AD', 'OBV',
             'x_earning_rate_rank', 'x_BIDLO_rate_rank', 'x_ASKHI_rate_rank', 'x_turnover_rank',
             'x_SMA5_rate_rank', 'x_SMA15_rate_rank', 'x_SMA30_rate_rank', 'x_MIDPRICE_rate_rank',
             'x_ADX_rank', 'x_MACD_rank',
             "x_AROONOSC_rank", "x_PPO_rank",
             'x_ATR_rank', 'x_NATR_rank',
             'x_AD_rank', 'x_OBV_rank']

    x_ = market_data[x_col].to_numpy().reshape(num_stock, num_timestep, -1).transpose(1, 0, 2)

    if task == "rank":
        y_ = market_data["y_earning_rate_tmr_rank"].to_numpy().reshape(num_stock, num_timestep).transpose(1, 0)
        y_ret = market_data["y_earning_rate_tmr"].to_numpy().reshape(num_stock, num_timestep).transpose(1, 0)
    elif task == "reg":
#     elif task == "reg":
        y_ = market_data["y_earning_rate_tmr"].to_numpy().reshape(num_stock, num_timestep).transpose(1, 0)
        y_ret = market_data["y_earning_rate_tmr"].to_numpy().reshape(num_stock, num_timestep).transpose(1, 0)
    else:
        print("error")

    return x_, y_, y_ret, supply_relation

def train(model, x, relation, y,  y_ret):
    model.train()
    x_train = torch.tensor(x, dtype=torch.double,device = DEVICE)
    y_train = torch.tensor(y, dtype=torch.double, device = DEVICE).unsqueeze(-1)
    relation = torch.tensor(relation, dtype=torch.double, device=DEVICE)
    num_sample = (x_train.size()[0] - rnn_length + 1) * x_train.size()[1]
    seq_len = len(x_train)
    train_seq = list(range(seq_len))[rnn_length - 1:]
    random.shuffle(train_seq)
    total_loss = 0
    total_loss_count = 0
    batch_train = args.batch_size

    outputs = []
    labels = []
    preds = []

    true_rets = []
    for i in train_seq:
        output = model(x_train[i - rnn_length + 1: i + 1], relation)
        label = y_train[i]
        outputs.append(output)
        labels.append(label)
        preds.append(output.detach().cpu().numpy())
        true_rets.append(y_ret[i])
        total_loss_count += 1
        if total_loss_count % batch_train == 0:
            outputs = torch.cat(outputs, dim=0)
            labels = torch.cat(labels, dim=0)
            loss = loss_fun(outputs, labels)
            total_loss += loss.item() * len(labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            outputs = []
            labels = []
    if total_loss_count % batch_train != 0:
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        loss = loss_fun(outputs, labels)
        total_loss += loss.item() * len(labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    metrics, _, _ = rank_evaluator(np.array(preds).squeeze(), np.array(true_rets))
    return model, total_loss / num_sample, metrics

def evaluate(model, x_eval, relation, y_eval, y_ret_eval):
    model.eval()
    x_eval = torch.tensor(x_eval, dtype=torch.double,device = DEVICE)
    y_eval = torch.tensor(y_eval, dtype=torch.double, device = DEVICE).unsqueeze(-1)
    relation = torch.tensor(relation, dtype=torch.double, device=DEVICE)
    num_sample = (x_eval.size()[0] - rnn_length + 1) * x_eval.size()[1]

    seq_len = len(x_eval)
    seq = list(range(seq_len))[rnn_length - 1:]

    preds = []
    true_ret = []
    total_loss = 0

    for i in seq:
        output = model(x_eval[i - rnn_length + 1: i + 1], relation)
        loss = loss_fun(output, y_eval[i])
        total_loss += loss.item() * x_eval.size()[1]
        preds.append(output.detach().cpu().numpy())
        true_ret.append(y_ret_eval[i])
    metrics, _ , _ = rank_evaluator(np.array(preds).squeeze(),np.array(true_ret))
    metrics["loss"] = total_loss / num_sample
    return metrics



if __name__ == "__main__":
    args = parser.parse_args()
    DEVICE = "cuda:" + args.device

    print("loading dataset")
    rnn_length = args.rnn_length
    x, y , y_ret, supply_relation= read_data(task = args.task, period = args.period)
    x_train = x[-1024 : -512]
    x_eval = x[-512 - rnn_length: -256]
    x_test = x[-256 - rnn_length:]

    y_train = y[-1024: -512]
    y_eval = y[-512 - rnn_length: -256]
    y_test = y[-256 - rnn_length:]

    y_ret_train = y_ret[-1024 : -512]
    y_ret_eval =  y_ret[-512 - rnn_length: -256]
    y_ret_test =  y_ret[-256 - rnn_length:]

    print("start")

    if args.task == "rank":
        # loss_fun = rank_loss
        loss_fun = torch.nn.MSELoss()
    elif args.task == "reg":
        loss_fun = torch.nn.MSELoss()
    else:
        print("task error train")
    # hyper-parameters
    T = x.shape[0]
    NUM_STOCK = x.shape[1]
    D_MARKET = x.shape[2]
    MAX_EPOCH = args.max_epoch
    task = args.task

    hidn_rnn = args.hidn_rnn
    heads_att = args.heads_att
    hidn_att = args.hidn_att
    dropout_p = args.dropout
    lr = args.lr
    ln = args.ln_rnn
    tau = args.tau
    K = args.k


    log_file  = "./eval_logs/period_{}revised_task_{}_new_hidden_{}_K{}_lr{}_T{}_p_{}_SqMode{}_seednun{}_log_DRSN_OriginDataset.txt".format(args.period,args.task,hidn_rnn, K, args.lr, args.rnn_length, dropout_p, args.sqmode, args.seednum)
    write_log(log_file, "", "w")
    log_str = "epoch{}, " \
              "Loss{:.7f},{:.7f},{:.7f}, MSE{:.7f},{:.7f},{:.7f}, MRR{:.7f},{:.7f},{:.7f}" \
              "AAR{:.7f},{:.7f},{:.7f}, EAAR{:.7f},{:.7f},{:.7f}," \
              "AV{:.7f},{:.7f},{:.7f}, EAV{:.7f},{:.7f},{:.7f}," \
              "MDD{:.7f},{:.7f},{:.7f}, EMDD{:.7f},{:.7f},{:.7f}."



    metric_list = []
    seed = 9
    with torch.cuda.device(DEVICE):
        set_seed(seed)
        model = DRSN(ipt_dim = D_MARKET, t = rnn_length, hid_dim= hidn_rnn, alpha=args.alpha, lr_num=ln, nheads=heads_att, dropout_p = dropout_p, tau_init=tau, sqmode=args.sqmode)
        model = model.to(torch.double)
        model = model.to(device=DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)
        wait_epoch = 0
        eval_epoch_best = 0
        min_loss = 1000000
        model.reset_parameters()
        for epoch in range(200):
            gc.collect()
            torch.cuda.empty_cache()
            model, train_loss, train_metrics = train(model, x_train, supply_relation, y_train, y_ret_train)
            evaluate_metrics = evaluate(model, x_eval, supply_relation, y_eval, y_ret_eval)
            test_metrics = evaluate(model, x_test, supply_relation, y_test, y_ret_test)
            epoch_log_str = log_str.format(epoch,
                                           train_loss, evaluate_metrics["loss"], test_metrics["loss"],
                                           train_metrics["MSE"],evaluate_metrics["MSE"], test_metrics["MSE"],
                                           train_metrics["MRR"], evaluate_metrics["MRR"], test_metrics["MRR"],
                                           train_metrics["AAR"], evaluate_metrics["AAR"], test_metrics["AAR"],
                                           train_metrics["EAAR"], evaluate_metrics["EAAR"], test_metrics["EAAR"],
                                           train_metrics["AV"],evaluate_metrics["AV"], test_metrics["AV"],
                                           train_metrics["EAV"], evaluate_metrics["EAV"], test_metrics["EAV"],
                                           train_metrics["MDD"],evaluate_metrics["MDD"], test_metrics["MDD"],
                                           train_metrics["EMDD"], evaluate_metrics["EMDD"], test_metrics["EMDD"]
                                           )
            if args.log:
                write_log(log_file, epoch_log_str)
            print(epoch_log_str)


            if evaluate_metrics["loss"] < min_loss:
                best_epoch = epoch
                min_loss = evaluate_metrics["loss"]
                # best_state_dict = copy.deepcopy(model.state_dict())
                best_eval_str = epoch_log_str
                best_eval_metric = evaluate_metrics
                test_metric_cor =  test_metrics
                wait_epoch = 0
                if args.save:
                    best_state_dict = copy.deepcopy(model.state_dict())
            else:
                wait_epoch += 1
            #
            if wait_epoch > 50:
                break
        if args.log:
            write_log(log_file, "#################round{}###################".format(str(n)))
            write_log(log_file, best_eval_str)
        if args.save:
            best_model_path = "./Saved_Models/period_{}_task_{}_new_alpha{}_hidden{}_K{}_lr{}_T{}_p_{}_SqMode{}_model:{}_seed:{}.model".format(
                args.period, args.task, args.alpha, hidn_rnn, K, args.lr, args.rnn_length, dropout_p, args.sqmode,
                "DRSN", seed)
            torch.save(best_state_dict ,best_model_path)
        print(best_eval_str)