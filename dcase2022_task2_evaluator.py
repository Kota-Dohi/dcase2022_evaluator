import os
import sys
import csv
from io import StringIO
import glob
import re
import numpy
import itertools
import scipy.stats
from sklearn import metrics
import pandas as pd


##############################################################################
# static values
##############################################################################
# Expected directory structure
# ./dcase2022_evaluator/
#       ./teams "Directory containing team results"
#               ./<team name> "Directory containing anomaly score and decision result"
#       ./ground_truth_data "Directory where the true value is stored"
#       ./ground_truth_domain "Directory where the domain assignment is stored"
#       ./teams_result "Directory created after execution."

# directory path
TEAMS_ROOT_DIR = "./teams"
RESULT_DIR = "./teams_result"
GROUND_TRUTH_DATA_DIR = "./ground_truth_data"
GROUND_TRUTH_DOMAIN_DIR = "./ground_truth_domain"

# variables that do not change
COLUMNS = ["AUC (all)", "AUC (source)", "AUC (target)", "pAUC", "precision (source)", "precision (target)",
    "recall (source)", "recall (target)", "F1 score (source)", "F1 score (target)"]
MAX_FPR = 0.1
SCORE_COL = 1

##############################################################################
# common def
##############################################################################
# save csv
def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


# CSV format text to a list of rows decomposed as lists
def csv_text_to_list(csv_text):
    f = StringIO(csv_text)
    reader = csv.reader(f, delimiter=',')
    return [row for row in reader]


# extract machine types from ground truth
def get_machines(load_dir, ext=".csv"):
    query = os.path.abspath("{base}/ground_truth_*{ext}".format(base=load_dir,
                                                                ext=ext))
    machines = sorted(glob.glob(query))
    machines = [os.path.basename(f).split("_")[2] for f in machines]
    machines = sorted(list(set(machines)))
    return machines


# extract section id from anomaly score csv
def get_section_ids(target_dir, ext=".csv"):
    query = os.path.abspath("{target_dir}/ground_truth_*{ext}".format(target_dir=target_dir,
                                                                      ext=ext))
    paths = sorted(glob.glob(query))
    ids = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('section_[0-9][0-9]', ext_id) for ext_id in paths]
    ))))
    return ids


# read score from csv
def read_score(file_path, decision=False):
    with open(file_path) as score_file:
        score_list = list(csv.reader(score_file))
    score_data = [float(score[SCORE_COL]) for score in sorted(score_list)]
    if decision:
        score_data = [int(s) for s in score_data]

    return numpy.array(score_data)


# Jackknife resampling - https://en.wikipedia.org/wiki/Jackknife_resampling
def jackknife_estimate(fn, var_list):
    # See section IV on page 6 on https://hal.inria.fr/hal-02067935/file/mesaros_TASLP19.pdf
    # Reference: A. Mesaros et al., "Sound Event Detection in the DCASE 2017 Challenge," in IEEE/ACM Transactions on Audio,
    #            Speech, and Language Processing, vol. 27, no. 6, pp. 992-1006, June 2019, doi: 10.1109/TASLP.2019.2907016.

    def removed_i(var_list, remove_i):
        return [v[[i for i in range(len(v)) if i != remove_i]] for v in var_list]
    var_list = [numpy.array(v) for v in var_list]
    N = len(var_list[0])
    # (1)
    theta_hat = fn(*var_list)
    # (2)
    thetai_hats = [fn(*removed_i(var_list, i)) for i in range(N)]
    # (3)
    theta_hat_mean = numpy.mean(thetai_hats)
    # (4)
    thetai_tildes = [N * theta_hat - (N - 1) * thetai_hat for thetai_hat in thetai_hats]
    # (5)
    theta_hat_jack = numpy.mean(thetai_tildes)
    # (6)
    sigma_hat_jack = numpy.sqrt(numpy.sum([(thi - theta_hat_mean)**2 for thi in thetai_hats]) / (N * (N-1)))
    # (7) - CI only
    confidence = 0.95
    dof = N - 1
    t_crit = numpy.abs(scipy.stats.t.ppf((1 - confidence) / 2, dof))
    ci95_jack = t_crit * sigma_hat_jack

    return theta_hat_jack, ci95_jack


# [main] output the result from the specified directory and machine type
def output_result(target_dir, machines, section_ids):
    print(target_dir)
    csv_lines = []
    all_y_preds, all_y_trues = [], []
    all_df = pd.DataFrame(columns=["section"] + COLUMNS)
    for machine_idx, target_machine in enumerate(machines):
        print("[{idx}/{total}] machine type : {target_machine}".format(target_machine=target_machine,
                                                                       idx=machine_idx+1,
                                                                       total=len(machines)))
        csv_lines.append([target_machine])
        df = pd.DataFrame(columns=["section"] + COLUMNS).set_index('section')

        for section_id in section_ids:
            sidx = section_id.split("_", 1)[1]
            print(section_id)

            # Load results and ground truth
            anomaly_score_path = "{dir}/anomaly_score_{machine}_{section}_test.csv".format(dir=target_dir,
                                                                                                machine=target_machine,
                                                                                                section=section_id)
            decision_result_path = "{dir}/decision_result_{machine}_{section}_test.csv".format(dir=target_dir,
                                                                                                    machine=target_machine,
                                                                                                    section=section_id)
            ground_truth_path = "{dir}/ground_truth_{machine}_{section}_test.csv".format(dir=GROUND_TRUTH_DATA_DIR,
                                                                                                machine=target_machine,
                                                                                                section=section_id)
            gt_domain_path = "{dir}/ground_truth_{machine}_{section}_test.csv".format(dir=GROUND_TRUTH_DOMAIN_DIR,
                                                                                                machine=target_machine,
                                                                                                section=section_id)

            y_pred_all = read_score(os.path.abspath(anomaly_score_path))
            y_true_all = read_score(os.path.abspath(ground_truth_path))
            y_domain = read_score(os.path.abspath(gt_domain_path))
            decision_result_data_all = read_score(os.path.abspath(decision_result_path), decision=True)
            all_y_preds.extend(y_pred_all)
            all_y_trues.extend(y_true_all)

            # Evaluate for whole section
            df.loc[sidx, 'AUC (all)'] = metrics.roc_auc_score(y_true_all, y_pred_all)
            df.loc[sidx, 'pAUC'] = metrics.roc_auc_score(y_true_all, y_pred_all, max_fpr=MAX_FPR)

            # Evaluate for each domain
            for domain in ['source', 'target']:
                domain_idx = {'source': 0, 'target': 1}[domain]

                # Filter by domain
                y_pred_auc = y_pred_all[(y_domain == domain_idx) | (y_true_all != 0)]
                y_true_auc = y_true_all[(y_domain == domain_idx) | (y_true_all != 0)]
                y_pred = y_pred_all[y_domain == domain_idx]
                y_true = y_true_all[y_domain == domain_idx]
                decision_result_data = decision_result_data_all[y_domain == domain_idx]

                if len(y_true) != len(y_pred) or len(y_true) != len(decision_result_data):
                    print("number of reference elements:", len(y_true))
                    print("anomaly score element count:", len(y_pred), " path:", anomaly_score_path)
                    print("decision data element count:", len(decision_result_data), " path:", decision_result_path)
                    print("some elements are missing")
                    return -1

                # calc result
                df.loc[sidx, f'AUC ({domain})'] = metrics.roc_auc_score(y_true_auc, y_pred_auc)
                tn, fp, fn, tp = metrics.confusion_matrix(y_true, decision_result_data).ravel()
                prec = tp / numpy.maximum(tp + fp, sys.float_info.epsilon)
                recall = tp / numpy.maximum(tp + fn, sys.float_info.epsilon)
                df.loc[sidx, f'precision ({domain})'] = prec
                df.loc[sidx, f'recall ({domain})'] = recall
                df.loc[sidx, f'F1 score ({domain})'] = 2.0 * prec * recall / numpy.maximum(prec + recall, sys.float_info.epsilon)

        csv_lines.extend(csv_text_to_list(df.to_csv()))
        all_df = pd.concat([all_df, df])

        csv_lines.append(["", "", "AUC", "pAUC", "precision", "recall", "F1 score"])
        performance = numpy.array([
            df[["AUC (source)"]].values[:, 0].tolist() + df[["AUC (target)"]].values[:, 0].tolist(),
            df[["pAUC"]].values[:, 0].tolist() + df[["pAUC"]].values[:, 0].tolist(),
            df[["precision (source)"]].values[:, 0].tolist() + df[["precision (target)"]].values[:, 0].tolist(),
            df[["recall (source)"]].values[:, 0].tolist() + df[["recall (target)"]].values[:, 0].tolist(),
            df[["F1 score (source)"]].values[:, 0].tolist() + df[["F1 score (target)"]].values[:, 0].tolist(),
        ], dtype=float)
        amean_performance = numpy.mean(performance, axis=1)
        csv_lines.append(["arithmetic mean", ""] + list(amean_performance))
        hmean_performance = scipy.stats.hmean(numpy.maximum(performance, sys.float_info.epsilon), axis=1)
        csv_lines.append(["harmonic mean", ""] + list(hmean_performance))
        hmean_performance = scipy.stats.hmean(numpy.maximum(performance[:, :len(section_ids)//2], sys.float_info.epsilon), axis=1)
        csv_lines.append(["source harmonic mean", ""] + list(hmean_performance))
        hmean_performance = scipy.stats.hmean(numpy.maximum(performance[:, len(section_ids)//2:], sys.float_info.epsilon), axis=1)
        csv_lines.append(["target harmonic mean", ""] + list(hmean_performance))
        csv_lines.append([])

    csv_lines.append(["", "", "AUC", "pAUC", "precision", "recall", "F1 score"])
    performance_over_all = numpy.array([
        all_df[["AUC (source)"]].values[:, 0].tolist() + all_df[["AUC (target)"]].values[:, 0].tolist(),
        all_df[["pAUC"]].values[:, 0].tolist() + all_df[["pAUC"]].values[:, 0].tolist(),
        all_df[["precision (source)"]].values[:, 0].tolist() + all_df[["precision (target)"]].values[:, 0].tolist(),
        all_df[["recall (source)"]].values[:, 0].tolist() + all_df[["recall (target)"]].values[:, 0].tolist(),
        all_df[["F1 score (source)"]].values[:, 0].tolist() + all_df[["F1 score (target)"]].values[:, 0].tolist(),
    ], dtype=float)
    # calculate averages for AUCs and pAUCs
    amean_performance = numpy.mean(performance_over_all, axis=1)
    csv_lines.append(["arithmetic mean over all machine types, sections, and domains", ""] + list(amean_performance))
    hmean_performance = scipy.stats.hmean(numpy.maximum(performance_over_all, sys.float_info.epsilon), axis=1)
    csv_lines.append(["harmonic mean over all machine types, sections, and domains", ""] + list(hmean_performance))
    n_source = len(all_df[["AUC (source)"]].values[:, 0])
    hmean_performance = scipy.stats.hmean(numpy.maximum(performance_over_all[:, :n_source], sys.float_info.epsilon), axis=1)
    csv_lines.append(["source harmonic mean over all machine types, sections, and domains", ""] + list(hmean_performance))
    hmean_performance = scipy.stats.hmean(numpy.maximum(performance_over_all[:, n_source:], sys.float_info.epsilon), axis=1)
    csv_lines.append(["target harmonic mean over all machine types, sections, and domains", ""] + list(hmean_performance))
    csv_lines.append([])

    all_perf = numpy.array([
        all_df[["AUC (source)"]].values[:, 0].tolist() + all_df[["AUC (target)"]].values[:, 0].tolist()
        + all_df[["pAUC"]].values[:, 0].tolist(),
    ], dtype=float)
    official_score = scipy.stats.hmean(numpy.maximum(all_perf, sys.float_info.epsilon), axis=None)
    csv_lines.append(["official score", "", str(official_score)])

    auc_jack, auc_ci95 = jackknife_estimate(fn=metrics.roc_auc_score, var_list=[all_y_trues, all_y_preds])
    pauc_jack, p_auc_ci95 = jackknife_estimate(fn=lambda a,b: metrics.roc_auc_score(a, b, max_fpr=MAX_FPR), var_list=[all_y_trues, all_y_preds])
    print('########## CI95', auc_ci95, p_auc_ci95, 'official score', official_score, 'auc/pauc jack', auc_jack, pauc_jack)
    csv_lines.append(["official score ci95", "", str(numpy.mean([auc_ci95, p_auc_ci95]))])
    csv_lines.append([])

    # output results
    os.makedirs(RESULT_DIR, exist_ok=True)
    result_file_path = "{result_dir}/{target_dir}_result.csv".format(result_dir=RESULT_DIR,
                                                                     target_dir=os.path.basename(target_dir))
    print("results -> {}".format(result_file_path))
    save_csv(save_file_path=result_file_path, save_data=csv_lines)
    return 0


##############################################################################
# main
##############################################################################
if __name__ == "__main__":
    machine_types = get_machines(load_dir=GROUND_TRUTH_DATA_DIR)
    section_ids = get_section_ids(target_dir=GROUND_TRUTH_DATA_DIR)

    team_dirs = glob.glob("{root_dir}/*/*".format(root_dir=TEAMS_ROOT_DIR))
    if os.path.isdir(RESULT_DIR):
        print("the result directory exist")
        sys.exit(-1)

    for idx, team_dir in enumerate(team_dirs):
        print("[{idx}/{total}] team name : {team_dir}".format(team_dir=os.path.basename(team_dir),
                                                              idx=idx+1,
                                                              total=len(team_dirs)))
        if os.path.isdir(team_dir):
            normal_end_flag = output_result(team_dir, machine_types, section_ids)
            if normal_end_flag == -1:
                print("abnormal termination")
                sys.exit(-1)
        else:
            print("{} is not directory.".format(team_dir))


