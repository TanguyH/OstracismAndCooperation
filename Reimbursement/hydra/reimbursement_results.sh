#!/bin/bash

module load worker
wsub -batch reimbursement/re_competition_FF.pbs -data exploration_20.csv
wsub -batch reimbursement/re_competition_FT.pbs -data exploration_20.csv
wsub -batch reimbursement/re_competition_TF.pbs -data exploration_20.csv
wsub -batch reimbursement/re_competition_TT.pbs -data exploration_20.csv
wsub -batch reimbursement/re_peer_FF.pbs -data exploration_20.csv
wsub -batch reimbursement/re_peer_FT.pbs -data exploration_20.csv
wsub -batch reimbursement/re_peer_TF.pbs -data exploration_20.csv
wsub -batch reimbursement/re_peer_TT.pbs -data exploration_20.csv
wsub -batch reimbursement/re_pool_FF.pbs -data exploration_20.csv
wsub -batch reimbursement/re_pool_FT.pbs -data exploration_20.csv
wsub -batch reimbursement/re_pool_TF.pbs -data exploration_20.csv
wsub -batch reimbursement/re_pool_TT.pbs -data exploration_20.csv
wsub -batch reimbursement/re_param.pbs -data param_RJ_all.csv
