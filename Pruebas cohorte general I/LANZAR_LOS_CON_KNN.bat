start cmd /k python baseline_reglog+KNN.py > log_rl.txt 2>&1
start cmd /k python baseline_XGBoost+KNN.py > log_xgb.txt 2>&1
start cmd /k python baseline_rforest+KKK.py > log_rf.txt 2>&1
start cmd /k python baseline_light+KNN.py > log_lgbm.txt 2>&1
start cmd /k python baseline_catboost+KNN.py > log_cat.txt 2>&1