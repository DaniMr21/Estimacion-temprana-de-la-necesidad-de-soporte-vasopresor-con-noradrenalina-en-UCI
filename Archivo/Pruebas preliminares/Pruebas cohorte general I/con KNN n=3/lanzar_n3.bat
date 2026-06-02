start cmd /k python baseline_catboost+KNN_n3.py > log_cat_n3.txt 2>&1
start cmd /k python baseline_light+KNN_n3.py > log_lgbm_n3.txt 2>&1
start cmd /k python baseline_reglog+KNN_n3.py > log_rl_n3.txt 2>&1
start cmd /k python baseline_rforest+KNN_n3.py > log_rf_n3.txt 2>&1
start cmd /k python baseline_XGBoost+KNN_n3.py > log_xgb_n3.txt 2>&1