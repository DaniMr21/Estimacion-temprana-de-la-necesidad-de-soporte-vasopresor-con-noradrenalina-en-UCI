start cmd /k python baseline_catboost+KNN_n5.py > log_cat_n5.txt 2>&1
start cmd /k python baseline_light+KNN_n5.py > log_lgbm_n5.txt 2>&1
start cmd /k python baseline_reglog+KNN_n5.py > log_rl_n5.txt 2>&1
start cmd /k python baseline_rforest+KNN_n5.py > log_rf_n5.txt 2>&1
start cmd /k python baseline_XGBoost+KNN_n5.py > log_xgb_n5.txt 2>&1