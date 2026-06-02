start cmd /k python catboostNoKNN.py > log_cat.txt 2>&1
start cmd /k python light+KNN.py > log_lgbm.txt 2>&1
start cmd /k python RegLog+KNN.py > log_rl.txt 2>&1
start cmd /k python rforest+KNN.py > log_rf.txt 2>&1
start cmd /k python XGBoost+KNN.py > log_xgb.txt 2>&1