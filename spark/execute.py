import os

os.system("spark-submit --deploy-mode cluster --master yarn --num-executors 7 --executor-cores 16 --executor-memory 16g clustering_analysis.py &> log.txt &")