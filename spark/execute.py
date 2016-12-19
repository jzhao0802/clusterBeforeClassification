import os
import time
import datetime


def run(script, n_executors, n_executor_cores, memory, submit_log):

    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    result_dir_s3 = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/Results/" + st + "/"
    result_dir_master = "/home/lichao.wang/code/lichao/test/Results/" + st + "/"
    if not os.path.exists(result_dir_s3):
        os.makedirs(result_dir_s3, 0o777)
    if not os.path.exists(result_dir_master):
        os.makedirs(result_dir_master, 0o777)

    command_str = ("spark-submit --deploy-mode client --master yarn --num-executors " 
                + str(n_executors) 
                + " --executor-cores " 
                + str(n_executor_cores) 
                + " --executor-memory " 
                + str(memory) + "g "
                + script + " "
                + result_dir_master + " " + result_dir_s3 
                + " &> " + result_dir_master + "log.txt &")
    
    with open(submit_log, "a") as log:
        log.write(command_str + "\n")
    
    os.system(command_str)
    
    
if __name__ == "__main__":
    main_script_name = "/home/lichao.wang/code/lichao/test/ClusteringBeforeClassification/clustering_analysis.py"
    main_n_executors = 7
    main_n_executor_cores = 16
    main_executor_memory = 16
    main_submit_log_file = "./submit_log.txt"
    run(main_script_name, main_n_executors, main_n_executor_cores, main_executor_memory, main_submit_log_file)