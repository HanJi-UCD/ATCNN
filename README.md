# ATCNN
Han Ji, Qiang Wang, Xiping Wu, Stephen J. Redmond, and Iman Tavakkolnia, 'Adaptive Target-Condition Neural Network: DNN-aided Load Balancing for Hybrid LiFi and WiFi Networks', under review by IEEE TWC.

This is the final code for ATCNN work, copyright @ Han Ji and Qiang Wang
# Original contributor: https://github.com/wq13552463699/Adaptive-DNN-Aided-Load-Balancing-for-Hybrid-LiFi-and-WiFi-Networks.git

General Introduction: 
Stage1: Dataset collection
1. Run main_dataset_collection_4LiFi.m and main_dataset_collection_9LiFi.m to collect 1000 batches as .csv files.
2. Run Mirror_Mapping.m and global_normalize_trainingdata.m to pre-process the dataset
Stage2: Training and testing
1. Run dataset_crestor.py to generate the .h5 file
2. Run ATCNN_train_loss.py and Global_DNN_train_loss.py to train ATCNN and DNN* respectively. Save the trained model as .pth files.
3. Run ATCNN_acc_test.py to test the prediction accuracy of ATCNN.
4. Run Global_DNN_acc_test.py to test the prediction accuracy of DNN.
Stage3: Evaluation
1. Run simulation1 and simulation2 codes to evaluate the throughput and fairness versus UE number and Required data rate.
2. Run ATCNN_Runtime.m, DNN_Runtime.m, and Benchmark_Runtime.m to record the average runtime in MATLAB.

Note: In each step, some key parameters and file names need revision accordingly.
