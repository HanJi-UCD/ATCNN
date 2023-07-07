%% Adative running time
clear
clc
UE_num = 30;
AP_num = 5;
M = UE_num;
input = rand((AP_num+1)*UE_num, 1)*1e3;
input((AP_num+1):(AP_num+1):(AP_num+1)*UE_num) = max(input((AP_num+1):(AP_num+1):(AP_num+1)*UE_num), 10); % set min R as 1 M
SNR_max = 1000;
SNR_min = 0;
R_max = 1000;
% load DNN parameters
FC1 = rand(128, (AP_num+1)*UE_num);
Bias1 = rand(128, 1);
FC2 = rand(64, 128);
Bias2 = rand(64, 1);
FC3 = rand(AP_num*UE_num, 64);
Bias3 = rand(AP_num*UE_num, 1);

index = 1:(AP_num+1)*M;
index((AP_num+1):(AP_num+1):(AP_num+1)*M) = [];
for i = 1:1100
    input_now = input;
    if i == 101
        tic
    end
    % normalize 
    input_now(index) = (input_now(index) - SNR_min)/(SNR_max - SNR_min);
    R = input_now(AP_num+1 : AP_num+1 : M*(AP_num+1));
    R = (log(R)./(log(R_max))); 
    input_now(AP_num+1 : AP_num+1 : M*(AP_num+1)) = R;
    % feed into C-DNN
    output1 = FC1*input_now + Bias1;
    output1 = max(normalize(output1), 0);
    output2 = FC2*output1 + Bias2;
    output2 = max(normalize(output2), 0);
    output3 = FC3*output2 + Bias3;
    output3 = sigmoid(normalize(output3));

    final_output = reshape(output3, AP_num, UE_num);
 
    output = find(final_output == max(final_output)); %% Costing time partly here
    output = output' - (0 : AP_num: (UE_num-1)*AP_num);   
end
toc

function y = sigmoid(x)
y = zeros(1, length(x));
    for i = 1:1:length(x)
        y(i) = 1/(1+exp(1)^(-x(i)));
    end
end


