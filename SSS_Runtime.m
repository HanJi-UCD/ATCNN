%SSS Running time
% 9 LiFi AP
clear
clc
k = 1; % affect the distribution of UE's data rate
Rb = 200;
% load env_4LiFi.mat
load env_9LiFi.mat
env.P_mod = 3; % 3 W
UE_num = 10;
%
sequence = 1100;
SSS_running_time = 0;
env.UE_num = UE_num; %
% normal distribution for UEs
UE_set_normal = zeros(env.UE_num, 3);
UE_set_normal(:, 1:2) = env.X_length*rand(env.UE_num, 2);
R_required = 1e6.*(gamrnd(k, Rb/k, 1, env.UE_num));   
% Calculate SNR
SNR = zeros(env.AP_num, env.UE_num);
for i = 1:env.UE_num
    for j = 1:env.AP_num
        AP = env.AP_set(j, :);
        UE = UE_set_normal(i, :);
        if j == 1              
            SNR(j, i) = SNR_calculation(env, AP, UE, 'WiFi'); % choose mode of network: WiFi
        else
            SNR(j, i) = SNR_calculation(env, AP, UE, 'LiFi'); % choose mode of network: LiFi
        end
    end    
end  
Capacity = env.B.*log2(1 + SNR);
SNR = 10*log10(SNR); % convert SNR to dB
SNR = max(max(SNR, -30), -30);  
for n = 1:sequence 
    %% SSS test 1: call subfunction  
%     if n > 100
%         tic;
%     end  
%     SSS_X_iu = SSS(SNR); % initialized states
%     if n > 100
%         t = toc;
%         SSS_running_time =  SSS_running_time + t;
%     else
%     end
    %% SSS test 2: No subfunction    
    X_iu = zeros(size(SNR));
    if n > 100
        tic;
    end
    for i = 1:env.UE_num
        max_value = 0;
        for j = 1:env.AP_num
            if SNR(j, i) > max_value
                row = j;
                max_value = SNR(j, i);
            end
        end
        X_iu(row, i) = 1;
    end
    if n > 100
        t = toc;
        SSS_running_time = SSS_running_time + t;
    else
    end
end
fprintf('SSS runtime is %d ms \n', SSS_running_time)




