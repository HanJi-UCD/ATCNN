% Runtime test for GT, iterative method, and FL
% 9 LiFi AP
clear
clc
k = 1; % affect the distribution of UE's data rate
Rb = 100;
% load env_4LiFi.mat
load env_9LiFi.mat
sequence = [11, 11, 11, 11, 11]; % Monto_Carlo
env.P_mod = 3; % 3 W
UE_num = 50;
B = 20*1e6;
conv_FL_rule_threshold = [0 0 Rb 2*Rb 10000; 20 40 50 60 70; 30 32 35 37 38.5; 0 0.2 0.5 0.8 1; 0 0.2 0.5 0.8 1];
running_time = zeros(1, length(UE_num));
for m = 1:length(UE_num)
    env.UE_num = UE_num(m); %
    for n = 1:sequence(m)
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
% test         
        if n > 1
            tic;
        end
        %% GameTheory runtime
%         SSS_X_iu = SSS(SNR); % initialized states
%         [SSS_Satisfaction_vector, ~] = measure_satisfaction_apprx(env, SSS_X_iu, R_required, SNR); 
%         [~, SSS_GT_X_iu, ~, iter] = game_theory_new(env, SSS_Satisfaction_vector, SSS_X_iu, R_required, SNR); %
        %% iterative method runtime
        SSS_X_iu = SSS(SNR); % initialized states
        iter_X_iu = iterative_LB(env, SSS_X_iu, SNR, R_required); %
        %% FL method runtime
%         conv_X_iu = Conv_FL(UE_num(m), B, SNR, R_required, conv_FL_rule_threshold);
        if n > 1
            t = toc;
            running_time(m) =  running_time(m) + t;
        else
            t = toc;
            running_time(m) =  running_time(m);
        end
        fprintf('UE number = %d ', UE_num(m));
        fprintf('Running time = %d ', t);
        fprintf('Sequence = %d \n', n);        
    end
    % running time
    running_time(m) = running_time(m)/10;
end
 fprintf('Average Running time = %d ', running_time);




