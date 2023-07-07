%% generate training data for 4 LiFi APs case
% Adaptive UE number with mirror operation  
clear;
clc;
%% paramaters setting
load env_4LiFi.mat env
env.P_mod = 3; %
k = 1; % shape parameter in gamma distribution of R
Rb = 100; % scale parameter in gamma distribution of R
batch_size = 256; % batch size
batch_num = 1005; % batch number
AP_num = 5; % 4 LiFi and 1 WiFi
UE_num_list = 20;
X_length = 5; % room size
B = 20000000; % bandwidth: 20 Mbps
%% Generate training dataset / testing dataset
input_names = arrayfun(@(i)['input_batch' num2str(i) '.csv'], 1001:batch_num, 'un',0); % need revise here
output_names = arrayfun(@(i)['output_batch' num2str(i) '.csv'], 1001:batch_num, 'un',0); % need revise here
for i = 1:length(input_names) 
    UE_index = floor((i-1)/5) + 1; % need revise here
    UE_num = UE_num_list(UE_index); 
    env.UE_num = UE_num;
    % training input data size for each batch
    input_data = zeros((AP_num + 1)*UE_num, batch_size); % ---> SNR + R_required
    % training output data size for each batch
    output_data = zeros(AP_num*UE_num, batch_size);
    for j = 1:batch_size
        UE_set = zeros(UE_num, 3);
        UE_set(:, 1:2) = X_length*rand(UE_num, 2); 
        % R_required = 1e6.*Rb*ones(1, env.UE_num); % without considering Rb
        R_required = max(min(1e6.*(gamrnd(k, Rb/k, 1, UE_num)), 1e9), 1e6); % Max R is 1000 M, Min R is 1M
        % Calculate SNR
        SNR = zeros(env.AP_num, env.UE_num);
        for ii = 1:env.UE_num
            for jj = 1:env.AP_num
                AP = env.AP_set(jj, :);
                UE = UE_set(ii, :);
                if jj == 1              
                    SNR(jj, ii) = SNR_calculation(env, AP, UE, 'WiFi'); % choose mode of network: WiFi
                else
                    SNR(jj, ii) = SNR_calculation(env, AP, UE, 'LiFi'); % choose mode of network: LiFi
                end
            end    
        end  
        Capacity = env.B.*log2(1 + SNR);
        SNR = 10*log10(SNR); % convert SNR to dB
        SNR = max(SNR, -20); % choose -20 dB as breakpoint for minimum SNR     
        input_data(1 : AP_num*UE_num, j) = reshape(SNR, AP_num*UE_num, 1); % save SNR data
        input_data(AP_num*UE_num + 1 : end, j) = R_required';  % save data rate requirement         
%         %% Mixed FL method ---> iterative method
%         [~, X_iu_FL] = FL_approximation(AP_num, UE_num, SINR, R_required, rule_threshold_NLoS, 0);
        SSS_X_iu = SSS(SNR); % initialized states
        [SSS_Satisfaction_vector, ~] = measure_satisfaction_apprx(env, SSS_X_iu, R_required, SNR);
        [~, X_iu, ~, iter] = game_theory_new(env, SSS_Satisfaction_vector, SSS_X_iu, R_required, SNR); %
%         SSS_X_iu = SSS(SNR); % initialized states
%         X_iu = iterative_LB(env, SSS_X_iu, SNR, R_required); %
        % save output data       
        output_data(:, j) = reshape(X_iu, AP_num*UE_num ,1);     
        fprintf('batch number = %d ', i);
        fprintf('batch sequence = %d \n', j);          
    end
    input_thisname = input_names{i};
    output_thisname = output_names{i};
    % normalize the training data
    % nor_input_data = zeros((env.AP_num+1)*env.UE_num, batch_size);
    nor_input_data = input_data; % without normalization
    % reorder the input data
    reorder_input_data = zeros((AP_num+1)*UE_num, batch_size);
    for h = 1:UE_num
        reorder_input_data((h-1)*(AP_num+1)+1 : h*(AP_num+1)-1, :) = nor_input_data((h-1)*AP_num+1:h*AP_num, :); % save SNR data
        reorder_input_data(h*(AP_num+1), :) = nor_input_data(AP_num*UE_num + h, :);  % save data rate requirement         
    end
    csvwrite(input_thisname, reorder_input_data);  % save input training data
    csvwrite(output_thisname, output_data); % save output training data
end


