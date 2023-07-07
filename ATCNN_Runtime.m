%% Adative running time
clear
clc
UE_num = 20;
AP_num = 10;
M = 50;
input = rand((AP_num+1)*UE_num, 1)*1e3;
input((AP_num+1):(AP_num+1):(AP_num+1)*UE_num) = max(input((AP_num+1):(AP_num+1):(AP_num+1)*UE_num), 10); % set min R as 1 M
SNR_max = 1000;
SNR_min = 0;
R_max = 1000;
% load TCNN parameters
FC1 = rand(6, AP_num+1);
Bias1 = rand(6, 1);
FC2 = rand(64, (AP_num+1)*M);
Bias2 = rand(64, 1);
FC3 = rand(6, 64);
Bias3 = rand(6, 1);
FC4 = rand(AP_num, 12);
Bias4 = rand(AP_num, 1);
time = 0;
quotient = floor(M/UE_num); % 
reminder = mod(M, UE_num); %
index = 1:(AP_num+1)*M;
index((AP_num+1):(AP_num+1):(AP_num+1)*M) = [];
for i = 1:1100
    input_now = input;
    if i == 101
        tic
    end
    % mirroring
    if reminder == 0
        input_now((AP_num+1)*(1:UE_num), :) = input_now((AP_num+1)*(1:UE_num), :)/(quotient); % split R by quotient times
    else
        input_now((AP_num+1)*(1:reminder), :) = input_now((AP_num+1)*(1:reminder), :)/(quotient+1); % split R by (quotient+1) times
        input_now((AP_num+1)*(reminder+1:UE_num), :) = input_now((AP_num+1)*(reminder+1:UE_num), :)/quotient; % split R by quotient times
    end       
    if quotient == 1 % M > 25
        mirrored_set = [input_now; input_now(1:(AP_num+1)*reminder, :)]; % running time is about 0.004 ms
    else % M <= 25
        mirrored_set = input_now; 
        for k = 1:quotient-1
            mirrored_set = [mirrored_set; input_now];      
        end 
        mirrored_set = [mirrored_set; input_now(1:(AP_num+1)*reminder, :)];
    end
    % normalize 
    mirrored_set(index) = (mirrored_set(index) - SNR_min)/(SNR_max - SNR_min);
    R = mirrored_set(AP_num+1 : AP_num+1 : M*(AP_num+1));
    R = (log(R)./(log(R_max))); 
    mirrored_set(AP_num+1 : AP_num+1 : M*(AP_num+1)) = R;
    % feed into C-DNN
    %% condition
    condition = mirrored_set; % stable
    u_C = max(normalize(FC3*max(normalize(FC2*condition + Bias2), 0) + Bias3), 0); %% <------ Mostly cost time here
%     output = zeros(1, UE_num);
%     for u = 1:UE_num
%         target = reorder_input_data(1+(u-1)*(AP_num+1) : u*(AP_num+1)); % choose any pedicted UE here
%         % 
%         u_j = FC1*target + Bias1; 
%         cat = [u_j; u_C];
%         output(u) = find(softmax(FC4*cat) == max(softmax(FC4*cat))); %% Mostly cost time here
%     end
    %% target
    target = reshape(condition(1:UE_num*(AP_num+1)), AP_num+1, UE_num);
    u_j = FC1*target + Bias1; 
    %% combiner
    cat = [u_j; repmat(u_C, 1, UE_num)];
    A = softmax(FC4*cat);
    output = find(A == max(A)); %% Costing time partly here
    output = output - (0 : AP_num: (UE_num-1)*AP_num)';   
end
toc


