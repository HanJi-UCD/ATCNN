%% change normalization for each batch into global batches
% mode = 1, with R 
AP_num = 5;
UE_num = 50;
batch_num = 1010;
batch_size = 256;
input_names1 = arrayfun(@(i)['mirror_input_batch' num2str(i) '.csv'], 1:batch_num, 'un',0);
input_names2 = arrayfun(@(i)['nor_mirror_input_batch' num2str(i) '.csv'], 1:batch_num, 'un',0);
%% set global maximum and mimimum values and then normalize 
SNR_max = 60;
SNR_min = 15;
R_max = 1e9;
R_min = 1e6;
%% global normalization
for i = 1:length(input_names1)  
    input = csvread(input_names1{i});
    % reorder SNR and R
    SNR = zeros(AP_num*UE_num, batch_size);
    R = zeros(UE_num, batch_size);
    for j = 1:UE_num
        SNR((j-1)*AP_num+1 : j*AP_num, :) = input((j-1)*(AP_num+1)+1 : j*(AP_num+1)-1, :);
        R(j, :) = input(j*(AP_num+1), :);            
    end
    % then globally normalize
    SNR = (SNR - SNR_min)/(SNR_max - SNR_min); % Linearly normalize SNR
    R = (log(R/1e6)./(log(R_max/1e6))); % Convert R into Mbps, and then do Logarithmic normalization for R
    nor_input_data = [SNR; R];
    % reorder training data  
    reorder_input_data = zeros((AP_num+1)*UE_num, batch_size);
    for h = 1:UE_num
        reorder_input_data((h-1)*(AP_num+1)+1 : h*(AP_num+1)-1, :) = nor_input_data((h-1)*AP_num+1:h*AP_num, :); % save SNR data
        reorder_input_data(h*(AP_num+1), :) = nor_input_data(AP_num*UE_num + h, :);  % save data rate requirement         
    end           
    csvwrite(input_names2{i}, reorder_input_data);  % save input training data
    fprintf('Finished Normalizaton Batch of %d \n', i);
end



