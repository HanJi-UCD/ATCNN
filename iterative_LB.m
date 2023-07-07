function X_iu = iterative_LB(env, X_iu, SNR, R_requirement)
N_f = env.AP_num; % number of AP candidates for each UE
N = env.UE_num;
count = 0;
while count <= N
    % use random order to update UEs
%     random_order = randperm(N); 
    random_order = 1:N;
    for ii = 1:N
        i = random_order(ii); % random order is not trainable
        sorted_AP = sort(SNR(:, i),'descend');
        sorted_AP = sorted_AP(1:N_f);
        AP_index = zeros(1, N_f);
        for jj = 1:N_f
            AP_index(jj) = find(SNR(:, i) == sorted_AP(jj), 1);
        end
        object_function_list = zeros(1, N_f);
        index_old = find(X_iu(:, i) == 1);
        for j = 1:N_f
            X_iu(:, i) = zeros(env.AP_num, 1); 
            X_iu(AP_index(j), i) = 1;
            [Satisfaction_vector, ~] = measure_satisfaction_apprx(env, X_iu, R_requirement, SNR);
            object_function_list(j) = sum(Satisfaction_vector)/N;    
        end    
        % choose AP with highest SNR
        chosen_AP = find(object_function_list == max(object_function_list), 1); 
        index_AP = AP_index(chosen_AP);
        X_iu(:, i) = zeros(env.AP_num, 1); 
        X_iu(index_AP, i) = 1;    
        if index_AP == index_old
            count = count + 1;
        else
            count = 0;
        end
    end
end
end