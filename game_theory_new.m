function [payoff_vector, X_iu, sat_new, iter] = game_theory_new(env, Satisfaction_vector, X_iu, R_required, SNR)
%% 'Load balancing game with shadowing effect for indoor hybrid LiFi/RF networks' 
%% Algorithm 2
% for all UE do
%     calculate the mutation probabilty for each UE
%     generate random number
%     apply mutation rule
% end
% 
% for all AP do
%     allocate time resource using 1/N
% end
%
% stop until no UE mutation occurs
%%
N_f = env.AP_num; % number of AP candidates for each UE
count = 0;
mode = 0;
payoff_vector = 0;
N = env.UE_num;
while mode <= N
    estimated_payoff = zeros(1, N_f);
    sat = zeros(N_f, env.UE_num);
    mutation_probability = zeros(1, env.UE_num); 
    aver_payoff = sum(Satisfaction_vector)/env.UE_num; 
    for i = 1:env.UE_num       
        if Satisfaction_vector(i) < aver_payoff
            mutation_probability(i) = 1 - Satisfaction_vector(i)/aver_payoff;
        else
            mutation_probability(i) = 0;
        end
        x = rand(1);
        %% apply mutation rule
        if x < mutation_probability(i)
            old_AP = find(X_iu(:, i) == 1); 
            for j = 1:N_f % <------------------- only consider 4 LiFi APs as connectable candidates          
                if env.AP_num == 5
                    X_iu(:, i) = [0 0 0 0 0]';
                    X_iu(j, i) = 1;
                    AP_index = 1:N_f;
                else
                    AP_index = zeros(1, N_f);
                    sorted_AP = sort(SNR(:, i),'descend');
                    sorted_AP = sorted_AP(1:N_f);             
                    for jj = 1:N_f
                        AP_index(jj) = find(SNR(:, i) == sorted_AP(jj), 1);
                    end
                    X_iu(:, i) = zeros(env.AP_num, 1);
                    %X_iu(:, i) = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]';
                    X_iu(AP_index(j), i) = 1;
                end 
                % use optimization or 1/N every time ???
                % method 1: 1/N, less complexity
                [sat(j, :), ~] = measure_satisfaction_apprx(env, X_iu, R_required, SNR);
                % method 2: distributed optimization, more accuracy but with more complexity
                % [sat(j, :), ~] = measure_satisfaction(env, X_iu, R_required, SNR);
                estimated_payoff(j) = sum(sat(j, :))/env.UE_num;    
            end  
            % find the mutated AP for UE j
            AP = find(estimated_payoff == max(estimated_payoff));   
            if length(AP) > 1
                % choose the closest AP  
                SNR_set = SNR(AP_index(AP), i);
                coloum = AP(SNR_set == max(SNR_set));
                if env.AP_num == 5
                    X_iu(:, i) = [0 0 0 0 0]';
                    X_iu(coloum, i) = 1;
                else
                    X_iu(:, i) = zeros(env.AP_num, 1);
                    % X_iu(:, i) = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]';
                    X_iu(AP_index(coloum), i) = 1;
                end          
            else
                if env.AP_num == 5
                    X_iu(:, i) = [0 0 0 0 0]';
                    X_iu(AP, i) = 1;
                else
                    X_iu(:, i) = zeros(env.AP_num, 1);
                    % X_iu(:, i) = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]';
                    X_iu(AP_index(AP), i) = 1;
                end       
            end
            AP_mutated = find(X_iu(:, i) == 1);
            if AP_mutated ~= old_AP % mutation still occurs
                % For updating satisfaction
                [Satisfaction_vector, ~] = measure_satisfaction_apprx(env, X_iu, R_required, SNR);
                aver_payoff = sum(Satisfaction_vector)/env.UE_num;
                if aver_payoff > payoff_vector(end)
                    mode = 0;
                    payoff_vector = [ payoff_vector,  aver_payoff]; 
                else
                    mode = mode + 1; % mutated UE still have the same payoff
                end                                            
            else 
                mode = mode + 1;  % no mutation occurs anymore
                payoff_vector = [ payoff_vector,  payoff_vector(end)]; 
            end            
        else
            % mode = mode + 1; % mutation probability is small or 0
            payoff_vector = [ payoff_vector,  payoff_vector(end)]; 
            % no mutation action for UE j
        end
        if mode > N
            break
        end
    count = count + 1;     
    
    end 
    
    if aver_payoff >= 0.99
        break
    end
    
end

iter = count;
sat_new = Satisfaction_vector;

end


