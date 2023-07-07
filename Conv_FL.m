function X_iu = Conv_FL(UE_num, B, SNR, R_required, rule_threshold)
%% Conventional Fuzzy logic LB method
% Two stages, Stage I: choose UEs connect to WiFi; Stage II: use SSS as PA
% using the closed-form approximation method to replace the fmincon steps
% for very small R, the optimal rho_u solution is 1/N
% for very large R and large UE number, the optimal rho_iu solution is also 1/N
%%
X_iu = zeros(1, UE_num); 
Capacity = B.*log2(1 + 10.^(SNR./10)); % convert from dB unit to ratio
Score = zeros(UE_num, 1);
for i = 1:UE_num
    SNR_WiFi = SNR(1, i);
    SNR_LiFi = max(SNR(2:end, i));
    % WiFi avaliability
    WiFi_ava = 1 - min(R_required(i)/Capacity(1, i), 1); 
    % LiFi avaliability
    LiFi_AP =  SNR(2:end, i) == max(SNR(2:end, i));
    LiFi_ava = 1 - min(R_required(i)/Capacity(LiFi_AP, i), 1);
    %% Step 1: Fuzzification, input vector not scalar
    crisp_values = Conv_Fuzzification(R_required(i), SNR_WiFi, SNR_LiFi, LiFi_ava, WiFi_ava, rule_threshold);
    %% Step 2: Rule evaluation
    rule_values = Conv_Rule_mapping(crisp_values);
    %% Step 3: Defuzzification           
    Score(i) = Conv_Defuzzification(rule_values); 
end
    %% Finally, make decision according to score rank
    sorted_score = sort(Score, 'descend');
    % UE_index = zeros(1, env.UE_num);
    for j = 1:UE_num % ??? order
        UE_index = find(Score' == sorted_score(j));
        X_iu(UE_index) = 1; % connect to WiFi       
        WiFi_connected_UE = find(X_iu(1, :) == 1);
        % use 1/N as PA solution
        time_resource = sum(min(R_required(WiFi_connected_UE)./Capacity(1, WiFi_connected_UE), 1));      
        if time_resource > 1
            break
        end       
    end
    UE_set = 1:UE_num;
    UE_set(WiFi_connected_UE) = [];
    LiFi_UE = UE_set;
    % SSS for LiFi assignment
    for i = 1:length(LiFi_UE)
         row = find(SNR(:, LiFi_UE(i)) == max(SNR(2:end, LiFi_UE(i))));
         X_iu(LiFi_UE(i)) = row; % connect for LiFi
    end
end

function crisp_values = Conv_Fuzzification(R, SNR_WiFi, SNR_LiFi, LiFi_ava, WiFi_ava, rule_threshold)
%% membership functions
crisp_values = zeros(5, 3);
% Required R
R_min = rule_threshold(1, 1)*10^6;
R_max = rule_threshold(1, 5)*10^6; % unit: Mbps
R_low = rule_threshold(1, 2)*10^6; % unit: Mbps
R_medium = rule_threshold(1, 3)*10^6;
R_high = rule_threshold(1, 4)*10^6;
% WiFi SNR
WiFi_SNR_min = rule_threshold(2, 1); 
WiFi_SNR_max = rule_threshold(2, 5); %
WiFi_SNR_low = rule_threshold(2, 2); 
WiFi_SNR_medium = rule_threshold(2, 3); %
WiFi_SNR_high = rule_threshold(2, 4); %
% LiFi SNR, only conisder nearest one LiFi AP for each UE
LiFi_SNR_min = rule_threshold(3, 1); 
LiFi_SNR_max = rule_threshold(3, 5); %
LiFi_SNR_low = rule_threshold(3, 2); 
LiFi_SNR_medium = rule_threshold(3, 3); %
LiFi_SNR_high = rule_threshold(3, 4); %
% WiFi Avalibility
WiFi_ava_min = rule_threshold(4, 1); 
WiFi_ava_max = rule_threshold(4, 5); %
WiFi_ava_low = rule_threshold(4, 2); 
WiFi_ava_medium = rule_threshold(4, 3); %
WiFi_ava_high = rule_threshold(4, 4); %
% LiFi Avalibility
LiFi_ava_min = rule_threshold(5, 1); 
LiFi_ava_max = rule_threshold(5, 5); %
LiFi_ava_low = rule_threshold(5, 2); 
LiFi_ava_medium = rule_threshold(5, 3); %
LiFi_ava_high = rule_threshold(5, 4); %
% membership functions
crisp_values(1, 1) = trimf(R,[R_min R_min R_medium]); % triangular membership function
crisp_values(1, 2) = trimf(R,[R_low R_medium R_high]);
crisp_values(1, 3) = trapmf(R,[R_medium R_high R_max R_max]); % trapezoidal membership function

crisp_values(2, 1) = trapmf(SNR_WiFi,[WiFi_SNR_min WiFi_SNR_min (WiFi_SNR_min+WiFi_SNR_low)/2 WiFi_SNR_medium]); 
crisp_values(2, 2) = trimf(SNR_WiFi,[WiFi_SNR_low WiFi_SNR_medium WiFi_SNR_high]);
crisp_values(2, 3) = trapmf(SNR_WiFi,[WiFi_SNR_medium WiFi_SNR_high WiFi_SNR_max WiFi_SNR_max]); % trapezoidal membership function

crisp_values(3, 1) = trimf(SNR_LiFi,[LiFi_SNR_min LiFi_SNR_min LiFi_SNR_medium]); 
crisp_values(3, 2) = trimf(SNR_LiFi,[LiFi_SNR_low LiFi_SNR_medium LiFi_SNR_high]);
crisp_values(3, 3) = trimf(SNR_LiFi,[LiFi_SNR_medium LiFi_SNR_max LiFi_SNR_max]); 

crisp_values(4, 1) = trimf(WiFi_ava,[WiFi_ava_min WiFi_ava_min WiFi_ava_medium]); 
crisp_values(4, 2) = trimf(WiFi_ava,[WiFi_ava_low WiFi_ava_medium WiFi_ava_high]);
crisp_values(4, 3) = trimf(WiFi_ava,[WiFi_ava_medium WiFi_ava_max WiFi_ava_max]); 

crisp_values(5, 1) = trimf(LiFi_ava,[LiFi_ava_min LiFi_ava_min LiFi_ava_medium]); 
crisp_values(5, 2) = trimf(LiFi_ava,[LiFi_ava_low LiFi_ava_medium LiFi_ava_high]);
crisp_values(5, 3) = trimf(LiFi_ava,[LiFi_ava_medium LiFi_ava_max LiFi_ava_max]);
end



function final_score = Conv_Defuzzification(rule_values)
% input values: 1*9 vector
% calculate centra gravity
x = 0 : 0.001 : 1 ;
rule_state_high = max([rule_values(1), rule_values(6), rule_values(9)]);
rule_state_medium = max([rule_values(2), rule_values(5), rule_values(7)]);
rule_state_low = max([rule_values(3), rule_values(4), rule_values(8)]);
mf1 = min(trapmf( x, [0, 0, 0.3, 0.6] ), repmat(rule_state_low, 1, length(x))); 
mf2 = min(trimf( x, [0.3, 0.6, 0.8] ), repmat(rule_state_medium, 1, length(x)));
mf3 = min(trapmf( x, [0.6, 0.8, 1, 1] ), repmat(rule_state_high, 1, length(x)));
mf = max ( max ( mf1, mf2 ), mf3 ) ; % upper edge function
final_score = sum(mf.*x, 2) ./ sum(mf, 2); 
end


function rule_values = Conv_Rule_mapping(crisp_values)
% 9 rules
% input crisp_values is a matrix with 5*3 dimension
% operator: AND
rule_values = zeros(1, 9);
%% rule 1: 
rule_values(1) = min(crisp_values(1, 1), max(crisp_values(4, 2), crisp_values(4, 3)));
%% rule 2: 
rule_values(2) = min(max(crisp_values(1, 2), crisp_values(1, 3)), max(crisp_values(4, 2), crisp_values(4, 3)));
%% rule 3: 
rule_values(3) = min(crisp_values(4, 1), max(crisp_values(5, 2), crisp_values(5, 3)));
%% rule 4:
rule_values(4) = min([crisp_values(2, 1), max(crisp_values(3, 2), crisp_values(3, 1)), crisp_values(4, 1), crisp_values(5, 1)]);
%% rule 5:
rule_values(5) = min([crisp_values(2, 1), crisp_values(3, 1), crisp_values(4, 1), crisp_values(5, 1)]);
%% rule 6:
rule_values(6) = min([max(crisp_values(2, 2), crisp_values(2, 3)), crisp_values(3, 1), crisp_values(4, 1), crisp_values(5, 1)]);
%% rule 7: 
rule_values(7) = min([max(crisp_values(2, 2), crisp_values(2, 3)), max(crisp_values(3, 2), crisp_values(3, 3)), crisp_values(4, 1), crisp_values(5, 1)]);
%% rule 8:
rule_values(8) = min(crisp_values(1, 3), crisp_values(3, 3));
%% rule 9:
rule_values(9) = min([max(crisp_values(1, 1), crisp_values(1, 2)), crisp_values(2, 3), max(crisp_values(4, 2), crisp_values(4, 3))]);
end






