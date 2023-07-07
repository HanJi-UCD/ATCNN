function [Satisfaction_vector, Rho_iu] = measure_satisfaction_apprx(env, X_iu, R_required, SNR)
B = env.B;
AP_num = env.AP_num;
UE_num = env.UE_num;
Capacity = B.*log2(1 + 10.^(SNR./10)); % convert from dB unit to ratio
Rho_iu = zeros(AP_num, UE_num);
Satisfaction_vector = zeros(AP_num, UE_num);
for i = 1:env.AP_num
    connected_UE = find(X_iu(i, :) == 1);
    if isempty(connected_UE) == 1 % the i-th AP has no connected users
        Satisfaction_vector(i, :) = 0;
    else
        % apprximation method
        X = 1/(length(connected_UE))*ones(1, length(connected_UE));
        Rho_iu(i, connected_UE) = X;
    end
end
Satisfaction_vector = min(sum(Rho_iu.* Capacity)./R_required, 1);
end

