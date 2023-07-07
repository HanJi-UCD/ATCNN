function X_iu = SSS(SNR)
% input_data = SNR matrix
% outpot_data = X_iu matrix;
X_iu = zeros(size(SNR));
    for i = 1:size(SNR, 2)
         row = SNR(:, i) == max(SNR(:, i));
         X_iu(row, i) = 1;
    end
end

