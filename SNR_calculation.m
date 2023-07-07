function SNR = SNR_calculation(env, AP, UE, mode)
switch mode
    case 'LiFi'
        %% parameters
        X_length = env.X_length;
        Y_length = env.Y_length;
        Z_height = env.Z_height;
        P_mod = env.P_mod; % Modulated power
        N_0 = env.N_0 ; % Noise power spectral density: Watt/Hz
        Phi = env.Phi; % semiangle: in radian, 60 degree
        % FOV = env.FOV; % FOV: radian
        FOV = 80*pi/180; % FOV: radian
        n = env.n; % Reflective index of concentrator
        R_pd = env.R_pd; % PD responsivity
        % k = 0.5; %  optical to electric conversion coefficient
        A = env.A; % Detector area: m^2
        m = env.m; % Lambertian order
        B = env.B;
        Ka = 0.8;
        %% LOS
        d_LOS = pdist2(AP, UE); 
        cos_phi = Z_height/d_LOS;
        if abs(acosd(cos_phi)/180*pi) <= Phi
           H_LOS = (m+1)*A*n^2*Z_height^(m+1) / (2*pi*(sin(FOV))^2*(d_LOS^(m+3)));  % correct          
        else
           H_LOS = 0;
        end      
        %% NLOS
        H_NLOS = Capacity_NLOS(AP(1), AP(2), Z_height, UE(1), UE(2), X_length, Y_length, Z_height); % call sub-function
        % H_NLORS = 0;
        %%
        % SNR = (H_LOS + H_NLOS); %%%
        SNR = (R_pd*P_mod*Ka*(H_LOS + H_NLOS))^2/N_0/B; 
%         if SNR == 0
%             fprintf('H_LOS = %d, H_NLOS = %d \n', H_LOS, H_NLOS);
%         end
    case 'WiFi'
        d_LOS = pdist2(AP, UE);
        radiation_angle = acos(0.5/d_LOS); % radian unit
        P_WiFi = 10^(env.P_WiFi/10)/1000; % 20 dBm, convert to watts: 0.1 W
        B_WiFi = env.B_WiFi; % 20 MHz
        N_WiFi = 10^(env.N_WiFi/10)/1000 ; % convert to W/Hz
        f = env.f_WiFi; % carrier frequency, 2.4 GHz
        % 20 dB loss for concreate wall attenuation
        L_FS = 20*log10(d_LOS) + 20*log10(f) + 20 - 147.5; % free space loss, unit: dB        
        if d_LOS <= env.d_BP
            K = 1; % Ricean K-factor 
            X = 3; % the shadow fading before breakpoint, unit: dB        
            LargeScaleFading = L_FS + X;                  
        else
            K = 0;
            X = 5; % the shadow fading after breakpoint, unit: dB                 
            LargeScaleFading = L_FS + 35*log10(d_LOS/env.d_BP) + X;
        end
        H_WiFi = sqrt(K/(K+1))*(cos(radiation_angle) + 1j*sin(radiation_angle)) + sqrt(1/(K+1))*(1/sqrt(2)*randn(1) + 1j/sqrt(2)*randn(1)); % WiFi channel transfer function                      
        channel =  (abs(H_WiFi))^2 *10.^ ( -LargeScaleFading / 10 ); % WiFi channel gain   
        SNR = P_WiFi*channel/(N_WiFi*B_WiFi); % range of (1000, 100000000)
end
end

