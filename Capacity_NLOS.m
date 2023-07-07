function H_NLOS = Capacity_NLOS(x_AP, y_AP, z_AP, x_UE, y_UE, X_length, Y_length, Z_height)
%% input x-y-z coordinat of APs to return channel gain H of NLOS
Phi = pi/3; % semiangle: radian
FOV = 80/180*pi; % FOV: radian
m = -1/(log2(cos(Phi))); % Lambertian order
A = 0.0001; % Detector area: m^2
n = 1.5; % Reflective index of concentrator
UE = [x_UE, y_UE, 0]; %UE Location
AP = [x_AP, y_AP, z_AP];
rho = 0.8; % reflection coefficient of room walls
% X_length=5; % m, room size
% Y_length=5; % m, room size
% Z_height=3; % m, room size
step = 0.1;   % <--- change from 0.2 to 0.1
Nx = X_length/step; Ny = Y_length/step; Nz = Z_height/step; % number of grid in each surface
X = linspace(0, X_length, Nx+1);
Y = linspace(0, Y_length, Ny+1);
Z = linspace(0, Z_height, Nz+1);
dA = 0.01; % reflective area of wall
H_NLOS_W1 = zeros(length(Y)-1,length(Z)-1);
H_NLOS_W2 = zeros(length(X)-1,length(Z)-1);
H_NLOS_W3 = zeros(length(Y)-1,length(Z)-1);
H_NLOS_W4 = zeros(length(X)-1,length(Z)-1);
for i = 1:1:length(X)-1
    for j = 1:1:length(Z)-1
        %% H11_NLOS of Wall 1 (Left), 1st reflection channel gain between AP1 and UE
        Refl_point_W1 = [0, (Y(i)+Y(i+1))/2, (Z(j)+Z(j+1))/2];
        % d1=pdist2(AP,Refl_point_W1); 
        % d2=pdist2(UE,Refl_point_W1); % pdist2 function is time-costing
        d1 = sqrt((AP(1) - Refl_point_W1(1))^2 + (AP(2) - Refl_point_W1(2))^2 + (AP(3) - Refl_point_W1(3))^2); 
        d2 = sqrt((UE(1) - Refl_point_W1(1))^2 + (UE(2) - Refl_point_W1(2))^2 + (UE(3) - Refl_point_W1(3))^2); % distance calculation in 3-D space
        cos_phi = abs(Refl_point_W1(3) - AP(3))/d1;
        cos_alpha = abs(AP(1) - Refl_point_W1(1))/d1;
        cos_beta = abs(UE(1) - Refl_point_W1(1))/d2;
        cos_psi = abs(UE(3) - Refl_point_W1(3))/d2; % /sai/
          if abs(acosd(cos_phi)/180*pi) <= Phi
             if abs(acosd(cos_psi)/180*pi) <= FOV
                H_NLOS_W1(i,j)=(m+1)*A*rho*dA*cos_phi^m*cos_alpha*cos_beta*cos_psi*n^2/(2*pi^2*d1^2*d2^2*(sin(FOV))^2);
             else
                H_NLOS_W1(i,j)=0;      
             end
          else
                H_NLOS_W1(i,j)=0;  
          end
        %% H11_NLOS of Wall 2 (Front)
        Refl_point_W2=[(X(i)+X(i+1))/2, 0, (Z(j)+Z(j+1))/2];
        % d1=pdist2(AP,Refl_point_W2); 
        % d2=pdist2(UE,Refl_point_W2); 
        d1 = sqrt((AP(1)-Refl_point_W2(1))^2 + (AP(2)-Refl_point_W2(2))^2 + (AP(3)-Refl_point_W2(3))^2); 
        d2 = sqrt((UE(1)-Refl_point_W2(1))^2 + (UE(2)-Refl_point_W2(2))^2 + (UE(3)-Refl_point_W2(3))^2); % distance calculation in 3-D space
        cos_phi = abs(Refl_point_W2(3)-AP(3))/d1;
        cos_alpha = abs(AP(1)-Refl_point_W2(1))/d1;
        cos_beta = abs(UE(1)-Refl_point_W2(1))/d2;
        cos_psi = abs(UE(3)-Refl_point_W2(3))/d2; % /sai/
          if abs(acosd(cos_phi)/180*pi) <= Phi
             if abs(acosd(cos_psi)/180*pi) <= FOV
                H_NLOS_W2(i,j) = (m+1)*A*rho*dA*cos_phi^m*cos_alpha*cos_beta*cos_psi*n^2/(2*pi^2*d1^2*d2^2*(sin(FOV))^2);
             else
                H_NLOS_W2(i,j) = 0;      
             end
          else
                H_NLOS_W2(i,j) = 0;  
          end
        %% H11_NLOS of Wall 3 (Right)
        Refl_point_W3 = [X_length, (Y(i)+Y(i+1))/2, (Z(j)+Z(j+1))/2];
        % d1=pdist2(AP,Refl_point_W3); 
        % d2=pdist2(UE,Refl_point_W3); 
        d1 = sqrt((AP(1)-Refl_point_W3(1))^2 + (AP(2)-Refl_point_W3(2))^2 + (AP(3)-Refl_point_W3(3))^2); 
        d2 = sqrt((UE(1)-Refl_point_W3(1))^2 + (UE(2)-Refl_point_W3(2))^2 + (UE(3)-Refl_point_W3(3))^2); % distance calculation in 3-D space
        cos_phi = abs(Refl_point_W3(3)-AP(3))/d1;
        cos_alpha = abs(AP(1)-Refl_point_W3(1))/d1;
        cos_beta = abs(UE(1)-Refl_point_W3(1))/d2;
        cos_psi = abs(UE(3)-Refl_point_W3(3))/d2; % /sai/
          if abs(acosd(cos_phi)/180*pi) <= Phi
             if abs(acosd(cos_psi)/180*pi) <= FOV
                H_NLOS_W3(i,j) = (m+1)*A*rho*dA*cos_phi^m*cos_alpha*cos_beta*cos_psi*n^2/(2*pi^2*d1^2*d2^2*(sin(FOV))^2);
             else
                H_NLOS_W3(i,j)=0;  
             end
          else
                H_NLOS_W3(i,j)=0;  
          end
        %% H11_NLOS of Wall 4 (Back)
        Refl_point_W4=[(X(i)+X(i+1))/2, Y_length, (Z(j)+Z(j+1))/2];
        % d1=pdist2(AP,Refl_point_W4); 
        % d2=pdist2(UE,Refl_point_W4);
        d1 = sqrt((AP(1)-Refl_point_W4(1))^2 + (AP(2)-Refl_point_W4(2))^2 + (AP(3)-Refl_point_W4(3))^2); 
        d2 = sqrt((UE(1)-Refl_point_W4(1))^2 + (UE(2)-Refl_point_W4(2))^2 + (UE(3)-Refl_point_W4(3))^2); % distance calculation in 3-D space
        cos_phi = abs(Refl_point_W4(3)-AP(3))/d1;
        cos_alpha = abs(AP(1)-Refl_point_W4(1))/d1;
        cos_beta = abs(UE(1)-Refl_point_W4(1))/d2;
        cos_psi = abs(UE(3)-Refl_point_W4(3))/d2; % /sai/
          if abs(acosd(cos_phi)/180*pi)<= Phi
             if abs(acosd(cos_psi)/180*pi)<= FOV
                H_NLOS_W4(i,j)=(m+1)*A*rho*dA*cos_phi^m*cos_alpha*cos_beta*cos_psi*n^2/(2*pi^2*d1^2*d2^2*(sin(FOV))^2);
             else
                H_NLOS_W4(i,j)=0;    
             end
          else
                H_NLOS_W4(i,j)=0;  
          end
    end
end
H_NLOS = H_NLOS_W1 + H_NLOS_W2 + H_NLOS_W3 + H_NLOS_W4; % matrix data
H_NLOS = sum(sum(H_NLOS));
end