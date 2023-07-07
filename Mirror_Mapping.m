% Name: Mirror function
% Useage: To mirror M users to 50 users
% do not map the target UE, mapping the remaining users
% Considering R
%% Mirror input training data
AP_num = 5;
UE_num = 49; % except the target UE
batch_num = 100;
index = 10; % input users number kind
M_set = 5:5:50;
%
for i = 1:index
    input_names = arrayfun(@(i)['input_batch' num2str(i) '.csv'], (i-1)*batch_num+1:i*batch_num, 'un',0);
    output_names = arrayfun(@(i)['mirror_input_batch' num2str(i) '.csv'], (i-1)*batch_num+1:i*batch_num, 'un',0);
    M = M_set(i) - 1;
    for j = 1:batch_num  
        user_set = csvread(input_names{j});
        % choose target UE data
        Target = user_set(1:(AP_num+1), :); % keep target unchangable
        user_set(1:(AP_num+1), :) = []; % remove target from condition
        % split R
        quotient = floor(UE_num/M); % 
        reminder = mod(UE_num, M); %
        if reminder == 0
            user_set((AP_num + 1)*(1:M), :) = user_set((AP_num + 1)*(1:M), :)/(quotient); % split R by quotient times
        else
            user_set((AP_num + 1)*(1:reminder), :) = user_set((AP_num + 1)*(1:reminder), :)/(quotient+1); % split R by (quotient+1) times
            user_set((AP_num + 1)*(reminder+1:M), :) = user_set((AP_num + 1)*(reminder+1:M), :)/quotient; % split R by quotient times
        end       
        % mirror users
        if quotient == 1 % M > 15
            mirrored_set = [user_set; user_set(1:(AP_num + 1)*reminder, :)];
        else % M <= 15
            mirrored_set = user_set; 
            for k = 1:quotient-1
                mirrored_set = [mirrored_set; user_set];      
            end 
            mirrored_set = [mirrored_set; user_set(1:(AP_num + 1)*reminder, :)];
        end
        % Add target into first position
        mirrored_set = [Target; mirrored_set];
        % save 
        csvwrite(output_names{j}, mirrored_set);  % save input training data   
    end    
    fprintf('Finished Mirroring User number of %d \n', i*batch_num);
end
%% Mirror input test data 
AP_num = 5;
UE_num = 49; % maximum UE number
batch_num = 1;
M_set = 5:5:50;
index = 10; % input users number from 5 to 29
for i = 1:index
    input_names = arrayfun(@(i)['input_batch' num2str(i) '.csv'], (i-1)*batch_num + 1001:i*batch_num + 1000, 'un',0);
    output_names = arrayfun(@(i)['mirror_input_batch' num2str(i) '.csv'], (i-1)*batch_num + 1001:i*batch_num + 1000, 'un',0);
    M = M_set(i) - 1; %
    for j = 1:batch_num  
        user_set = csvread(input_names{j});
        % choose target UE data
        Target = user_set(1:(AP_num+1), :);
        user_set(1:(AP_num+1), :) = []; % remove target from condition
        % split R
        quotient = floor(UE_num/M); % 
        reminder = mod(UE_num, M); %
        if reminder == 0
            user_set((AP_num + 1)*(1:M), :) = user_set((AP_num + 1)*(1:M), :)/(quotient); % split R by quotient times
        else
            user_set((AP_num + 1)*(1:reminder), :) = user_set((AP_num + 1)*(1:reminder), :)/(quotient+1); % split R by (quotient+1) times
            user_set((AP_num + 1)*(reminder+1:M), :) = user_set((AP_num + 1)*(reminder+1:M), :)/quotient; % split R by quotient times
        end       
        % mirror users
        if quotient == 1 % M > 15
            mirrored_set = [user_set; user_set(1:(AP_num + 1)*reminder, :)];
        else % M <= 15
            mirrored_set = user_set; 
            for k = 1:quotient-1
                mirrored_set = [mirrored_set; user_set];      
            end 
            mirrored_set = [mirrored_set; user_set(1:(AP_num + 1)*reminder, :)];
        end
        % Add target into first position
        mirrored_set = [Target; mirrored_set];
        % save 
        csvwrite(output_names{j}, mirrored_set);  % save input training data   
    end    
    fprintf('Finished Mirroring User number of %d \n', i*batch_num);
end

%% Mirror output training data 
AP_num = 5;
UE_num = 49; % maximum UE number
batch_num = 100;
M_set = 5:5:50;
index = 10; % input users number from 5 to 29
for i = 1:index
    input_names = arrayfun(@(i)['output_batch' num2str(i) '.csv'], (i-1)*batch_num+1:i*batch_num, 'un',0);
    output_names = arrayfun(@(i)['mirror_output_batch' num2str(i) '.csv'], (i-1)*batch_num+1:i*batch_num, 'un',0);
    M = M_set(i) - 1; % number of input users from 5 to 30 <------ CHANEG THIS PARAMETER
    for j = 1:batch_num  
        user_set = csvread(input_names{j});
        % choose target UE data
        Target = user_set(1:AP_num, :);
        user_set(1:AP_num, :) = []; % remove target from condition

        quotient = floor(UE_num/M); % 
        reminder = mod(UE_num, M); %      
        % mirror users
        if quotient == 1 % M > 15
            mirrored_set = [user_set; user_set(1:AP_num*reminder, :)];
        else % M <= 15
            mirrored_set = user_set; 
            for k = 1:quotient-1
                mirrored_set = [mirrored_set; user_set];      
            end 
            mirrored_set = [mirrored_set; user_set(1:AP_num*reminder, :)];
        end
        % Add target into first position
        mirrored_set = [Target; mirrored_set];
        % save 
        csvwrite(output_names{j}, mirrored_set);  % save input training data   
    end    
    fprintf('Finished Mirroring User number of %d \n', i*batch_num);
end

%% Mirror output test data 
AP_num = 5;
UE_num = 49; 
batch_num = 1;
batch_size = 256;
M_set = 5:5:50;
index = 10; % input users number from 5 to 29
for i = 1:index
    input_names = arrayfun(@(i)['output_batch' num2str(i) '.csv'], (i-1)*batch_num + 1001:i*batch_num + 1000, 'un',0);
    output_names = arrayfun(@(i)['mirror_output_batch' num2str(i) '.csv'], (i-1)*batch_num + 1001:i*batch_num + 1000, 'un',0);
    M = M_set(i) - 1; % number of input users from 5 to 30 <------ CHANEG THIS PARAMETER
    for j = 1:batch_num  
        user_set = csvread(input_names{j});
        % choose target UE data
        Target = user_set(1:AP_num, :);
        user_set(1:AP_num, :) = []; % remove target from condition

        quotient = floor(UE_num/M); % 
        reminder = mod(UE_num, M); %      
        % mirror users
        if quotient == 1 % M > 15
            mirrored_set = [user_set; user_set(1:AP_num*reminder, :)];
        else % M <= 15
            mirrored_set = user_set; 
            for k = 1:quotient-1
                mirrored_set = [mirrored_set; user_set];      
            end 
            mirrored_set = [mirrored_set; user_set(1:AP_num*reminder, :)];
        end
        % Add target into first position
        mirrored_set = [Target; mirrored_set];
        % save 
        csvwrite(output_names{j}, mirrored_set);  % save input training data   
    end    
    fprintf('Finished Mirroring User number of %d \n', i*batch_num);
end




