clear all
close all 
clc


%%
time_window = '15s';

data_dir = 'C:\Users\andre\Desktop\Salerno\projects\gpt\scripts\GPT2_word_saliency_fMRI\data';
pred_dir = 'C:\Users\andre\Desktop\Salerno\projects\gpt\scripts\GPT2_word_saliency_fMRI\data\predictors';
out_dir = ['C:\Users\andre\Desktop\Salerno\projects\gpt\scripts\GPT2_word_saliency_fMRI\outputs\',time_window,'\surprisal_analysis'];

%% Load predictors

raw_predictors = load([pred_dir,'\',time_window,'\saliency_design_matrices.mat']);
predictors = zeros(size(raw_predictors.surp,1),size(raw_predictors.surp,2),3); %no duration

predictors(:,:,1) = zscore(raw_predictors.surp);
% predictors(:,:,2) = zscore(raw_predictors.dur); %no duration that has
% high VIF
predictors(:,:,2) = zscore(raw_predictors.freq);
predictors(:,:,3) = zscore(raw_predictors.env);

%% Load atlas labels
atlas_data = readtable([data_dir,'\schaefer_2018\Schaefer2018_1000Parcels_7Networks_order.csv']);
labels = atlas_data.Area;

%% Load subjects data

%FW and BW separeted 
fw_subs = dir([data_dir,'/1000_regions/*FW.csv']);
bw_subs = dir([data_dir,'/1000_regions/*BW.csv']);

fw_n_subs = length(fw_subs);
bw_n_subs = length(bw_subs);

%equal for both BW and FW
first_vol = 709-size(predictors,2)+1; % the narrative starts at t=5
last_vol = first_vol + size(predictors,2)-1;

%FW data
fw_subs_data = zeros(last_vol,size(labels,1),fw_n_subs); %timepoint X regions X subjects
for i = 1:fw_n_subs
 
    %cut the data on the uself window of timepoints
    tmp_data = readmatrix([data_dir,'/1000_regions/',fw_subs(i).name]);
    tmp_data = tmp_data(5:713,:); 
    % insert in the subs_data
    fw_subs_data(:,:,i)= zscore(tmp_data); %this time we have NrofTP X 
    
end

%BW data
bw_subs_data = zeros(last_vol,size(labels,1),bw_n_subs); %timepoint X regions X subjects
for i = 1:bw_n_subs
 
    %cut the data on the uself window of timepoints
    tmp_data = readmatrix([data_dir,'/1000_regions/',bw_subs(i).name]);
    tmp_data = tmp_data(5:713,:); 
    % insert in the subs_data
    bw_subs_data(:,:,i)= zscore(tmp_data); %this time we have NrofTP X 
    
end


%% both subs_id and group should be n_subs*30 long as we will use them in the time-window glm

glm_window = size(predictors,1);

%% Random-effect GLMs

sliding_time_axis = first_vol-1:last_vol-1;

%%


%timepoints X predictors X ROIs X subjects
fw_betas = zeros(size(predictors,2),size(predictors,3),...
    size(fw_subs_data,2),fw_n_subs);
fw_stats = zeros(size(predictors,2),4,... % stats has four outputs 
    size(fw_subs_data,2),fw_n_subs);

%FW Loop on subject to analyze the data of one subject at time
for s=1:fw_n_subs
    tic;
    
    disp(['Subject #',num2str(s)]);
    

        
    %loop on the ROIs
    for i=1:size(fw_subs_data,2)
%         disp(['ROI: ',num2str(i)]) 

        y = squeeze(fw_subs_data(:,i,s)); %select first only the region of interest 
        
        %sliding window for the subject
        t=1;
        for curr_t=sliding_time_axis
            
            %sliding window
            dm_curr = squeeze(predictors(1:glm_window,t,:));
            dm_curr = [dm_curr, ones(size(dm_curr,1),1)];
            y_curr = y(curr_t-glm_window+1:curr_t);


            [betas,~,~,~,stats] = regress(y_curr,dm_curr);

            fw_betas(t,:,i,s) = betas(1:end-1); %excluding the beta of the constant
            fw_stats(t,:,i,s) = stats;

            %this index is useful to access the predictors
            t=t+1;

        end
    end
toc;
end



save([out_dir,'/fw_betas.mat'],'fw_betas');
save([out_dir,'/fw_stats.mat'],'fw_stats');


%%
clc
%timepoints X predictors X ROIs X subjects
bw_betas = zeros(size(predictors,2),size(predictors,3),...
    size(bw_subs_data,2),bw_n_subs);
bw_stats = zeros(size(predictors,2),4,... % stats has four outputs 
    size(bw_subs_data,2),bw_n_subs);

for s=1:fw_n_subs
    tic;
    
    disp(['Subject #',num2str(s)]);
    

        
    %loop on the ROIs
    for i=1:size(bw_subs_data,2)
        %disp(['ROI: ',num2str(i)]) 

        y = squeeze(bw_subs_data(:,i,s)); %select first only the region of interest 
        
        %sliding window for the subject
        t=1;
        for curr_t=sliding_time_axis
            
        %sliding window
        dm_curr = squeeze(predictors(1:glm_window,t,:));
        dm_curr = [dm_curr, ones(size(dm_curr,1),1)];
        y_curr = y(curr_t-glm_window+1:curr_t);


        [betas,~,~,~,stats] = regress(y_curr,dm_curr);
                
        bw_betas(t,:,i,s) = betas(1:end-1);
        bw_stats(t,:,i,s) = stats;
        
        %this index is useful to access the predictors
        t=t+1;

        end
   end
toc;
end



save([out_dir,'/bw_betas.mat'],'bw_betas');
save([out_dir,'/bw_stats.mat'],'bw_stats');
