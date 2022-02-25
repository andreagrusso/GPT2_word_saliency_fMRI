clear all
close all 
clc

%%

time_window = '60s';

data_dir = 'C:\Users\andre\Desktop\Salerno\projects\gpt\scripts\GPT2_word_saliency_fMRI\data';
pred_dir = 'C:\Users\andre\Desktop\Salerno\projects\gpt\scripts\GPT2_word_saliency_fMRI\data\predictors';
out_dir = ['C:\Users\andre\Desktop\Salerno\projects\gpt\scripts\GPT2_word_saliency_fMRI\outputs\',time_window,'\surprisal_analysis'];

%% This script uses the timeseries betas of each subject and perform a GLM
% analysis with subjects as random factor and a group variable that is the
% the speech direction (BW, FW). We are interested in the contrast FW vs BW

%% Load atlas labels
atlas_data = readtable([data_dir,'\schaefer_2018\Schaefer2018_1000Parcels_7Networks_order.csv']);
labels = atlas_data.Area;

%% Load subjects data

%FW and BW separeted 
fw_subs_mat = load([out_dir,'/fw_betas.mat']).fw_betas;
bw_subs_mat = load([out_dir,'/bw_betas.mat']).bw_betas;

fw_n_subs = size(fw_subs_mat,4);
bw_n_subs = size(bw_subs_mat,4);

n_rois = size(fw_subs_mat,3);


fw_subs_data = [];
bw_subs_data = [];

for i=1:fw_n_subs
    
    fw_subs_data = [fw_subs_data;fw_subs_mat(:,1,:,i)];
    bw_subs_data = [bw_subs_data;bw_subs_mat(:,1,:,i)];

end


fw_subs_data = squeeze(fw_subs_data);
bw_subs_data = squeeze(bw_subs_data);


fw_group = ones(size(fw_subs_data,1),1);
bw_group = zeros(size(bw_subs_data,1),1);
group = nominal([bw_group;fw_group]);%BW (0) and FW (1)

subs_id = [];
for i = 1:bw_n_subs
    subs_id = [subs_id; i*ones(size(fw_subs_mat,1),1)];   
end
subs_id = nominal([subs_id;subs_id]); %BW and FW concatenated


%%
results = []; %t-Stat and p-value for each region
all_coefficients = []; %just to complete the output of the model

for i = 1:n_rois
    
    y_roi = [bw_subs_data(:,i);fw_subs_data(:,i)];

    tbl = table(y_roi,subs_id, group,...
        'VariableNames',{'SubjData','SubsName',...
        'Group'});

    model = fitglme(tbl, 'SubjData ~ 1 + Group + (1|SubsName)');%,'dummyVarCoding','effects');
    all_coefficients =  [all_coefficients; model.Coefficients];
    results = [results; model.Coefficients(2,4).tStat, model.Coefficients(2,6).pValue];

    
end

%%

bonf_th = 0.05/length(labels);

bonf_p = zeros(length(labels),1);
bonf_p(results(:,2)<bonf_th)=1;
fdr_p = fdr_bh(results(:,2));


sum(bonf_p)
sum(fdr_p)

output_table = table(atlas_data.ID,labels,...
    results(:,1),results(:,2),bonf_p,fdr_p,...
    'VariableNames',{'ID','Area', 't stat','p-value','Bonf','FDR'});

writetable(output_table,[out_dir,'/Saliency_FW_vs_BW.csv']);