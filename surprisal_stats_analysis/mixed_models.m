clear all
close all 
clc

%%
time_windows = {'15s', '30s','45s', '60s','75s','90s'};
for i = 1:length(time_windows)
    
    time_window = time_windows{i};
    disp(['Current time window: ' time_window]) 

    data_dir = 'C:\Users\andre\Desktop\Salerno\projects\gpt\scripts\GPT2_word_saliency_fMRI\data';
    pred_dir = 'C:\Users\andre\Desktop\Salerno\projects\gpt\scripts\GPT2_word_saliency_fMRI\data\predictors';
    out_dir = ['C:\Users\andre\Desktop\Salerno\projects\gpt\scripts\GPT2_word_saliency_fMRI\outputs\',time_window,'\surprisal_analysis'];


    %% Load predictors

    predictors = readmatrix([pred_dir,'\',time_window,'\surprisal_full_timecourse_preds.csv']);

    %% Load atlas labels
    atlas_data = readtable([data_dir,'\schaefer_2018\Schaefer2018_1000Parcels_7Networks_order.csv']);
    labels = atlas_data.Area;

    %% Load subjects data

    %first subX_BW and then subX_FW
    subs = dir([data_dir,'/1000_regions/*.csv']);

    n_subs = length(subs);
    first_vol = 5; % the narrative starts at t=5
    last_vol = first_vol + length(predictors(:,1));

    subs_data = [];
    subs_id = [];
    subs_preds = [];

    for i = 1:n_subs

        %cut the data on the uself window of timepoints
        tmp_data = readmatrix([data_dir,'/1000_regions/',subs(i).name]);
        tmp_data = tmp_data(first_vol:last_vol-1,:); 
        % insert in the subs_data
        subs_data = [subs_data; zscore(tmp_data)]; % (N_subs*n_tp) X n_roi (27*709) X 1000

    end


    subs_id = [];
    for i = 1:n_subs/2
        subs_id = [subs_id; i*ones(2*length(first_vol:last_vol-1),1)];    
    end

    %Vector for the subject random effect
    subs_id = nominal(subs_id);

    %FW and BW condition
    group = [zeros(size(predictors,1),1); ones(size(predictors,1),1)];
    group = repmat(group,n_subs/2,1);
    group = nominal(group);

    preds = repmat(predictors,n_subs,1);    


    %% create design matrix

    all_stats = [];
    all_pvalues = [];


    for i=1:size(subs_data,2)

   
        y=subs_data(:,i);
        tbl = table(y,subs_id, ...
            preds(:,1),preds(:,3),preds(:,4),group, ...
            'VariableNames',{'SubjData','SubsName',...
            'Surp',...
            'Freq','Env','Group'});

        model =  fitglme(tbl, 'SubjData ~ 1 + Surp + Freq + Env + Group + Group*Surp + (1|SubsName)');%,'dummyVarCoding','effects');
        all_stats = [all_stats; model.Coefficients];
        all_pvalues = [all_pvalues; [model.Coefficients(6,4).tStat,model.Coefficients(6,6).pValue]];

    end

    %%
    bonf_th = 0.05/length(labels);

    bonf_p = zeros(length(labels),1);
    bonf_p(all_pvalues(:,2)<bonf_th)=1;


    sum(bonf_p)

    output_table = table(atlas_data.ID,labels,...
        all_pvalues(:,1),all_pvalues(:,2),bonf_p,...
        'VariableNames',{'ID','Area', 't stat','p-value','Bonf'});

    writetable(output_table,[out_dir,'/Surp_FW_vs_BW_interaction.csv']);

end

