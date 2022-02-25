# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 09:44:15 2021

@author: Andrea Gerardo Russo, BME, PhD
University of Salerno, Fisciano, Italy

"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
import csv



wdir = 'C:/Users/andre/Desktop/Salerno/projects/gpt/scripts/GPT2_word_saliency_fMRI'
data_dir = 'C:/Users/andre/Desktop/Salerno/projects/gpt/scripts/GPT2_word_saliency_fMRI/data'
out_dir = 'C:/Users/andre/Desktop/Salerno/projects/gpt/scripts/GPT2_word_saliency_fMRI/outputs'
time_window = '15s'

#%%

word_probs = pd.read_csv(os.path.join(wdir,data_dir,time_window,'clean_probabilities_win' + time_window + '.csv'),index_col=0)
word_probs['freq'] = np.zeros((len(word_probs['probability']),1))

#change the index as it was linked to the version with all the words
word_probs.index = list(range(len(word_probs)))


#change the only useful punctuation mark with its na,e
word_probs.loc[1807,'token_predicted'] = 'punto'

import string

punct_indices = [word_probs[word_probs['token_predicted'] == p].index.to_list() for p in string.punctuation]
punct_indices = [tmp for tmp in punct_indices if tmp]
punct_indices = [val for tmp in punct_indices for val in tmp]
punct_indices.sort()




#%%
corpus_frequency = pd.read_csv(os.path.join(data_dir,'narrative_data','wordlist_freq_ittenten16_20210805.csv'))

word_freq = np.zeros((word_probs['token_predicted'].shape[0]))

missing_words = []

for i, tmp_word in enumerate(word_probs['token_predicted']):
    if tmp_word.lower() in corpus_frequency['Item'].to_list():
        idx = corpus_frequency.index[corpus_frequency['Item'] == tmp_word.lower()].to_list()[0]
        word_freq[i] = corpus_frequency['Relative frequency'][idx]       
        
word_probs['freq'] = word_freq


only_word_probs = word_probs.copy()
#remove the remaining punctuation marks

  
only_word_probs.drop(punct_indices, inplace=True)  

    
only_word_probs.to_csv(os.path.join(data_dir,'narrative_data','word_predictors+freq.txt'))

#%% FULL TIME COURSE PREDICTION CREATION 
##############################################################################
#                 DESIGN MATRIX WITH SURPRISAL                               #
##############################################################################


from nilearn.glm.first_level import compute_regressor
import soundfile as sf


# we need to estimate a predictor for each word
#the time can be estimated from the dataframe (stop-start)
predictors = pd.DataFrame([],columns=['Surprisal','Duration','Frequency','Audio'])


#function to load data and return predictor in TR resolution
def create_pred(word_probs,measure):
 
    #time axis with 0.01 resolution
    frame_times = np.arange(0,
                        np.ceil((word_probs['context_stop_time'].iloc[-1]+word_probs['duration'].iloc[-1])),0.01)     
    
    #select the measure
    if measure == 'surprisal':
        values = -np.log10(word_probs['probability'])
        print('AVG Surprisal:', np.average(values))
        print('SD Surprisal:', np.std(values))
        

    if measure == 'duration':
        values = np.ones((1,len(word_probs['duration'])))
    if measure == 'frequency':
        word_probs[word_probs['freq']==0]=1
        values = -np.log10(word_probs['freq'])
    if measure == 'audio':
        data, samplerate = sf.read(os.path.join(data_dir,'narrative_data','gianna.wav'))         
       
        values = []
        for i,cond in enumerate(np.vstack((word_probs['context_stop_time'],
                                    word_probs['duration'])).T):
            on = np.where(np.isclose(frame_times, cond[0]))[0][0] #returns a tuple
            off = np.where(np.isclose(frame_times, cond[0]+cond[1]))[0][0] #returns a tuple
            tmp_vec = data[round(on/100*samplerate):round(off/100*samplerate)]
            values.append(np.sqrt(np.mean(tmp_vec*tmp_vec)))
    
    
    #timepoints of interest    
    TR_time = np.arange(0,frame_times[-1] , 1)

    #stack onset, duration and probability before the hrf convolution
    prob_exp_condition = np.vstack((word_probs['context_stop_time'],
                                    word_probs['duration'],
                                    values)).T
        

    #compute the regressor with the SPM HRF
    signal,name = compute_regressor(
            prob_exp_condition.T, 'spm', TR_time, con_id=measure)
    
    
    return signal


predictors['Surprisal'] = np.squeeze(create_pred(only_word_probs,'surprisal'))
predictors['Duration'] = np.squeeze(create_pred(only_word_probs,'duration'))
predictors['Frequency'] = np.squeeze(create_pred(only_word_probs,'frequency'))
predictors['Audio'] = np.squeeze(create_pred(only_word_probs,'audio'))

predictors.to_csv(os.path.join(data_dir,'predictors',time_window,'surprisal_full_timecourse_preds.csv'), index=False)

#%% FULL TIME COURSE PREDICTION CREATION 
##############################################################################
#                 DESIGN MATRIX WITH AVERAGE UNNORMALIZED SALIENCE           #
##############################################################################

# # we need to estimate a predictor for each word
# #the time can be estimated from the dataframe (stop-start)
# unnormalized_predictors = pd.DataFrame([],columns=['Average_salience','Duration','Frequency','Audio'])


# unnormalized_clean_salience = pickle.load(open(os.path.join(wdir,'output','window30s','unnormalized_clean_salience_win30s.pkl'),'rb'))
# unnormalized_clean_past_tokens = pickle.load(open(os.path.join(wdir,'output','window30s','unnormalized_clean_past_token_win30s.pkl'),'rb'))

# #these files contain punctuations marks that has to be removed 
# only_words_clean_salience = [unnormalized_clean_salience[idx] for idx in range(len(unnormalized_clean_salience)) if idx not in punct_indices]
# only_words_clean_past_tokens = [unnormalized_clean_past_tokens[idx] for idx in range(len(unnormalized_clean_salience)) if idx not in punct_indices]


# #now we can estimate the AUC for each vector. This value could be assigned to 
# #each word to have a timeseries
# word_auc = [np.trapz(np.squeeze(salience_vec[-1]))
#             for salience_vec in only_words_clean_salience]


# #actually we can estimate only the AUC for the curve of the unnormalized clean
# #salience and then use the other predictors estimated before

 
# #time axis with 0.01 resolution
# frame_times = np.arange(0,np.ceil((only_word_probs['context_stop_time'].iloc[-1]+only_word_probs['duration'].iloc[-1])),0.01)     
 
# #average clean salience   
# values = np.array(word_auc)

# #stack onset, duration and probability before the hrf convolution
# prob_exp_condition = np.vstack((only_word_probs['context_stop_time'],
#                                 only_word_probs['duration'],
#                                 values)).T

# TR_time = np.arange(0,frame_times[-1] , 1)
        
# #compute the regressor with the SPM HRF
# signal,name = compute_regressor(
#         prob_exp_condition.T, 'spm', frame_times, con_id='average salience')
   


# unnormalized_predictors['Average_salience'] = np.squeeze(signal)
# unnormalized_predictors['Duration'] = predictors['Duration']
# unnormalized_predictors['Frequency'] = predictors['Frequency']
# unnormalized_predictors['Audio'] = predictors['Audio']

# predictors.to_csv(os.path.join(data_dir,'predictors',time_window,'average_salience_full_timecourse_preds.csv'),index=False)



#%% sliding window salience predictor

##############################################################################
#            SET OF DESIGN MATRICES FOR THE SLIDING WINDOW ANALYSIS          #
##############################################################################

#frequency of all words, including the multi-token words
all_full_words_freq = pd.read_csv(os.path.join(data_dir,'narrative_data','all_words_freq.csv'))
#load audio
audiodata, samplerate = sf.read(os.path.join(data_dir,'narrative_data','gianna.wav'))         


clean_salience = pickle.load(open(os.path.join(data_dir,time_window,'clean_salience_win'+ time_window +'.pkl'),'rb'))
clean_past_tokens = pickle.load(open(os.path.join(data_dir,time_window,'clean_past_token_win'+ time_window +'.pkl'),'rb'))

#these files contain punctuations marks that has to be removed 
only_words_clean_salience = [clean_salience[idx] 
                             for idx in range(len(clean_salience)) if idx not in punct_indices]
only_words_clean_past_tokens = [clean_past_tokens[idx] 
                                for idx in range(len(clean_salience)) if idx not in punct_indices]

pickle.dump(only_words_clean_past_tokens,
            open(os.path.join(out_dir,time_window,'only_words_clean_past_tokens.pkl'),'wb'))

#we need to treat the multi-tokens word. we can sum the salience associated to each token 
all_toks_timestamps = pd.read_csv(os.path.join(data_dir,'narrative_data','all_tokens+timestamp.txt'),delimiter='\t')

#load the text with the tokens to remove 
tok2remove = []
with open(os.path.join(data_dir,'narrative_data','token2remove.txt')) as csvfile:
   csvreader = csv.reader(csvfile, delimiter=',')
   for row in csvreader:
    tok2remove.append(row)

#we need also the clean version of the token2remove
tok2remove_flat = [int(item) for tmp in tok2remove for item in tmp]



########OUTPUTS#####################
#create the list to store the 30TR predictors for each word and measure
salience_preds = []
durations_preds = []
freq_preds = []
rms_audio = []
all_outputs = [salience_preds,
               durations_preds,
               freq_preds,
               rms_audio]

unconvolved_measures = []

###################################


all_past_words = []

#loop on the salience input data
for j,curr_word in enumerate(only_words_clean_salience):
    
    
    #indices of the context words
    word_context_indices = curr_word[-1]
    #past tokens
    past_tokens = only_words_clean_past_tokens[j][1]
    
    #tipically only the first woord does not have context
    if word_context_indices.size != 0:
        
        #salience numpy array
        sal_vec = curr_word[-2][0]
        

        ##############################################
        # We opted to average the salience realtive to the tokens composing the
        # the multi-tokens words. This is the only operation that we perform on 
        # the data. In fact, for the other predictors we have a specific duration,
        # frequency and RMS of the sound associated to each word (included the 
        # multi-tokens word)
        
        #multi-tokens words: average salience, duration of the word, frequency
        # of the word and RMS of the sound
        ######################################
        
        new_sal_vec = [] 
        onsets = []
        durations = []
        frequency = []
        rms_values = []

        past_words = []
        #loop on the indices of the words that creates the context of a specific word
        for i,idx in enumerate(word_context_indices):
            
            
            
            #idx corresponds to the ind of the word, i to the position in the array
            #that should be used to access the corresponding salience
            
            
            #first check. The index is the indices to remove?
            if idx in tok2remove_flat: 
            
                #Loop on the non-flat list (list of list)
                for element in tok2remove:
                    
                    #if the first element of the list of successive indices to remove is in the context
                    #if idx is not the first of a set of indices to remove we skip this part                    
                    if int(element[0])==idx:
                        
                        #get the salience values of the tokens to remove/merge
                        set_sal_values = np.array([sal_vec[k] for k in range(i,i+len(element))])
                        #get the mean salience of the tokens
                        new_sal_vec.append(np.mean(set_sal_values))
                        #join the tokens to have the real word
                        word = ''.join([past_tokens[k] for k in range(i,i+len(element))])
                        
                        #the word più is weirdly read as "pià". This is an hard fix
                        if word == 'pià':
                            word = 'più'
                            
                        if word == 'new_paragraphnew_paragraph':
                            word = '.'
                            
                        if ' ' in word:
                            word = word.replace(' ','')
                            
                        if 'Ġ' in word:
                            word = word.replace('Ġ','')

                        if 'Ã²' in word:
                            word = word.replace('Ã²','ò')
                            
                        if 'Ã¬' in word:
                            word = word.replace('Ã¬','ì')
                            
                        if 'Ã©' in word:
                            word = word.replace('Ã©','è')
                            
                        if 'Ã¹' in word:
                            word = word.replace('Ã¹','ù')
                            
                        if 'Ã¨' in word:
                            word = word.replace('Ã¨','è')
                            
                            
                            
                        ######################################################
                        # tokens have the same timestamps, duration and frequency
                        # of the real word. Thus we need only the timestamp,
                        # the duration and the frequency associated to the 
                        # first token 
                        ######################################################
                        
                        #past words
                        past_words.append(word)                       
                        #onset
                        onsets.append(all_toks_timestamps['time'].iloc[idx])
                        #duration
                        durations.append(all_toks_timestamps['duration'].iloc[idx])
                        #frequency
                        frequency.append(all_full_words_freq['Relative frequency'][all_full_words_freq.index[all_full_words_freq['Item'] == word.lower()].to_list()[0]])                         
                        #rms audio
                        on = all_toks_timestamps['time'].iloc[idx]
                        off = all_toks_timestamps['time'].iloc[idx] + all_toks_timestamps['duration'].iloc[idx]
                        audio_vec = audiodata[round(on/100*samplerate):round(off/100*samplerate)]
                        rms_values.append(np.sqrt(np.mean(audio_vec*audio_vec)))
                    
                        # once we have obtained all the information associated to
                        # the first token of the multi-tokens words we can break the
                        # the loop
                        break
                
            else:
                #if idx is not in the indices to remove we keep the original value
                word = past_tokens[i]
                new_sal_vec.append(sal_vec[i])
                onsets.append(all_toks_timestamps['time'].iloc[idx])
                durations.append(all_toks_timestamps['duration'].iloc[idx])
                
                #the word più is weirdly read as "pià". This is an hard fix
                if word == 'pià':
                    word = 'più'
                    
                if word == 'new_paragraphnew_paragraph':
                    word = '.'
                    
                if ' ' in word:
                    word = word.replace(' ','')
                    
                if 'Ġ' in word:
                    word = word.replace('Ġ','')

                if 'Ã²' in word:
                    word = word.replace('Ã²','ò')
                    
                if 'Ã¬' in word:
                    word = word.replace('Ã¬','ì')
                    
                if 'Ã©' in word:
                    word = word.replace('Ã©','è')
                    
                if 'Ã¹' in word:
                    word = word.replace('Ã¹','ù')
                    
                if 'Ã¨' in word:
                    word = word.replace('Ã¨','è')
 
                
                #past words
                past_words.append(word)   
                frequency.append(all_full_words_freq['Relative frequency'][all_full_words_freq.index[all_full_words_freq['Item'] == word.lower()].to_list()[0]])
                on = all_toks_timestamps['time'].iloc[idx]
                off = all_toks_timestamps['time'].iloc[idx] + all_toks_timestamps['duration'].iloc[idx]
                audio_vec = audiodata[round(on/100*samplerate):round(off/100*samplerate)]
                rms_values.append(np.sqrt(np.mean(audio_vec*audio_vec)))
              
                
       
          
        
        # once obtained all the values associated to each token in the context
        # of the word we can apply the convolution with the HRF
        all_measures = [new_sal_vec, np.ones((1,len(new_sal_vec))), frequency, rms_values]
        
        #it is good also to save the unconcolved data
        unconvolved_measures.append(all_measures)
        
        all_past_words.append(past_words)
        

        #loop on the measure and create the fMRI predictor
        for z,measure in enumerate(all_measures):
            
            output = all_outputs[z]
    
            prob_exp_condition = np.vstack((onsets,durations,np.array(measure))).T
            
            # if j>=74:
            #     #frame_times = np.arange(np.floor(curr_word[1]),np.ceil(curr_word[2]),0.01)
            #     TR_time = np.arange(0,30,1)
                
            # else:            
            #     #frame_times = np.arange(np.floor(curr_word[1]),np.ceil(curr_word[2]),0.01)
                
            TR_time = np.arange(round(curr_word[2])-int(time_window[:-1]),round(curr_word[2]),1)
            
            #compute the regressor with the SPM HRF
            signal,name = compute_regressor(
                    prob_exp_condition.T, 'spm', TR_time, con_id='pred')
            
            output.append([curr_word[0],curr_word[2],signal])

        
    
pickle.dump(output, open(os.path.join(data_dir,'predictors',time_window,'saliency_design_matrices.pkl'),'wb'))

import scipy.io
scipy.io.savemat(os.path.join(data_dir,'predictors',time_window,'word_level_saliency_design_matrices.mat'), 
                 mdict={'sliding_window_dm': output})

pickle.dump(unconvolved_measures, open(os.path.join(data_dir,'predictors',time_window,'saliency_unconvolved_design_matrices.pkl'),'wb')) 

pickle.dump(all_past_words, open(os.path.join(data_dir,'predictors',time_window,'past_words.pkl'),'wb')) 






#%% Estimate average number of past tokens and past words

avg_words = np.average(np.array([len(tmp) for tmp in all_past_words]))
avg_tokens = np.average(np.array([len(tmp[1]) for tmp in clean_past_tokens]))

std_words = np.std(np.array([len(tmp) for tmp in all_past_words]))
std_tokens = np.std(np.array([len(tmp[1]) for tmp in clean_past_tokens]))

print(avg_words,avg_tokens)
print(std_words, std_tokens)    




#%% Salience vectors need to be downsampled to the timepoint axis, i.e. from 
# words to timepoints
from scipy.interpolate import interp1d

for idx,tmp in enumerate(only_words_clean_salience[1:]): #skip the first word (no past)
    if tmp[1]!=0:
        first_useful_tp = idx
        break
    
last_useful_tp = np.ceil((word_probs['context_stop_time'].iloc[-1]+word_probs['duration'].iloc[-1]))

nr_of_useful_tp = int((last_useful_tp+1)-(round(output[first_useful_tp][1])))
print(nr_of_useful_tp)

TR_time = np.arange(0, nr_of_useful_tp, 1)

labels = ['surp','dur','freq','env']

downsampled_output = []

for n,measure in enumerate(all_measures):
    
    full_output = all_outputs[n]
    
    data2downsample = np.empty((len(full_output[first_useful_tp:]),int(time_window[:-1])))
    
    for i,tmp in enumerate(full_output[first_useful_tp:]):
        
        data2downsample[i,:]=tmp[2].reshape(1,-1)
    
    word_vector = np.arange(0,data2downsample.shape[0],1)
    #linear downsample of the vector values
    linear_downsample = interp1d(word_vector, data2downsample.T, kind='linear')
    # Specify the number of time points of interest
    downsampled_data= linear_downsample(TR_time)
    
    downsampled_output.append(downsampled_data)
        

    
    
scipy.io.savemat(os.path.join(data_dir,'predictors',time_window,'saliency_design_matrices.mat'), 
                  mdict={labels[0]: downsampled_output[0],
                         labels[1]: downsampled_output[1],
                         labels[2]: downsampled_output[2],
                         labels[3]: downsampled_output[3]})    