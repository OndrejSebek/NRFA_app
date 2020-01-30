import streamlit as st

import app_utils as au


# ''' ____________________________ SIDEBAR _______________________________ '''

# station
sub_stations = st.sidebar.selectbox('category', 
                                    ('all', 'chalk sites', 'densely monitored',
                                     'sparsely monitored', 'small catchments',
                                     'top of catchments', 'lower in catchments',
                                     'natural sites', 'unnatural sites',
                                     '__test_sites__', '_PRES_'))

if sub_stations == 'chalk sites':
    st_id = st.sidebar.selectbox('station', 
                                 ('38029', '38029'))
elif sub_stations == 'densely monitored':
    st_id = st.sidebar.selectbox('station', 
                                 ('38003', '47006'))
elif sub_stations == 'sparsely monitored':
    st_id = st.sidebar.selectbox('station', 
                                 ('23011', '23011'))
elif sub_stations == 'small catchments':
    st_id = st.sidebar.selectbox('station', 
                                 ('29002', '34012', '75017'))
elif sub_stations == 'top of catchments':
    st_id = st.sidebar.selectbox('station', 
                                 ('32008', '39026'))
elif sub_stations == 'lower in catchments':
    st_id = st.sidebar.selectbox('station', 
                                 ('28022', '45001'))
elif sub_stations == 'natural sites':
    st_id = st.sidebar.selectbox('station', 
                                 ('39065', '52010'))
elif sub_stations == 'unnatural sites':
    st_id = st.sidebar.selectbox('station', 
                                 ('33066', '33066'))
elif sub_stations == '__test_sites__':
    st_id = st.sidebar.selectbox('station', 
                                 ('28015', '28044', '30002', '33013', '34010',
                                  '34012', '34018', '39001', '39056', '39125',
                                  '40017', '46005', '46014', '47019', '48001',
                                  '49006', '54017', '54057', '54110', '76017'
                                  ))
elif sub_stations == '_PRES_':
    st_id = st.sidebar.selectbox('station', 
                                 ('48001', '49006', '47019', '40017'))
else:
    st_id = st.sidebar.selectbox('station', 
                                 ('23011', '28022', '29002', '32008', '33066',
                                  '34012', '35003', '37008', '38003', '38029',
                                  '39026', '39065', '45001', '46014', '47006', 
                                  '47019', '52010', '75017', '76021'))

st.sidebar.markdown("[*-> station info*](https://nrfa.ceh.ac.uk/data/station/meanflow/"+st_id+")")


# ''' _____________________________ DATA ________________________________ '''

merged, nn_preds = au.load_data(st_id)
n_dt = st.sidebar.slider('subset data', 1, merged.shape[0], [merged.shape[0]-400, merged.shape[0]])
merged, nn_preds = au.subset_data(merged, nn_preds, n_dt)



# ''' ____________________________ OPTS _______________________________ '''

st.sidebar.markdown('  ')
st.sidebar.markdown('  ')

ens_opt = st.sidebar.checkbox('conf intervals')
if ens_opt:
    flag_std = st.sidebar.slider('Z-score', .1, 10., step=.01, value=1.96)
else:
    flag_std = 1.96


# ''' ___________________________ FLAG OPTS ______________________________ '''


flags = st.sidebar.checkbox('flags')
if flags:
    flag_opt = st.sidebar.selectbox('algorithm', 
                                    ('Z-score', 'abs', 'KDE', 'KDE_3'))
    
    if flag_opt == 'Z-score':
        flag_abs_d = st.sidebar.number_input('|threshold|', value=0.01)
    elif flag_opt == 'abs':
        flag_abs_d = st.sidebar.number_input('|threshold|', value=0.01)
    elif flag_opt == 'KDE' or flag_opt == 'KDE_3':
        flag_kde_smoothing = st.sidebar.slider('KDE smoothing', .01, 1., 
                                               step=.01, value=.5)
else:
    flag_opt='no_flags'
    flag_abs_d=0
    
    
    
# ''' _________________________ PLOT VIZ OPTS ____________________________ '''

st.sidebar.markdown('  ')
st.sidebar.markdown('  ')
    
rel_errors = st.sidebar.checkbox('relative errors')
log_opt = st.sidebar.checkbox('log y axis')



# ''' ____________________________ ST INPS _______________________________ '''

st.sidebar.markdown('  ')
st.sidebar.markdown('  ')

show_st_inps = st.sidebar.checkbox('show model inps')
if show_st_inps:
    st.sidebar.plotly_chart(au.heatmap_model_inps(st_id))
    
    
# ''' ____________________________ STATS ________________________________ '''

Qn_stats = st.sidebar.checkbox('show fit stats')

    
# ''' ___________________________ PLOT OPTS ______________________________ '''

st.sidebar.markdown('  ')
st.sidebar.markdown('  ')

plot_opts = st.sidebar.checkbox('plot opts')
if plot_opts:
    plt_height = st.sidebar.slider('height', 1, 2000, 900)
    plt_width = st.sidebar.slider('width', 1, 4000, 800)
else:
    plt_height = 900
    plt_width = 800
    
    
    
# ''' ____________________________ FLAGS ________________________________ '''

if flag_opt == 'no_flags':
    flags = []
elif flag_opt == 'abs':
    flags = au.flag_outliers_fixed_abs(merged, flag_abs_d)
elif flag_opt == 'Z-score':
    flags = au.flag_outliers_zscore(merged, flag_std, flag_abs_d)
else:
    flags_1, flags_3 = au.flag_outliers_kde(merged, nn_preds, flag_kde_smoothing)
    if flag_opt == 'KDE':
        flags = flags_1
    elif flag_opt == 'KDE_3':
        flags = flags_3

if flag_opt != 'no_flags':    
    flags_qc = au.flag_qc_corrections(merged)    
else:
    flags_qc = []


# ''' __________________________ MAIN PLOT ______________________________ '''
    
if ens_opt:
    figc = au.fig_comb(merged, flag_std, flags, flags_qc, rel_errors, log_opt)
else:
    figc = au.fig_comb_nns(merged, nn_preds, flags, flags_qc, rel_errors, log_opt)


# ''' __________________________ STATS PLOT _____________________________ '''

if Qn_stats:
    st.plotly_chart(au.heatmap(), width=plt_width, height=plt_height-200)
else:
    st.plotly_chart(figc, width=plt_width, height=plt_height)





















