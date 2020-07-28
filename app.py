import streamlit as st

import app_utils as au


def main():
    
    # ''' __________________________ SIDEBAR ______________________________ '''
    
    st.sidebar.markdown('**Station selection:**')
    
    st_id = st.sidebar.selectbox('station ID', 
                                 ("33013", "34010", "34012", "34018", "39056",
                                  "40017", "46005", "47019", "48001", "49006"))
    
    st.sidebar.markdown("[station info](https://nrfa.ceh.ac.uk/data/station/meanflow/"+st_id+")")
    
    
    # ''' _____________________________ DATA ________________________________ '''
    
    merged, nn_preds = au.load_data(st_id)
    n_dt = st.sidebar.slider('subsetting', 0, merged.shape[0],
                             (merged.shape[0]-1400, merged.shape[0]))
    merged, nn_preds = au.subset_data(merged, nn_preds, n_dt)
    
    
    # ''' ____________________________ OPTS _______________________________ '''
    
    st.sidebar.markdown('  ')
    st.sidebar.markdown('  ')
    
    st.sidebar.markdown('**Plot pars:**')
    ens_opt = st.sidebar.checkbox('conf intervals', True)
    
    
    # ''' ___________________________ FLAGS  ______________________________ '''
    
    flags = st.sidebar.checkbox('flags')
    
    # default flaggers
    def_flaggers = au.get_def_flaggers()
    
    flag_std = float(def_flaggers.loc[int(st_id), "std"])
    flag_abs_d = float(def_flaggers.loc[int(st_id), "abs_d"])
    flag_kde_smoothing = .5
    
    if flags:
        flag_opt = st.sidebar.selectbox('method', 
                                        ('Z-score', '|threshold|', 'KDE'))
        
        if flag_opt == 'Z-score':
            flag_std = st.sidebar.number_input('Z-score', step=1., value=flag_std)
            flag_abs_d = st.sidebar.number_input('|threshold|', value=flag_abs_d)
        elif flag_opt == '|threshold|':
            flag_abs_d = st.sidebar.number_input('value', value=.0)
        elif flag_opt == 'KDE':
            flag_kde_smoothing = st.sidebar.slider('KDE smoothing', .01, 1., 
                                                   step=.01, value=.5)
    else:
        flag_opt = 'no_flags'
        
        
    # ''' ______________________ PLOT VIZ OPTS ____________________________ '''
    
    st.sidebar.markdown('  ')
    st.sidebar.markdown('  ')
        
    log_opt = st.sidebar.checkbox('log y axis')   
    rel_errors = st.sidebar.checkbox('relative errors')
    
    
    # ''' __________________________ ST INPS ______________________________ '''
    
    st.sidebar.markdown('  ')
    st.sidebar.markdown('  ')
    
    show_st_inps = st.sidebar.checkbox('show model inps')
    if show_st_inps:
        st.sidebar.plotly_chart(au.heatmap_model_inps(st_id),
                                use_container_width=True)
        
        
    # ''' ___________________________ STATS _______________________________ '''
    
    Qn_stats = st.sidebar.checkbox('show fit stats')
    
        
    # ''' _________________________ PLOT OPTS _____________________________ '''
    
    st.sidebar.markdown('  ')
    st.sidebar.markdown('  ')
    
    plot_opts = st.sidebar.checkbox('adjust plot size')
    if plot_opts:
        plt_height = st.sidebar.slider('height', 1, 2000, 900)
        plt_width = st.sidebar.slider('width', 1, 4000, 800)
    else:
        plt_height = 900
        plt_width = 800
        
        
    # ''' ___________________________ FLAGS _______________________________ '''
    
    if flag_opt == 'no_flags':
        flags = []
    elif flag_opt == '|threshold|':
        flags = au.flag_outliers_fixed_abs(merged, flag_abs_d)
    elif flag_opt == 'Z-score':
        flags = au.flag_outliers_zscore(merged, flag_std, flag_abs_d)
    else:
        flags = au.flag_outliers_kde(merged, nn_preds, flag_kde_smoothing)
    
    if flag_opt != 'no_flags':    
        flags_qc = au.flag_qc_corrections(merged)    
    else:
        flags_qc = []


    # ''' __________________________ LOGOS ______________________________ '''
        
    xgb_logo, keras_logo = au.load_logos()

    st.sidebar.markdown(' ')
    st.sidebar.markdown(' ')
    
    st.sidebar.image(xgb_logo, width=80)   
    st.sidebar.image(keras_logo, width=160)
    
    # ''' __________________________ MAIN PLOT ______________________________ '''
        
    if ens_opt:
        figc = au.fig_comb(merged, flag_std, flag_abs_d, flags, flags_qc,
                           flag_opt, rel_errors, log_opt,
                           plt_height, plt_width)
    else:
        figc = au.fig_comb_nns(merged, nn_preds, flag_abs_d, flags, flags_qc,
                               flag_opt, rel_errors, log_opt,
                               plt_height, plt_width)
    
    
    # ''' __________________________ STATS PLOT _____________________________ '''
    
    if Qn_stats:
        st.plotly_chart(au.heatmap(), use_container_width=False)
    else:
        st.plotly_chart(figc, use_container_width=False)
    
   
if __name__ == "__main__":
    main()
