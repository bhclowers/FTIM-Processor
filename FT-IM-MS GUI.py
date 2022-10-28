import panel as pn
import numpy as np
from pyteomics import mzml
from scipy import signal
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import UnivariateSpline
import itertools, os
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from lmfit.models import GaussianModel

from FTGUI_funcs import getMS_v2, getXIC, loadOrbiMS_v2, getXIC_Orbi, \
    regrid, scanOrbi, zero_padding, getAFT, multiFT_data, XIC_preview, moCal

from CCS_funcs import calcThermalVelocity, calcAlpha, calcTransverseVelCoeff,\
    calcNumberDensity, ccsFromDriftTime, calcCorrectedCCS

pn.extension('tabulator')


#========================================================================
#default current working directory - relative to root
DEF_CWD = '/Documents/GUI Testing'

#default m/z ranges to extract XICs from
DEF_MZ_VALS = [[298.0, 302.0], [354.0, 357.0], [410.0, 413.0],
               [466.0, 469.0], [578.0, 582.0], [690.0, 694.0]]

#slicing off large values in frequency domain post-FT
ATD_SLICE = 30

#color palettes for XIC and ATD plots - see seaborn color palettes
XIC_COLORS = 'magma'
ATD_COLORS = 'icefire_r'

#go through and change enable/disable options
#clear all plots when appropriate


#containers for all content
main_grid = pn.GridSpec(nrows=9,
                        ncols=16,
                        sizing_mode='stretch_both')

main_tabs = pn.Tabs(('Main figures', main_grid))


fname_input = pn.widgets.TextInput(name='Filename',
                                   placeholder = 'Enter csv filename',
                                   sizing_mode = 'stretch_width')


exp_df_button = pn.widgets.Button(name='Export Fits',
                                  margin = (24,5,5,5))
exp_button_row = pn.Row(fname_input,
                        exp_df_button)

df_viewer = pn.widgets.Tabulator(sizing_mode='stretch_width',
                                 layout = 'fit_columns')


def fitsDF_exporter(event):
    curDF = df_viewer.value
    if len(fname_input.value) == 0:
        curDF.to_csv('placeholder_name.csv', sep=',')
    else:
        curDF.to_csv(fname_input.value, sep=',')

exp_df_button.on_click(fitsDF_exporter)

df_divider = pn.layout.Divider()
export_df_col = pn.Column(exp_button_row,
                          df_divider,
                          df_viewer,
                          sizing_mode = 'stretch_width')


ccs_button = pn.widgets.Button(name='Calculate CCS',
                               button_type = 'success',
                               margin = (24,5,5,5))
charge_state_vals = pn.Card(title='m/z charge states')
neutral_amu_input = pn.widgets.FloatInput(name='Drift Gas Mass (amu)',
                                        value = 28.0,
                                        step = 1,
                                        start = 1,
                                        end = 200,
                                        width = 250)
charge_state_vals.append(neutral_amu_input)
ccs_col = pn.Column(ccs_button,
                    charge_state_vals)

fit_tab_row = pn.Row(export_df_col, ccs_col,
                     sizing_mode = 'stretch_both')

main_tabs.append(('Fits view', fit_tab_row))


#initializing plotly figures =============================================
mzFig =  go.Figure()
atdFig = go.Figure()
xicFig = go.Figure()


mzFig.update_xaxes(title='m/z', showline=True, linecolor='black')
mzFig.update_yaxes(title='Intensity (a.u.)', showline=True, linecolor='black',
                   linewidth=1.1)
mzFig.update_layout(template = 'plotly_white',
                    autosize=False,
                    width= 650,
                    height = 350,
                    margin=dict(t=25,r=5,b=5,l=5))

atdFig.update_layout(template = 'plotly_white',
                     autosize=False,
                     legend = dict(x=0.80,y=0.95),
                     width=980,
                     height=420,
                     margin=dict(t=5,l=5,b=5,r=5))

xicFig.update_layout(template = 'plotly_white',
                     autosize = False,
                     width=980,
                     height=200,
                     margin=dict(t=5,r=5,b=5,l=5))


#creating containers for each plotly plot
mzPane = pn.pane.Plotly(mzFig, height=350)
xicPane = pn.pane.Plotly(xicFig)
atdPane = pn.pane.Plotly(atdFig, width=980, height=450)

#temporary tab to maintain gui shape, will be cleared at XIC extraction
xicTabs = pn.Tabs(('curxic',xicPane))



file_input = pn.widgets.FileSelector('~'+DEF_CWD,
                                     refresh_period=5000,
                                     height = 325)


#value argument specifies the default value of the radio buttons
ms_type_radio = pn.widgets.RadioBoxGroup(name='Instrument',
                                         options=['HiRes', 'Std'],
                                         value = 'Std',
                                         inline=True,
                                         margin = 15,
                                         width = 100)

load_mz_button = pn.widgets.Button(name='Load selected file',
                                   sizing_mode='stretch_width',
                                   margin=10)


current_dataset = {}
current_mzSpectrum = {}
def load_file(event):
    current_dataset.clear()
    current_mzSpectrum.clear()
    mzFig.data = []
    curFile = file_input.value[0]
    fname = os.path.basename(curFile)

    if ms_type_radio.value == 'Std':
        spectral_data = [s for s in mzml.read(curFile)]
        mz_x, mz_y = getMS_v2(spectral_data)

    elif ms_type_radio.value == 'HiRes':
        mz_x, mz_y, spectral_data = loadOrbiMS_v2(curFile)

    current_dataset['DATASET'] = spectral_data
    #this is referenced again in CCS calc function
    current_mzSpectrum['mz_x'] = mz_x
    current_mzSpectrum['mz_y'] = mz_y


    curTrace = go.Scatter(x=mz_x, y=mz_y,
                          line=dict(color='crimson', width=1))
    mzFig.update_layout(title=fname)
    mzFig.update_xaxes(range=(mz_x.min(), mz_x.max()))
    mzFig.add_trace(curTrace)



load_mz_button.on_click(load_file)

file_load_row = pn.Row(load_mz_button, ms_type_radio)



mzranges_input = pn.widgets.ArrayInput(
    name='XICs for the following m/z ranges will be extracted:',
    max_array_size = 20,
    value = np.array(DEF_MZ_VALS),
    sizing_mode='stretch_width')

add_xlims_button = pn.widgets.Button(name='Add current m/z range to list',
                                     sizing_mode='stretch_width',
                                     button_type='default')

extract_xic_button = pn.widgets.Button(name='Extract XICs',
                                       sizing_mode='stretch_width',
                                       button_type='primary',
                                       disabled=False)


def add_mz_xlims(event):
    cur_min_mz, cur_max_mz = mzFig.layout.xaxis.range
    print(cur_min_mz, cur_max_mz)
    mzranges_input.value = np.append(mzranges_input.value,
                                     [[round(cur_min_mz,1),
                                       round(cur_max_mz,1)]],
                                     axis=0)

add_xlims_button.on_click(add_mz_xlims)


xic_color_gen = sns.color_palette(XIC_COLORS, 10).as_hex()
xic_colors = itertools.cycle(xic_color_gen)
xic_trace_dict = {}
monoPeak_dict = {}


def extract_XICs(event):
    xicTabs.clear()
    xic_trace_dict.clear()
    monoPeak_dict.clear()

    mzx = current_mzSpectrum['mz_x']
    mzy = current_mzSpectrum['mz_y']

    for curmz_pair in mzranges_input.value:
        mz1 = curmz_pair[0]
        mz2 = curmz_pair[1]
        mzCenter = (mz1+mz2)/2
        mzTol = mz2 - mzCenter

        if ms_type_radio.value == 'Std':
            xic_x, xic_y = getXIC(current_dataset['DATASET'],
                                  mzCenter, tol=mzTol)

        elif ms_type_radio.value == 'HiRes':
            xic_x, xic_y = getXIC_Orbi(current_dataset['DATASET'],
                                       mzCenter, tol=mzTol)

        xicKey = 'm/z {:.1f}-{:.1f}'.format(mz1,mz2)
        xic_trace_dict[xicKey] = (xic_x, xic_y)

        #used for CCS calculations
        mzLo_ind, mzHi_ind = np.searchsorted(mzx, [mz1, mz2])
        maxPeakInd = np.argmax(mzy[mzLo_ind:mzHi_ind])
        monoPeak = mzx[mzLo_ind:mzHi_ind][maxPeakInd]
        monoPeak_dict[xicKey] = round(monoPeak,1)


        cur_xicFig = go.Figure()
        cur_xicFig.update_layout(template='plotly_white',
                                 autosize=False,
                                 legend=dict(x=0.85, y=0.95),
                                 width=980,
                                 height=200,
                                 margin=dict(t=5,r=5,b=5,l=5))

        cur_xicFig.update_xaxes(title='Time (s)',
                                showline=True,
                                linecolor='black')

        cur_xicFig.update_yaxes(showline=True, linecolor='black')

        cur_line = go.Scatter(x=xic_x, y=xic_y, name = xicKey,
                                  line=dict(color=next(xic_colors), width=1))

        cur_xicFig.add_trace(cur_line)

        cur_xic_pane = pn.pane.Plotly(cur_xicFig, width=980, height=200)
        xicTabs.append((xicKey, cur_xic_pane))


extract_xic_button.on_click(extract_XICs)


mz_opts_row = pn.Row(add_xlims_button, extract_xic_button,
                     sizing_mode='stretch_width')

mz_opts_col = pn.Column(mzranges_input, mz_opts_row,
                        sizing_mode='stretch_width')

apdz_checkbox = pn.widgets.Checkbox(name='Apodization',
                                    width=100,
                                    margin= (22,5,5,50))

apdz_list = ['bartlett', 'blackman', 'hamming', 'hanning',
             'barthann', 'bohman', 'nuttall', 'parzen', 'tukey']
apdz_type = pn.widgets.Select(name='Type:',
                              options=apdz_list,
                              value = 'hanning',
                              width = 150)

zpad_checkbox = pn.widgets.Checkbox(name='Zero padding',
                                    width=100,
                                    margin=(22,5,5,10))


zpad_len_box = pn.widgets.FloatInput(name='Pad length', value=1, step=1,
                                     start=0, end=10,
                                     width=100)

xic_change_button = pn.widgets.Button(name='Apply XIC changes / Confirm XIC',
                                      button_type='primary',
                                      margin=(22,5,5,5),
                                      disabled=False)

xic_reset_button = pn.widgets.Button(name='Reset XICs',
                                     margin=(22,5,5,5),
                                     width=100,
                                     disabled=True)

xic_mods_row = pn.Row(apdz_checkbox, apdz_type,
                      zpad_checkbox, zpad_len_box,
                      xic_change_button, xic_reset_button,
                      margin=(5,5,0,5))

xic_mod_div = pn.layout.Divider(margin=(5,5,0,22))


def xic_changer(event):
    for curTab in xicTabs:
        curDataContainer = curTab.object.data[0]
        curDataKey = curDataContainer.name

        x_orig, y_orig = xic_trace_dict[curDataKey]

        y_mod = XIC_preview(amp = y_orig,
                               windowBool = apdz_checkbox.value,
                               window = apdz_type.value,
                               padBool = zpad_checkbox.value,
                               padLen = zpad_len_box.value)

        if zpad_checkbox.value:
            curDataContainer.x = None
            curDataContainer.figure.update_xaxes(title='Bins')

        curDataContainer.y = y_mod

        xic_change_button.disabled = True
        xic_reset_button.disabled = False
        plot_ATD_button.disabled = False


xic_change_button.on_click(xic_changer)


def xic_resetter(event):
    for curTab in xicTabs:
        curDataContainer = curTab.object.data[0]
        curDataKey = curDataContainer.name
        x_orig, y_orig = xic_trace_dict[curDataKey]

        curDataContainer.x = x_orig
        curDataContainer.y = y_orig


        curDataContainer.figure.update_xaxes(title='Time (s)')

    xic_change_button.disabled = False
    xic_reset_button.disabled = True
    plot_ATD_button.disabled = True

xic_reset_button.on_click(xic_resetter)




sRate_input = pn.widgets.FloatInput(name='Sweep Rate (Hz/s)',
                                    width= 150,
                                    value=10,
                                    step=0.5,
                                    start=0.1,
                                    end=100)

length_input = pn.widgets.FloatInput(name='Length (cm)',
                                     width=120,
                                     value=17.4,
                                     step=0.1,
                                     start=0.1,
                                     end=100)

voltage_input = pn.widgets.FloatInput(name='Voltage (V)',
                                      width=120,
                                      value=1000,
                                      step=1,
                                      start=0.1,
                                      end=50000)

temp_input = pn.widgets.FloatInput(name='Temp (C)',
                                   width=120,
                                   value=25.0,
                                   step=1,
                                   start=-273.15,
                                   end=1000)

press_input = pn.widgets.FloatInput(name='Pressure (Torr)',
                                    width=120,
                                    value = 700,
                                    step = 1,
                                    start=0,
                                    end=10000)

tOffset_input = pn.widgets.FloatInput(name='tOffset',
                                      width=120,
                                      value=0.1,
                                      step=0.1,
                                      start=0,
                                      end=10,
                                      disabled=True)

expParams_row = pn.Row(sRate_input, length_input, voltage_input,
                       temp_input, press_input, tOffset_input,
                       margin=(0,0,0,50))


plot_ATD_button = pn.widgets.Button(name='Reconstruct ATD',
                                    button_type='success',
                                    min_width = 400,
                                    disabled = True)

only_FT_checkbox = pn.widgets.Checkbox(name='FT only',
                                       margin=(15,5),
                                       width=70)

aFT_checkbox = pn.widgets.Checkbox(name='aFT',
                                   width=60,
                                   margin=(15,0),
                                   disabled=True)


palette_gen = sns.color_palette(ATD_COLORS, 10).as_hex()
atd_colors = itertools.cycle(palette_gen)

atd_traces = {}
def ATD_plotter(event):
    atd_traces.clear()
    atdFig.data = []
    if only_FT_checkbox.value:
        atd_xlabel = 'Frequency (Hz)'
    else:
        atd_xlabel = 'Time (ms)'

    sRate = sRate_input.value

    atdFig.update_xaxes(title=atd_xlabel)
    atdFig.update_yaxes(title='Intensity (a.u.)')
    for key, val in xic_trace_dict.items():
        time, amp = val
        ft_x, ft_y = multiFT_data(time = time, amp = amp,
                                  sweepRate=sRate,
                                  method = 'FFT',
                                  windowBool=apdz_checkbox.value,
                                  window = apdz_type.value,
                                  padBool = zpad_checkbox.value,
                                  padLen = zpad_len_box.value,
                                  tOffset = tOffset_input.value)

        if only_FT_checkbox.value:
            FT_x = ft_x[ATD_SLICE:] #keeping as frequency domain
        else:
            FT_x = (ft_x[ATD_SLICE:]/sRate)*1000 #converting to time (ms)

        FT_y = ft_y[ATD_SLICE:]

        atd_traces[key] = (FT_x, FT_y)

        atdLine = go.Scatter(x=FT_x, y=FT_y, name=key,
                             line=dict(color=next(atd_colors), width=1))
        atdFig.add_trace(atdLine)


plot_ATD_button.on_click(ATD_plotter)


min_dist_input = pn.widgets.FloatInput(name='Min distance',
                                       value = 1,
                                       width = 100)
min_width_input = pn.widgets.FloatInput(name='Min width',
                                        value = 1,
                                        width = 100)
min_prom_input = pn.widgets.FloatInput(name='Prominence',
                                       value = 0.4,
                                       step = 0.1,
                                       end = 1.0,
                                       width = 100)
update_fits_button = pn.widgets.Button(name='Update fits',
                                       width = 100)

fits_dropdown = pn.Card(min_dist_input,
                        min_width_input,
                        min_prom_input,
                        update_fits_button,
                        collapsed = True,
                        title = 'Fit Params',
                        max_width = 200,
                        sizing_mode = 'stretch_width')

fit_peaks_button = pn.widgets.Button(name='Fit Peaks',
                                     width=100,
                                     margin=(5,5))

atd_reset_button = pn.widgets.Button(name='Reset plot',
                                     width=100,
                                     margin=(5,5))

def atd_resetter(event):
    atdFig.data = atdFig.data[:len(atd_traces.keys())]

atd_reset_button.on_click(atd_resetter)

def peak_fitter(event):
    resultsList = []
    #removing any peak fitting traces, keys should only reflect data
    atdFig.data = atdFig.data[:len(atd_traces.keys())]
    # charge_state_vals.clear()
    #required to keep drift gas amu input box initialized at start of script
    charge_state_vals.objects = [charge_state_vals.objects[0]]
    for k, (key, val) in enumerate(atd_traces.items()):
        xvec, yvec = val
        scaleFactor = yvec.max()
        y_norm = yvec/scaleFactor
        peakInds,_ = find_peaks(y_norm,
                                prominence = min_prom_input.value,
                                width = min_width_input.value,
                                distance = min_dist_input.value)
        width_results = peak_widths(y_norm, peakInds, rel_height=0.5)[0]
        #average delta time to calculate peak width
        avg_del_xvec = np.diff(xvec).mean()
        #convert to std dev for fitting instead of fwhm
        raw_sigma_est = (width_results*avg_del_xvec)/2.35482

        for cp, curParams in enumerate(zip(peakInds, raw_sigma_est)):
            curAnalyte = 'Peak{}_'.format(cp+1) #arb name
            curGaussMod = GaussianModel(prefix=curAnalyte)

            if cp == 0:
                # initializing the composite fit model
                pars = curGaussMod.guess(y_norm, x=xvec)
            else:
                pars.update(curGaussMod.make_params())

            raw_dt = xvec[curParams[0]]
            raw_sigma = curParams[1]
            raw_amp = y_norm[curParams[0]]

            pars['{}center'.format(curAnalyte)].set(value=raw_dt,
                                                     min=raw_dt - 0.1,
                                                     max=raw_dt + 0.1)

            pars['{}sigma'.format(curAnalyte)].set(value=raw_sigma)

            pars['{}amplitude'.format(curAnalyte)].set(value=raw_amp)

            if cp == 0:
                mod = curGaussMod
            else:
                mod += curGaussMod

            init = mod.eval(pars, x=xvec)
            out = mod.fit(y_norm, pars, x=xvec)
            comps = out.eval_components(x=xvec)

            fitLine = go.Scatter(x=xvec, y=comps[curAnalyte]*scaleFactor,
                                name = key+' - fit',
                                line=dict(color='crimson', width=1))

            atdFig.add_trace(fitLine)

            ctr = out.params['{}center'.format(curAnalyte)].value
            fwhm = out.params['{}fwhm'.format(curAnalyte)].value
            K0 = moCal(dtime=ctr, length = length_input.value,
                       voltage = voltage_input.value,
                       T = temp_input.value + 273.15, P = press_input.value)

            temp_dict = {'m/z Range' : key,
                         'Peak': curAnalyte,
                         'Drift Time': ctr,
                         'K0': K0,
                         'RP': ctr/fwhm,
                         'Monoisotopic Peak': monoPeak_dict[key]}

            resultsList.append(temp_dict)

        curFloatInput = pn.widgets.IntInput(name = key,
                                            value = 1,
                                            step = 1,
                                            start = 1,
                                            end = 200,
                                            width = 250)


        charge_state_vals.append(curFloatInput)

    columns = ['Monoisotopic Peak', 'm/z Range', 'Peak',
               'Drift Time', 'K0', 'RP']
    df = pd.DataFrame(resultsList, columns=columns)
    DF = df.sort_values(by=['m/z Range', 'Peak'], ignore_index=True)

    df_viewer.value = DF


fit_peaks_button.on_click(peak_fitter)
update_fits_button.on_click(peak_fitter)

vdvtfc = np.loadtxt('velocityRatio.csv', skiprows=1, delimiter=',')
vdvtCalc = vdvtfc[:, 0]
fcCalc = vdvtfc[:, 1]

f = UnivariateSpline(vdvtCalc, fcCalc, s=0)
xnew = np.linspace(vdvtCalc.min(), vdvtCalc.max(),
                   len(vdvtCalc) * 100)  #increase the length of vdvtCalc
fcCalcNew = f(xnew)

def ccs_calc(event):
    curDF = df_viewer.value

    zDict = {}
    for zBox in charge_state_vals.objects:
        zDict[zBox.name] = zBox.value

    curDF['Charge State'] = [zDict[pk] for pk in curDF['m/z Range']]

    curDF['CCS A^2'] = calcCorrectedCCS(
        ionMass=curDF['Monoisotopic Peak'] * curDF['Charge State'],
        ionCharge=curDF['Charge State'],
        gasMass=neutral_amu_input.value,
        driftLength=length_input.value,
        driftPotential=voltage_input.value,
        driftTime=curDF['Drift Time']/1000,
        gasPress=press_input.value,
        gasTemp=temp_input.value + 273.15,
        fcList=[xnew, fcCalcNew],
        debug=False)


    df_viewer.value = curDF


ccs_button.on_click(ccs_calc)




plot_atd_row = pn.Row(plot_ATD_button,
                      only_FT_checkbox,
                      aFT_checkbox,
                      fit_peaks_button,
                      atd_reset_button,
                      fits_dropdown)

ft_opts_col = pn.Column(xic_mods_row, xic_mod_div,
                        expParams_row, plot_atd_row)


left_col = pn.Column(file_input, file_load_row, mzPane, mz_opts_col)
right_col = pn.Column(xicTabs, ft_opts_col, atdPane)


main_grid[0:8, 0:6] = left_col
main_grid[0:8, 6:15] = right_col



app = pn.panel(main_tabs)
server = app.show(threaded=True)
