import streamlit as st
import polars as pl
from polars import col as c
import numpy as np

st.title('NPV Drug Launch Application')

peak_us_sales = st.number_input(label='Peak US Sales', value=1_200_000_000, key='peak_us_sales')
launch_wac = st.number_input(label='WAC Launch Price',value=20000,key='launch_price')
gross_margin = st.number_input(label='Gross Margin', value=.77, key='gross_margin')
peak_units = peak_us_sales / launch_wac

eunet = launch_wac * .50


with st.expander("GTN Controls"):
    cola,colb = st.columns(2)
    with cola:
        commercial_p_discount = st.number_input('Commercial Plan/PBM discount', .3)
        medicare_p_discount = st.number_input('Medicare Plan/PBM discount', .3)
        medicaid_p_discount = st.number_input('Medicaid discount', .23)
        va_p_discount = st.number_input('VA/Dod discount', .25)
        b340_p_discount = st.number_input('340B discount', .23)
        cms_statutory_p_discount = st.number_input('CMS -Statutory discount', .05)
        cms_inflation_penalty_p_discount = st.number_input('CMS - Inflation penalty discount', 0)
        prompt_pay_p_discount = st.number_input('Prompt Pay discount', .02)
        fees_chargbacks_p_discount = st.number_input('Fees/Chargbacks discount', .02)
        returns_p_discount = st.number_input('Returns discount', .01)
        in_office_dispensing_p_discount = st.number_input('In Office dispensing discount', .04)
        patient_assistance_p_discount = st.number_input('Patient Assistance discount', .2)

    with colb:
        commercial_p_business = st.number_input('Commercial Plan/PBM % business', .35)
        medicare_p_business = st.number_input('Medicare Plan/PBM % business', .5)
        medicaid_p_business = st.number_input('Medicaid % business', .1)
        va_p_business = st.number_input('VA/Dod % business', .05)
        b340_p_business = st.number_input('340B % business', .3)
        cms_statutory_p_business = st.number_input('CMS -Statutory % business', .5)
        cms_inflation_penalty_p_business = st.number_input('CMS - Inflation penalty % business', .5)
        prompt_pay_p_business = st.number_input('Prompt Pay % business', 1)
        fees_chargbacks_p_business = st.number_input('Fees/Chargbacks % business', 1)
        returns_p_business = st.number_input('Returns % business', 1)
        in_office_dispensing_p_business = st.number_input('In Office dispensing % business', .1)
        patient_assistance_p_business = st.number_input('Patient Assistance % business', .3)

source = pl.Series('source',[
'Payer',
'Payer',
'Payer',
'Payer',
'Payer',
'Payer',
'Payer',
'Distributors',
'Distributors',
'Distributors',
'Provider',
'Patient',
])

channel = pl.Series('channel',[
'Commercial Plan/PBM',
'Medicare Plan/PBM',
'Medicaid',
'VA/Dod',
'340B',
'CMS -Statutory',
'CMS - Inflation penalty',
'Prompt Pay',
'Fees/Chargbacks',
'Returns',
'In Office dispensing',
'Patient Assistance',
])

percent_business = pl.Series('percent_business',[
commercial_p_business,
medicare_p_business,
medicaid_p_business,
va_p_business,
b340_p_business,
cms_statutory_p_business,
cms_inflation_penalty_p_business,
prompt_pay_p_business,
fees_chargbacks_p_business,
returns_p_business,
in_office_dispensing_p_business,
patient_assistance_p_business,
])

average_discount = pl.Series('average_discount',[
commercial_p_discount,
medicare_p_discount,
medicaid_p_discount,
va_p_discount,
b340_p_discount,
cms_statutory_p_discount,
cms_inflation_penalty_p_discount,
prompt_pay_p_discount,
fees_chargbacks_p_discount,
returns_p_discount,
in_office_dispensing_p_discount,
patient_assistance_p_discount,
])


###gross to net
def load_gtn() -> pl.LazyFrame:
    return (
        pl.DataFrame(
        {
            'source': source,
            'channel': channel,
            'percent_business': percent_business,
            'average_discount': average_discount
        })
        .with_columns(gtn = c.percent_business.cast(pl.Float64)*c.average_discount.cast(pl.Float64))
        .lazy()
        )

st.write('Gross to Net Frame')
st.dataframe(load_gtn())


###Calender

with st.expander('Calendar Controls'):
    development_time_phase_1 = st.number_input(label='Development Time Phase 1', value=2.95,
                                               key='development_time_phase_1')
    development_time_phase_2 = st.number_input(label='Development Time Phase 2', value=3.35,
                                               key='development_time_phase_2')
    development_time_phase_3 = st.number_input(label='Development Time Phase 3', value=3.65,
                                               key='development_time_phase_3')
    development_time_registration = st.number_input(label='Development Time Registration', value=1.05,
                                                    key='development_time_registration')
    commercial_window = st.number_input(label='Commercial Window (Years)', value=14, key='commercial_window')


def calendar() -> pl.LazyFrame:
    cal = (pl.DataFrame({
        'phase': (["Development"] * 4 + ["Launched"] * commercial_window),
        't': ([development_time_phase_1, development_time_phase_2, development_time_phase_3,
               development_time_registration] + [1] * commercial_window)})
    ).lazy()
    return cal.filter(c.t > 0).with_columns(t=c.t.cum_sum())

st.dataframe(calendar().collect())

def ramp_frame():
    data = pl.DataFrame({'units': ramp(t=commercial_window,K=(peak_us_sales/launch_wac),r=growth_rate,t0=inflection_point)})
    data = data.with_columns(percent_peak = ((c.units*launch_wac) / peak_us_sales).round(2)).with_row_index('t',offset=1)
    return data

with st.expander('RAMP Controls'):
    colc,cold = st.columns(2)
    with colc:
        growth_rate = st.number_input(label='Growth Rate', value=1, key='growth_rate')
    with cold:
        inflection_point= st.number_input(label='Inflection Point',value=5,key='inflection_point')

def ramp(t=commercial_window,K=(peak_us_sales/launch_wac),r=growth_rate,t0=inflection_point) -> np.ndarray:
    t = pl.arange(1, t+1, eager=True)
    return K / (1 + np.exp(-r * (t - t0)))

st.line_chart(ramp_frame(),x='t',y='percent_peak',use_container_width=True)


with st.expander('Price Evolution Controls'):
    cole,colf = st.columns(2)
    with cole:
        medicare_negotiation_year =  st.number_input(label='Medicare Negotiation Year',value=0,key='medicare_negotiation_year')
        dpn = st.number_input(label='DPN',value=.17,key='dpn')
        cpi = st.number_input(label='CPI', value=0.03, key='cpi')
    with colf:
        increase = st.number_input(label='CPI Yearly Scaler', value=0.03, key='increase')
        decrease = st.number_input(label='ExUs Yearly Price Erosion', value=0.01, key='decrease')

#######################
def price_evolution_frame(wac_start: float | int = launch_wac) -> pl.LazyFrame:

    #filter calendar for launch years and set t_launched to cumcount of years
    #set t to years to preserve actual number of years in each phase
    t_launched = calendar().with_columns(years=c.t.round(2)).with_columns(t = c.t.cum_count().over('phase')-1).filter(c.phase == "Launched")
    t_development = calendar().with_columns(years=c.t.round(2)).with_columns(t =c.t.cum_count().over('phase') - 1).filter(c.phase == "Development")

    #Expressions
    dpn_rebate = (
        pl.when(c.t < (medicare_negotiation_year - 1))
        .then(0)
        .when(((medicare_negotiation_year == 0)&(c.t.is_not_null())))
        .then(0)
        .otherwise(dpn)
        .alias('dpn_rebate')
        )

    update_discounts = (
        pl.when(c.channel.is_in(["340B", "Medicaid"]))
        .then(c.best_price_rebate)
        .when(c.channel == "CMS - Inflation Penalty")
        .then(c.medicare_rebate)
        .when(c.channel == "Medicare Plan/PBM")
        # dpn + set Medicare discount
        .then(c.dpn_rebate + c.average_discount)
        .otherwise(c.average_discount)
        .alias('average_discount')
    )


    t_launched = (
    t_launched
    .with_columns(
        wac=(wac_start * (1 + increase) ** c.t).round(4),
        cpi_wac=(wac_start * (1 + cpi) ** c.t).round(4),
    )
    .with_columns(
        best_price_rebate=.231 + (c.wac - c.cpi_wac) / c.wac,  # Adds inflation penalty to mandatory 23.1% rebate
        medicare_rebate=(c.wac - c.cpi_wac) / c.wac, # Inflation penalty based on WAC not AMP/ASP
        dpn_rebate=dpn_rebate,
    ))


    #cross join to join all gtn rows with rows in t2
    gtn = (
        t_launched
        .join(load_gtn(),how='cross')
        .with_columns(update_discounts)
        #update gtn after updating discounts
        .with_columns(gtn=c.percent_business * c.average_discount)
        .group_by(c.t)
        .agg(c.gtn.sum().round(4))
    )

    t_launched = (
        t_launched
        .join(gtn, on='t')
        .with_columns(
            wwnetprice=(pl.when(c.phase == "Launched").then(eunet * (1 + (-1*decrease)) ** c.t).otherwise(0)).round(4),
            usnetprice=((1 - c.gtn) * c.wac).round(4)
        )
    )

    #concat t_launched with t_development to add development phase to data
    data = pl.concat([t_launched,t_development],how='diagonal').sort(by=['phase','t'])

    return data

st.write('Price Evolution Frame')
st.dataframe(price_evolution_frame().collect())

with st.expander('Breakout Controls'):
    colg,colh = st.columns(2)
    with colg:
        development_cost_phase_1 = st.number_input(label='Development Cost Phase 1', value=34_900_000,
                                                   key='development_cost_phase_1')
        development_cost_phase_2 = st.number_input(label='Development Cost Phase 2', value=102_600_000,
                                                   key='development_cost_phase_2')
        development_cost_phase_3 = st.number_input(label='Development Cost Phase 3', value=334_800_000,
                                                   key='development_cost_phase_3')
        development_cost_registration = st.number_input(label='Development Cost Registration', value=89_300_000,
                                                        key='development_cost_registration')
    with colh:
        cogs = st.number_input(label='Cost of Goods Sold', value=.24, key='cogs')
        min_cogs = st.number_input(label='Minimum Cost of Goods Sold', value=.05, key='min_cogs')
        sga_run_rate = st.number_input(label='Sales and Administrative Expenses', value=.28, key='sga_run_rate')
        us_tax_rate = st.number_input(label='US Tax Rate', value=.20, key='us_tax_rate')
        exus_tax_rate = st.number_input(label='ExUS Tax Rate', value=.14, key='exus_tax_rate')


def development_cost_frame():
    dcost = [development_cost_phase_1, development_cost_phase_2, development_cost_phase_3,
         development_cost_registration]
    dcost = pl.DataFrame({'dcost': dcost}).with_columns(c.dcost*-1).with_row_index('t').with_columns(phase=pl.lit('Development'))
    return dcost.lazy()

# TODO: set parms to input dynamic inputs


def volume_frame():
    yearly_volume_series = ramp(
        t=commercial_window,
        # set as project yearly units sold
        K=peak_units,  # peak_volumes (anticipated units sold)
        # growth rate
        r=1,  # 1
        t0=inflection_point  # t0
    )
    # create frame
    # added phase and t column for join onto price evolution frame
    volume_frame = pl.DataFrame({'volume': yearly_volume_series}).with_row_index('t').with_columns(
        phase=pl.lit('Launched')).lazy()
    return volume_frame

def cogs_frame():
    yearly_cogs_series = pl.linear_spaces(cogs,min_cogs,commercial_window,eager=True)
    cogs_frame = pl.DataFrame({'cogs_percent': yearly_cogs_series}).explode('cogs_percent').with_row_index('t').with_columns(phase=pl.lit('Launched'))
    return cogs_frame.lazy()

def sga_frame():
    sga = [1] * (commercial_window - 4) + list(0.8 ** np.arange(1, 5))
    sga = [round(float(x),2) for x in sga]
    sga = pl.DataFrame({'sga_percent': sga}).with_columns(sga_rate = (c.sga_percent*sga_run_rate).round(2)).with_row_index('t').with_columns(phase=pl.lit('Launched'))
    return sga.lazy()

def run_breakout(wac_start: float = launch_wac):

    #US Commercial cash flows
    revenue_us = (c.volume*c.usnetprice).cast(pl.Int64).alias('revenue_us')
    gross_profit_us = (gross_margin * c.revenue_us).cast(pl.Int64).alias('gross_profit_us')
    cogs_us = (c.cogs_percent*c.volume*c.usnetprice).cast(pl.Int64).alias('cogs_us')
    sga_us = (c.sga_rate*c.volume*c.usnetprice).cast(pl.Int64).alias('sga_us')
    ebit_us = (c.gross_profit_us - c.cogs_us - c.sga_us).cast(pl.Int64).alias('ebit_us')
    us_after_tax = ((1-us_tax_rate)*ebit_us).cast(pl.Int64).alias('after_tax_us')

    #EXUS Commercial cash flows
    revenue_exus = (c.volume * c.wwnetprice).cast(pl.Int64).alias('revenue_exus')
    gross_profit_exus = (gross_margin * c.revenue_exus).cast(pl.Int64).alias('gross_profit_exus')
    cogs_exus = (c.cogs_percent*c.volume*c.wwnetprice).cast(pl.Int64).alias('cogs_exus')
    sga_exus = (c.sga_rate*c.volume*c.wwnetprice).cast(pl.Int64).alias('sga_exus')
    ebit_exus = (c.gross_profit_exus - c.cogs_exus - c.sga_exus).cast(pl.Int64).alias('ebit_exus')
    exus_after_tax = ((1-exus_tax_rate)*ebit_exus).cast(pl.Int64).alias('after_tax_exus')

    peFrame= price_evolution_frame(wac_start)
    vfFrame = volume_frame()
    cogsFrame = cogs_frame()
    sgaFrame = sga_frame()

    data = (
        peFrame
        .join(development_cost_frame(),on=['phase','t'],how='left')
        .join(vfFrame,on=['phase','t'],how='left')
        .join(cogsFrame,on=['phase','t'],how='left')
        .join(sgaFrame, on=['phase', 't'], how='left')
        .with_columns(revenue_us)
        .with_columns(gross_profit_us,cogs_us,sga_us)
        .with_columns(ebit_us)
        .with_columns(us_after_tax)
        .with_columns(revenue_exus)
        .with_columns(gross_profit_exus, cogs_exus, sga_exus)
        .with_columns(ebit_exus)
        .with_columns(exus_after_tax)
    )
    return data
# join volume frame to price evolution frame on phase and t

def run_global(wac_start: float = launch_wac):
    return (
    run_breakout(wac_start)
    .group_by(c.phase,c.t,c.years)
    .agg(
        c.dcost.sum(),
        revenue = c.revenue_us.sum()+c.revenue_exus.sum(),
        gross_profit = c.gross_profit_us.sum() + c.gross_profit_exus.sum(),
        cogs = c.cogs_us.sum() + c.cogs_exus.sum(),
        sga = c.sga_us.sum() + c.sga_exus.sum(),
        ebit = c.ebit_us.sum() + c.ebit_exus.sum(),
        after_tax = c.after_tax_us.sum() + c.after_tax_exus.sum()
    )
    .sort(by='years',descending=False)
    )

st.write('Breakout Frame')
st.write(run_breakout().collect())

st.write('Global Frame')
st.write(run_global().collect())

with st.expander('NPV Controls'):
    r = st.number_input('Net Present Value Discount Rate',.11)
    pos_phase_1 = st.number_input('Probability of Successes Phase 1' ,.41)  # probility of success
    pos_phase_2 = st.number_input('Probability of Successes Phase 2' ,.46)
    pos_phase_3 = st.number_input('Probability of Successes Phase 3' ,.67)
    pos_registration = st.number_input('Probability of Successes Registration' ,.88)
pos = pos_phase_1 * pos_phase_2 * pos_phase_3 * pos_registration

def npv_calculation(global_cashflow: pl.LazyFrame) -> pl.Int64:
    rcflow = (c.after_tax * pos).cast(pl.Int64).alias('rcflow')  # risk adjusted cash flow
    ddcost = (c.dcost / (1 + r) ** c.years).alias('ddcost')  # dicounted development cost
    return (
    global_cashflow
    .with_columns(rcflow,ddcost)
    .select(npv=(c.rcflow.sum()+c.ddcost.sum()).cast(pl.Int64))
    .collect()
    .item()
    )

st.write(f"NPV: {'${:,.0f}'.format(npv_calculation(run_global()))}")

with st.expander('WAC Analysis Controls'):
    min_wac = st.number_input('Min WAC',value=launch_wac*.5)
    max_wac = st.number_input('Max WAC',value=launch_wac*2)
    steps = st.number_input('Steps',value=25)

def wac_npv_analysis(min_wac: float = min_wac, max_wac: float = max_wac,wac_steps: int = steps):
    wac_to_test = np.linspace(min_wac, max_wac, wac_steps)
    npvs = ([npv_calculation(run_global(wac)) for wac in wac_to_test])
    return pl.DataFrame({'wac': wac_to_test, 'npv': npvs})

wac_analysis_frame = wac_npv_analysis()
st.write('WAC NPV Analysis')
st.write(wac_analysis_frame)

st.line_chart(data = wac_analysis_frame,x='wac',y='npv')





