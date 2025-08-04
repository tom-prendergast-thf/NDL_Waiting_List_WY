from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, when, count, lit, sum as spark_sum, max as spark_max,
    sequence, explode, to_date, expr, date_sub, add_months,
    countDistinct, datediff
)
from pyspark.sql.types import IntegerType, DoubleType, StringType, StructType, StructField

import pandas as pd
import numpy as np
from scipy.stats import norm, gamma, chi2_contingency

import pyfixest as pf
import formulaic

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

from functools import reduce
import os
from importlib import resources

from pyspark.sql.functions import *


from scipy.stats import norm, gamma
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# Initialize Spark session
spark = SparkSession.builder.appName("simulate_waiting_list_data").getOrCreate()

cancel_pat = ['WH50A']
cancel_hosp = ['WH50B']

antidep="0403"
pain="0407"
antib="050101"

sicknote_codes = [
    "751641000000105",
    "751621000000103",
    "751601000000107",
    "751481000000104",
    "775351000000103",
    "775321000000108",
    "775301000000104",
    "775281000000100",
    "775261000000109",
    "775241000000108",
    "775221000000101",
    "751751000000104",
    "751731000000106",
    "1341000000107",
    "5121000000100",
    "1351000000105",
    "1321000000100",
    "5121000000100",
    "1331000000103",
    "14241000000101",
    "165801000000106",
    "374171000000109",
    "925481000000108"
]

data = [
    ("C_100", "General Surgery", 100),
    ("C_101", "Urology", 101),
    ("C_110", "Trauma and Orthopaedic", 108),
    ("C_110", "Trauma and Orthopaedic", 110),
    ("C_110", "Trauma and Orthopaedic", 111),
    ("C_110", "Trauma and Orthopaedic", 115),
    ("C_120", "Ear Nose and Throat", 120),
    ("C_130", "Ophthalmology", 130),
    ("C_140", "Oral Surgery", 140),
    ("C_140", "Oral Surgery", 144),
    ("C_140", "Oral Surgery", 145),
    ("C_150", "Neurosurgery", 150),
    ("C_160", "Plastic Surgery", 160),
    ("C_170", "Cardiothoracic Surgery", 170),
    ("C_170", "Cardiothoracic Surgery", 172),
    ("C_170", "Cardiothoracic Surgery", 173),
    ("C_300", "General Internal Medicine", 300),
    ("C_301", "Gastroenterology", 301),
    ("C_320", "Cardiology", 320),
    ("C_330", "Dermatology", 330),
    ("C_340", "Respiratory Medicine", 340),
    ("C_400", "Neurology", 400),
    ("C_410", "Rheumatology", 410),
    ("C_430", "Elderly Medicine", 430),
    ("C_502", "Gynaecology", 502),
    ("X02", "Other - Medical Services", 180),
    ("X02", "Other - Medical Services", 190),
    ("X02", "Other - Medical Services", 191),
    ("X02", "Other - Medical Services", 192),
    ("X02", "Other - Medical Services", 200),
    ("X02", "Other - Medical Services", 302),
    ("X02", "Other - Medical Services", 303),
    ("X02", "Other - Medical Services", 304),
    ("X02", "Other - Medical Services", 305),
    ("X02", "Other - Medical Services", 306),
    ("X02", "Other - Medical Services", 307),
    ("X02", "Other - Medical Services", 308),
    ("X02", "Other - Medical Services", 309),
    ("X02", "Other - Medical Services", 310),
    ("X02", "Other - Medical Services", 311),
    ("X02", "Other - Medical Services", 312),
    ("X02", "Other - Medical Services", 313),
    ("X02", "Other - Medical Services", 314),
    ("X02", "Other - Medical Services", 315),
    ("X02", "Other - Medical Services", 316),
    ("X02", "Other - Medical Services", 317),
    ("X02", "Other - Medical Services", 318),
    ("X02", "Other - Medical Services", 319),
    ("X02", "Other - Medical Services", 322),
    ("X02", "Other - Medical Services", 323),
    ("X02", "Other - Medical Services", 324),
    ("X02", "Other - Medical Services", 325),
    ("X02", "Other - Medical Services", 326),
    ("X02", "Other - Medical Services", 327),
    ("X02", "Other - Medical Services", 328),
    ("X02", "Other - Medical Services", 329),
    ("X02", "Other - Medical Services", 331),
    ("X02", "Other - Medical Services", 333),
    ("X02", "Other - Medical Services", 335),
    ("X02", "Other - Medical Services", 341),
    ("X02", "Other - Medical Services", 342),
    ("X02", "Other - Medical Services", 343),
    ("X02", "Other - Medical Services", 344),
    ("X02", "Other - Medical Services", 345),
    ("X02", "Other - Medical Services", 346),
    ("X02", "Other - Medical Services", 347),
    ("X02", "Other - Medical Services", 348),
    ("X02", "Other - Medical Services", 349),
    ("X02", "Other - Medical Services", 350),
    ("X02", "Other - Medical Services", 352),
    ("X02", "Other - Medical Services", 360),
    ("X02", "Other - Medical Services", 361),
    ("X02", "Other - Medical Services", 370),
    ("X02", "Other - Medical Services", 371),
    ("X02", "Other - Medical Services", 401),
    ("X02", "Other - Medical Services", 422),
    ("X02", "Other - Medical Services", 424),
    ("X02", "Other - Medical Services", 431),
    ("X02", "Other - Medical Services", 450),
    ("X02", "Other - Medical Services", 451),
    ("X02", "Other - Medical Services", 460),
    ("X02", "Other - Medical Services", 461),
    ("X02", "Other - Medical Services", 501),
    ("X02", "Other - Medical Services", 503),
    ("X02", "Other - Medical Services", 504),
    ("X02", "Other - Medical Services", 505),
    ("X02", "Other - Medical Services", 834),
    ("X03", "Other - Mental Health Services", 656),
    ("X03", "Other - Mental Health Services", 700),
    ("X03", "Other - Mental Health Services", 710),
    ("X03", "Other - Mental Health Services", 711),
    ("X03", "Other - Mental Health Services", 712),
    ("X03", "Other - Mental Health Services", 713),
    ("X03", "Other - Mental Health Services", 715),
    ("X03", "Other - Mental Health Services", 720),
    ("X03", "Other - Mental Health Services", 721),
    ("X03", "Other - Mental Health Services", 722),
    ("X03", "Other - Mental Health Services", 723),
    ("X03", "Other - Mental Health Services", 724),
    ("X03", "Other - Mental Health Services", 725),
    ("X03", "Other - Mental Health Services", 726),
    ("X03", "Other - Mental Health Services", 727),
    ("X03", "Other - Mental Health Services", 728),
    ("X03", "Other - Mental Health Services", 729),
    ("X03", "Other - Mental Health Services", 730),
    ("X03", "Other - Mental Health Services", None),
    ("X04", "Other – Paediatric Services", 142),
    ("X04", "Other – Paediatric Services", 171),
    ("X04", "Other – Paediatric Services", 211),
    ("X04", "Other – Paediatric Services", 212),
    ("X04", "Other – Paediatric Services", 213),
    ("X04", "Other – Paediatric Services", 214),
    ("X04", "Other – Paediatric Services", 215),
    ("X04", "Other – Paediatric Services", 216),
    ("X04", "Other – Paediatric Services", 217),
    ("X04", "Other – Paediatric Services", 218),
    ("X04", "Other – Paediatric Services", 219),
    ("X04", "Other – Paediatric Services", 220),
    ("X04", "Other – Paediatric Services", 221),
    ("X04", "Other – Paediatric Services", 222),
    ("X04", "Other – Paediatric Services", 223),
    ("X04", "Other – Paediatric Services", 230),
    ("X04", "Other – Paediatric Services", 240),
    ("X04", "Other – Paediatric Services", 241),
    ("X04", "Other – Paediatric Services", 242),
    ("X04", "Other – Paediatric Services", 250),
    ("X04", "Other – Paediatric Services", 251),
    ("X04", "Other – Paediatric Services", 252),
    ("X04", "Other – Paediatric Services", 253),
    ("X04", "Other – Paediatric Services", 254),
    ("X04", "Other – Paediatric Services", 255),
    ("X04", "Other – Paediatric Services", 256),
    ("X04", "Other – Paediatric Services", 257),
    ("X04", "Other – Paediatric Services", 258),
    ("X04", "Other – Paediatric Services", 259),
    ("X04", "Other – Paediatric Services", 260),
    ("X06", "Other - Other Services", 920),
]

schema = ["RTT specialty code", "Specialty", "TFC"]


def simulate_waiting_list_data(my_size=1000, seed=1814):
    np.random.seed(seed)

    # Create patient IDs
    epp_pid = list(range(1, my_size+1))

    # Generate ages with specified probabilities
    age_probs = np.concatenate([np.repeat(0.1, 14*my_size), np.repeat(0.2, 14*my_size), np.repeat(0.3, 31*my_size), np.repeat(0.1, 14*my_size)])
    ages = np.random.choice(np.tile(np.arange(18, 91), my_size), my_size, p=age_probs/age_probs.sum())
    
    #add Sex
    Sex = np.random.choice(["F", "M"], size=len(ages), p=[0.51, 0.49])

    #add ltc
    ltc = np.where(
        ages < 60, np.random.randint(0, 6, size=len(ages)),
        np.random.choice(np.arange(11), size=len(ages), p=[0.05, 0.05, 0.1, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05])
    )

    #Frailty_Index
    frailty_index = np.where(
    ages > 70,
    np.random.choice([0, 1, 2, 3, 4, 5], size=len(ages), p=[0.05, 0.05, 0.1, 0.3, 0.3, 0.2]),
    np.random.choice([0, 1, 2, 3], size=len(ages), p=[0.4, 0.3, 0.2, 0.1])
    )

    # Generate deprivation levels
    ndl_imd_quantile = np.random.choice(np.tile(np.arange(1, 6), my_size), my_size)

    # Generate clock starts within specified date range
    date_range = pd.date_range(start="2022-01-01", end="2023-11-30")
    epp_rtt_start_date = np.random.choice(date_range, my_size)

    # Generate epp_referral_priority
    referral_options = ["Routine", "Urgent", "Cancer", "Unknown"]
    epp_referral_priority = np.random.choice(referral_options, my_size, p=[0.6, 0.25, 0.1, 0.05])

    # Generate primary diagnosis
    diagnosis_options = ["J18", "I10", "E11", "M54", "Other"]
    sus_primary_diagnosis = np.random.choice(diagnosis_options, my_size)

    # Generate ethnicity
    ethnicity_options = ["White", "Asian", "Black", "Mixed", "Other", "Unknown"]
    ndl_ethnicity = np.random.choice(ethnicity_options, my_size)

    # Set constant epp_tfc
    epp_tfc = ["X01"] * my_size

    # Generate healthcare use propensity scores
    healthcare_use_propensity = np.random.normal(0, 1, my_size)

    # Create the DataFrame
    time_invariant_df = pd.DataFrame({
        "epp_pid": epp_pid,
        "ages": ages,
        "Sex": Sex,
        "ndl_ethnicity": ndl_ethnicity,
        "ndl_imd_quantile": ndl_imd_quantile,
        "epp_rtt_start_date": epp_rtt_start_date,
        "healthcare_use_propensity": healthcare_use_propensity,
        "epp_referral_priority": epp_referral_priority,
        "epp_tfc": epp_tfc,
        "sus_primary_diagnosis": sus_primary_diagnosis,
        "ltc": ltc, 
        "Frailty_Index": frailty_index  
    })

    time_invariant_df["REG_DATE_OF_DEATH"] = None

    # Define wait time probability distributions
    wait_time_probs_1 = np.concatenate([np.repeat(0.3, 126*my_size), np.repeat(0.3, 126*my_size), np.repeat(0.2, 127*my_size), np.repeat(0.15, 126*my_size), np.repeat(0.05, 126*my_size)])
    wait_time_probs_2 = np.concatenate([np.repeat(0.2, 126*my_size), np.repeat(0.2, 126*my_size), np.repeat(0.3, 127*my_size), np.repeat(0.2, 126*my_size), np.repeat(0.1, 126*my_size)])

    # Assign wait times based on deprivation levels
    time_invariant_df['wait_times'] = np.where(
        time_invariant_df['ndl_imd_quantile'].isin([3, 4, 5]),
        np.random.choice(np.repeat(np.arange(15, 646), my_size), my_size, p=wait_time_probs_1/wait_time_probs_1.sum()),
        np.random.choice(np.repeat(np.arange(15, 646),my_size), my_size, p=wait_time_probs_2/wait_time_probs_2.sum())
    )

    # Calculate clock stops
    time_invariant_df['epp_rtt_end_date'] = time_invariant_df['epp_rtt_start_date'] + pd.to_timedelta(time_invariant_df['wait_times'], unit='D')

    # Expand the DataFrame by repeating each row 851 times
    expanded_df = time_invariant_df.loc[time_invariant_df.index.repeat(851)].reset_index(drop=True) 

    # Generate date range for time-variant data
    date1 = pd.to_datetime("2022-01-01")
    date2 = pd.to_datetime("2024-04-30")
    date_range_full = pd.date_range(date1, date2)

    days = pd.DataFrame({'days': np.tile(date_range_full, my_size)})

    # Combine the dataframes
    full_time_df = pd.concat([expanded_df.reset_index(drop=True), days.reset_index(drop=True)], axis=1)

    full_time_df = spark.createDataFrame(full_time_df)

    full_time_df = full_time_df.withColumn("days_since_clock_start", datediff(col("days"), col("epp_rtt_start_date"))).withColumn("treated", when((col("wait_times") > 126) & (col("days_since_clock_start") < col("wait_times")), 1).otherwise(0))

    # Convert to Pandas DataFrame for easier manipulation
    full_time_pd = full_time_df.select("ages", "treated", "healthcare_use_propensity").toPandas()

    # Define the correlation matrix
    correlation_matrix = np.array([
        [1, 0.2, 0.2],
        [0.2, 1, 0.5],
        [0.2, 0.5, 1]
    ])

    # Generate multivariate normal samples
    mean = [0, 0, 0]
    samples = np.random.multivariate_normal(mean, correlation_matrix, size=len(full_time_pd))

    # Convert samples to probabilities using the normal CDF
    healthcare_use_probs = norm.cdf(samples)

    # Calculate the quantiles using the gamma distribution
    use_gam = gamma.ppf(healthcare_use_probs[:, 0], a=4, scale=1)

    # Add the quantiles back to the Pandas DataFrame
    a = np.round(use_gam, 0)
    use_gam_spark_df = spark.createDataFrame(pd.DataFrame(a, columns=["healthcare_use"]))

    # Create a sequential index column for both DataFrames
    window_spec = Window.orderBy(F.lit(1))

    # Add sequential index to `use_gam_spark_df`
    use_gam_spark_df = use_gam_spark_df.withColumn("row_index", F.row_number().over(window_spec))

    # Add sequential index to `full_time_df`
    full_time_df = full_time_df.withColumn("row_index", F.row_number().over(window_spec))

    # Join the DataFrames on the index column
    full_time_df = full_time_df.join(use_gam_spark_df, on="row_index").drop("row_index")

    # List of new columns to create
    target_cols = [
        "gp_healthcare_use",
        "gp_Total_cost",
        "u111_healthcare_use",
        "u999_healthcare_use",
        "u00H_healthcare_use",
        "ae_healthcare_use",
        "ae_Total_cost",
        "nel_healthcare_use",
        "nel_Total_cost",
        "el_healthcare_use",
        "el_Total_cost",
        "op_healthcare_use",
        "op_Total_cost",
        "all_pres_count",
        "antib_pres_count",
        "antipres_pres_count",
        "pain_pres_count", 
        "sick_note"
    ]

    # Loop over and create each new column based on 'healthcare_use' with slight noise
    for i, col_name in enumerate(target_cols):
        full_time_df = full_time_df.withColumn(
            col_name,
            when(
                col("healthcare_use") + (floor(rand() * 5) - 2) < 0,
                0
            ).otherwise(
                (col("healthcare_use") + (floor(rand() * 5) - 2))
            ).cast("int")
        )
    # Add change of teh priority
    full_time_df = full_time_df.withColumn(
    "wlmds_priority_set_last",
    expr("""
        CASE
            WHEN rand() < 0.25 THEN '1.1|2'
            WHEN rand() < 0.5 THEN '2'
            WHEN rand() < 0.75 THEN '1|1'
            ELSE '2|2'
        END
    """)
    )

    
    full_time_df = full_time_df.withColumn(
        "ndl_age_band",
        when((col("ages") > 0) & (col("ages") <= 10), "<= 10")
        .when((col("ages") >= 11) & (col("ages") <= 17), "11-17")
        .when((col("ages") >= 18) & (col("ages") <= 24), "18-24")
        .when((col("ages") >= 25) & (col("ages") <= 34), "25-34")
        .when((col("ages") >= 35) & (col("ages") <= 44), "35-44")
        .when((col("ages") >= 45) & (col("ages") <= 54), "45-54")
        .when((col("ages") >= 55) & (col("ages") <= 64), "55-64")
        .when((col("ages") >= 65) & (col("ages") <= 74), "65-74")
        .when((col("ages") >= 75) & (col("ages") <= 84), "75-84")
        .when(col("ages") >= 85, "84+")
        .otherwise("unknown")
    )
    full_time_df = full_time_df.withColumn(
        "ndl_ltc",
        when(col("ltc") == 0, "No LTCs")
        .when(col("ltc") == 1, "Single LTC")
        .when(col("ltc") == 2, "Comorbidities")
        .when(col("ltc") > 2, "Multimorbidities")
        .otherwise("unknown")
    )
    full_time_df = full_time_df.withColumn(
        "Frailty_level",
        when(col("Frailty_Index") < 2, "Not Frail")
        .otherwise("Frail")
    )


    # Ensure epp_rtt_start_date is a DateType column
    full_time_df = full_time_df.withColumn("epp_rtt_start_date", to_date("epp_rtt_start_date"))
    full_time_df = full_time_df.withColumn("epp_rtt_end_date", to_date("epp_rtt_end_date"))

    return full_time_df


def filter_data_function_NDL(df, intervention_group, control_group, time_procedure_received, length_of_recovery):
    # Define start and end of the reference period
    start_of_reference_period = time_procedure_received + length_of_recovery
    end_of_reference_period = start_of_reference_period + 180



    # Filter and add time periods
    df_filtered = (
        df.filter((col("group") == intervention_group) | (col("group") == control_group))
        .filter(col("days_since_clock_start") <= end_of_reference_period)
        .withColumn(
            "time_period",
            when(col("days_since_clock_start") < time_procedure_received, 1)
            .when((col("days_since_clock_start") >= time_procedure_received) &
                  (col("days_since_clock_start") < start_of_reference_period), 100)
            .otherwise(0)
        )
    )

    # Calculate max time covered and filter
    window = Window.partitionBy("epp_pid", "epp_tfc", "epp_rtt_start_date")
    df_filtered = (
        df_filtered.withColumn("max_time_covered", spark_max("days_since_clock_start").over(window))
        .filter(col("max_time_covered") == end_of_reference_period)
    )

    return df_filtered


def filter_data_function_HF(df, intervention_group, control_group):  
    # Filter and add time periods
    df_filtered = (
        df.filter((col("group") == intervention_group) | (col("group") == control_group))
        .filter(col("days_since_clock_start") <= col("follow_up_end_days"))
        .withColumn(
            "time_period",
            when(col("days_since_clock_start") < col("wt"), 1)
            .when((col("days_since_clock_start") >= col("wt")) &
                  (col("days_since_clock_start") < col("washout_period_end_days")), 100)
            .otherwise(0)
        )
    )
    #print("Initial control count:",     df_filtered.filter((col("group") == control_group)).count())
    # Calculate max time covered and filter
    window = Window.partitionBy("epp_pid", "epp_tfc", "epp_rtt_start_date")
    df_filtered = (
        df_filtered.withColumn("max_time_covered", spark_max("days_since_clock_start").over(window))
        .filter(col("max_time_covered") == col("follow_up_end_days"))
    )
    #print("Initial control count:", df_filtered.filter((col("group") == control_group)).count())
    return df_filtered

def group_data_function_HF(df_filtered, intervention_group, control_group, var_hc):
    # Group and aggregate
    df_grouped_by_time_period = (
        df_filtered.groupBy("epp_pid", "epp_tfc", "epp_rtt_start_date", "time_period", "group", "max_time_covered")
        .agg(F.sum(col(var_hc)).alias("total_hc_use"))  # Use F.sum and col(var_hc)
        .withColumn(
            "treated",
            when(col("group") == control_group, 0).when(col("group") == intervention_group, 1)
        )
        .filter(col("time_period") != 100)
        .withColumn(
            "total_hc_use", 
            when(col("total_hc_use").isNull(), 0).otherwise(col("total_hc_use"))
        )
        .withColumn(
            "avg_weekly_use", 
            (col("total_hc_use") / col("max_time_covered").cast("double")) * 7
        )
    )
    return df_grouped_by_time_period

def do_placebo_test_HF(filtered_df, intervention_group, control_group):
    # Create placebo period
    more_filtered = (
        filtered_df.filter(col("time_period") == 0)
        .withColumn(
            "placebo_period",
            when(col("days_since_clock_start") <= col("wt") + (col("wt")/2), 0).otherwise(1)
        )
    )

    # Group and aggregate
    df_grouped_by_time_period = (
        more_filtered.groupBy("epp_pid", "placebo_period", "group",  "max_time_covered")
        .agg({"healthcare_use": "sum"})
        .withColumnRenamed("sum(healthcare_use)", "total_hc_use")
        .withColumn(
            "treated",
            when(col("group") == control_group, 0).when(col("group") == intervention_group, 1)
        )
    )

   # formula = 'total_hc_use ~ i(placebo_period, treated, ref=0) | epp_pid + placebo_period'
    ## Run the fixed effects OLS regression
    #fe_ols_30weeks = pf.feols(formula, df_grouped_by_time_period.toPandas())


    #return(fe_ols_30weeks.summary())
    return df_grouped_by_time_period

def filter_data_function_HF_all_cohort(df):  
    # Filter and add time periods
    df_filtered = (
        df.filter(col("days_since_clock_start") <= col("follow_up_end_days"))
        .withColumn(
            "time_period",
            when(col("days_since_clock_start") < col("wt"), 1)
            .when((col("days_since_clock_start") >= col("wt")) &
                  (col("days_since_clock_start") < col("washout_period_end_days")), 100)
            .otherwise(0)
        )
    )
    #print("Initial control count:",     df_filtered.filter((col("group") == control_group)).count())
    # Calculate max time covered and filter
    window = Window.partitionBy("epp_pid", "epp_tfc", "epp_rtt_start_date")
    df_filtered = (
        df_filtered.withColumn("max_time_covered", spark_max("days_since_clock_start").over(window))
        .filter(col("max_time_covered") == col("follow_up_end_days"))
    )
    #print("Initial control count:", df_filtered.filter((col("group") == control_group)).count())
    return df_filtered


def totals_table_function_HF(df, control_group, intervention_group):
    """
    Generates a totals table comparing healthcare use between control and intervention groups.
    """
    # Group by time_period and group, calculate aggregates
    df_group = df.groupBy("time_period", "group").agg(
        F.sum("total_hc_use").alias("hc_use"),
        F.sum(F.col("max_time_covered").cast("double")).alias("days_covered"),
        #F.countDistinct("epp_pid").alias("nr_epp_pid"),
        F.countDistinct("rpp_pid", "epp_tfc", "epp_rtt_start_date").alias("nr_unique_pathways")
    )

    # Collect results into a list of Rows
    df_group_list = df_group.collect()

    # Convert to a dictionary for easier lookup
    group_data = {(row["time_period"], row["group"]): row for row in df_group_list}

    # Construct table structure
    table_fill = {
        "metrics": [
            "Healthcare use reference period", 
            "Healthcare use intervention period", 
            "Excess use", 
            "person-weeks"
        ],
        control_group: [
            group_data.get((0, control_group), {}).hc_use if (0, control_group) in group_data else 0,
            group_data.get((1, control_group), {}).hc_use if (1, control_group) in group_data else 0,
            None,  # Excess use calculated later
            group_data.get((0, control_group), {}).days_covered / 7 if (0, control_group) in group_data else 0
        ],
        intervention_group: [
            group_data.get((0, intervention_group), {}).hc_use if (0, intervention_group) in group_data else 0,
            group_data.get((1, intervention_group), {}).hc_use if (1, intervention_group) in group_data else 0,
            None,  # Excess use calculated later
            group_data.get((0, intervention_group), {}).days_covered / 7 if (0, intervention_group) in group_data else 0
        ]
    }

    # Calculate excess use
    control_excess = (
        (group_data.get((1, control_group), {}).hc_use if (1, control_group) in group_data else 0) -
        (group_data.get((0, control_group), {}).hc_use if (0, control_group) in group_data else 0)
    )
    intervention_excess = (
        (group_data.get((1, intervention_group), {}).hc_use if (1, intervention_group) in group_data else 0) -
        (group_data.get((0, intervention_group), {}).hc_use if (0, intervention_group) in group_data else 0)
    )
    table_fill[control_group][2] = control_excess
    table_fill[intervention_group][2] = intervention_excess - control_excess

    # Convert to Pandas DataFrame
    table_fill_df = pd.DataFrame(table_fill)
    
    return table_fill_df

def totals_table_function_HF2(df, control_group, intervention_group):
    # Summarize the DataFrame by group and time period
    df_group = df.groupBy("time_period", "group").agg(
        F.sum("total_hc_use").alias("hc_use"),
        F.sum(F.col("max_time_covered").cast("double")).alias("days_covered"),
        F.countDistinct("epp_pid").alias("no_patients"),
       # F.countDistinct("rpp_pid", "epp_tfc", "epp_rtt_start_date").alias("nr_unique_pathways")
    )

   # Initialize schema for the resulting DataFrame
    schema = StructType([
        StructField("metrics", StringType(), True),
        StructField(control_group, DoubleType(), True),
        StructField(intervention_group, DoubleType(), True)
    ])

# Calculate the required metrics and ensure they are floats
    control_reference_use = float(df_group.filter((F.col("time_period") == 0) & (F.col("group") == control_group)).select("hc_use").first()[0])
    control_intervention_use = float(df_group.filter((F.col("time_period") == 1) & (F.col("group") == control_group)).select("hc_use").first()[0])
    control_person_weeks = float(df_group.filter((F.col("time_period") == 0) & (F.col("group") == control_group)).select((F.col("days_covered") / 7)).first()[0])
    #control_pathwayay_count = float(df_group.filter((F.col("time_period") == 0) & (F.col("group") == control_group)).select("nr_unique_pathways").first()[0])
    
    intervention_reference_use = float(df_group.filter((F.col("time_period") == 0) & (F.col("group") == intervention_group)).select("hc_use").first()[0])
    intervention_intervention_use = float(df_group.filter((F.col("time_period") == 1) & (F.col("group") == intervention_group)).select("hc_use").first()[0])
    intervention_person_weeks = float(df_group.filter((F.col("time_period") == 0) & (F.col("group") == intervention_group)).select((F.col("days_covered") / 7)).first()[0])
    #intervention_patwayay_count = float(df_group.filter((F.col("time_period") == 0) & (F.col("group") == intervention_group)).select("nr_unique_pathways").first()[0])
    
    excess_use = float(
        (intervention_intervention_use - intervention_reference_use) -
        (control_intervention_use - control_reference_use)
    )

    # Fill the table
    table_fill = spark.createDataFrame([
        ("Healthcare use reference period", control_reference_use, intervention_reference_use),
        ("Healthcare use intervention period", control_intervention_use, intervention_intervention_use),
        ("Excess use", None, excess_use),
        ("person-weeks", control_person_weeks, intervention_person_weeks),
        #("patways", control_pathwayay_count, intervention_patwayay_count)
    ], schema)

    return table_fill

#person week average for hc use
def totals_table_function_HF_avg(df, control_group, intervention_group):
    # Summarize the DataFrame by group and time period
    df_group = df.groupBy("time_period", "group").agg(
        F.sum("total_hc_use").alias("hc_use"),
        F.sum(F.col("max_time_covered").cast("double")).alias("days_covered"),
        F.countDistinct("epp_pid").alias("no_patients")
    )

   # Initialize schema for the resulting DataFrame
    schema = StructType([
        StructField("metrics", StringType(), True),
        StructField(control_group, DoubleType(), True),
        StructField(intervention_group, DoubleType(), True)
    ])

# Calculate the required metrics and ensure they are floats
    control_person_weeks = float(df_group.filter((F.col("time_period") == 0) & (F.col("group") == control_group)).select((F.col("days_covered") / 7)).first()[0])
    control_reference_use = float(df_group.filter((F.col("time_period") == 0) & (F.col("group") == control_group)).select("hc_use").first()[0])/control_person_weeks
    control_intervention_use = float(df_group.filter((F.col("time_period") == 1) & (F.col("group") == control_group)).select("hc_use").first()[0])/control_person_weeks
    #control_person_weeks = float(df_group.filter((F.col("time_period") == 0) & (F.col("group") == control_group)).select((F.col("days_covered") / 7)).first()[0])
    
    intervention_person_weeks = float(df_group.filter((F.col("time_period") == 0) & (F.col("group") == intervention_group)).select((F.col("days_covered") / 7)).first()[0])
    intervention_reference_use = float(df_group.filter((F.col("time_period") == 0) & (F.col("group") == intervention_group)).select("hc_use").first()[0])/intervention_person_weeks
    intervention_intervention_use = float(df_group.filter((F.col("time_period") == 1) & (F.col("group") == intervention_group)).select("hc_use").first()[0])/intervention_person_weeks
    
    
    excess_use = float(
        (intervention_intervention_use - intervention_reference_use) -
        (control_intervention_use - control_reference_use)
    )

    # Fill the table
    table_fill = spark.createDataFrame([
        ("Healthcare use reference period", control_reference_use, intervention_reference_use),
        ("Healthcare use intervention period", control_intervention_use, intervention_intervention_use),
        ("Excess use", None, excess_use),
        ("person-weeks", control_person_weeks, intervention_person_weeks)
    ], schema)

    return table_fill


def run_fixed_effects_HF(df):
    formula = 'avg_weekly_use ~ i(time_period, treated, ref=0) | epp_pid + time_period'
    ##run the fixed effects OLS regression
    fe_ols_30weeks = pf.feols(formula, df.toPandas())
    return(fe_ols_30weeks)

def run_fixed_effects_HF_total(df):
    #df=df.withColumn("total_hc_use_log", log("total_hc_use"))
    formula = 'total_hc_use  ~ i(time_period, treated, ref=0) | epp_pid + time_period'
    ##run the fixed effects OLS regression
    fe_ols_30weeks = pf.feols(formula, df.toPandas())
    return(fe_ols_30weeks)



def calculate_wait_band_distribution_characteristics(df, input_cols):
    """
    Calculate the distribution of wait bands for the given input columns,
    including counts and percentages for each wait band value, and total counts
    for each input column value, renaming null values in the input columns to "unknown".
 
    Args:
        df (DataFrame): Input DataFrame.
        input_cols (list): List of column names for the input grouping.
 
    Returns:
        DataFrame: A DataFrame containing input column values, total counts,
                   and affixed columns with each wait band value: count and percentage.

    """

    personal_ch = df.groupBy(input_cols + ["epp_pid", "epp_tfc", "epp_rtt_start_date", "group"]).agg(count("*").alias("count"))

    grouped_counts_list = []

    for var in input_cols:
        personal_ch = personal_ch.withColumn(var, when(col(var).isNull(), "unknown").otherwise(col(var)))
        grouped_counts = personal_ch.groupBy(var).pivot("group").agg(count("*").alias("count")).withColumnRenamed(var, "value")
        total_counts = personal_ch.groupBy(var).agg(count("*").alias("total_count")).withColumnRenamed(var, "value")
        grouped_counts = grouped_counts.join(total_counts, on="value", how="left")
        for c in grouped_counts.columns:
            if c not in ["value", "total_count"]:
                grouped_counts = grouped_counts.withColumn(f"{c}_percentage", (coalesce(col(c), lit(0)) / col("total_count")) * 100)
                grouped_counts = grouped_counts.withColumn(c, round(coalesce(col(c), lit(0)), 2))
                grouped_counts = grouped_counts.withColumn(f"{c}_percentage", round(col(f"{c}_percentage"), 2))
        grouped_counts = grouped_counts.withColumn("Variable", lit(var))
        grouped_counts_list.append(grouped_counts)

    grouped_counts = reduce(lambda df1, df2: df1.unionByName(df2), grouped_counts_list)
        
    return grouped_counts    

def calculate_wait_band_distribution(df, input_cols):
    """
    Calculate the distribution of wait bands for the given input columns,
    including counts and percentages for each wait band value, and total counts
    for each input column value, renaming null values in the input columns to 0.
 
    Args:
        df (DataFrame): Input DataFrame.
        input_cols (list): List of column names for the input grouping.
 
    Returns:
        DataFrame: A DataFrame containing input column values, total counts,
                   and affixed columns with each wait band value: count and percentage.

    """

    personal_ch = df.groupBy(input_cols+["epp_pid", "group"]).agg(count("*").alias("count"))

    grouped_counts_list = []

    for var in input_cols:
        personal_ch = personal_ch.withColumn(var, when(col(var).isNull(), 0).otherwise(col(var)))
        grouped_counts = personal_ch.groupBy(var).pivot("group").agg(count("*").alias("count")).withColumnRenamed(var, "value")
        total_counts = grouped_counts.select([sum(col(c)).alias(c) for c in grouped_counts.columns if c != "value"])
        for c in grouped_counts.columns:
            if c != "value":
                grouped_counts = grouped_counts.withColumn(f"{c}_percentage", (col(c) / total_counts.first()[c]) * 100)
                grouped_counts = grouped_counts.withColumn(c, round(col(c), 2))
                grouped_counts = grouped_counts.withColumn(f"{c}_percentage", round(col(f"{c}_percentage"), 2))
        grouped_counts = grouped_counts.withColumn("Variable", lit(var))
        grouped_counts_list.append(grouped_counts)

    grouped_counts = reduce(lambda df1, df2: df1.unionByName(df2), grouped_counts_list)
        
    return grouped_counts

# Objective 1-3 Functions
def rename_and_add_column(df, new_name):
    old_columns = df.columns
    new_columns = [new_name] + old_columns[1:]
    df = df.toDF(*new_columns)
    return df.withColumn("original_column", lit(old_columns[0]))

def rename_and_add_column_cross(df, new_name, new_s_name):
    old_columns = df.columns  # Get original column names
    first_col_name = old_columns[0]  # Save the first column name
    second_col_name = old_columns[1]  # Save the second column name   
    new_columns = [new_name, new_s_name] + old_columns[2:] 
    
    # Rename dataframe columns
    df = df.toDF(*new_columns)
    
  
    # Add new columns storing the original column names as constant values
    df = df.withColumn("First_variable", lit(first_col_name))
    df = df.withColumn("Second_variable", lit(second_col_name))
    return df


# TABLE 1.1 FUNCTION
def calculate_wait_band_distribution(df: DataFrame, input_col: str) -> DataFrame:
    """
    Calculate the distribution of wait bands for the given input column,
    including counts and percentages for each wait band value, and total counts
    for each input column value, renaming null values in the input column to "unknown".

    Args:
        df (DataFrame): Input DataFrame.
        input_col (str): Column name for the input grouping.

    Returns:
        DataFrame: A DataFrame containing input column values, total counts,
                   and affixed columns with each wait band value: count and percentage.
    """
    wait_band_col = "ndl_wait_band"
    
    df = df.withColumn(input_col, when(col(input_col).isNull(), "unknown").otherwise(col(input_col)))
    
    wait_band_counts = df.groupBy(input_col, wait_band_col).agg(
        count("*").alias("count")
    )
    
    input_totals = df.groupBy(input_col).agg(
        count("*").alias("total_count")
    )
    
    result = wait_band_counts.join(input_totals, on=input_col, how="left")
    
    result = result.withColumn(
        "percentage",
        round((col("count") / col("total_count")) * 100, 2)
    )
    
    result = result.withColumn(
        "count",
        when(col("count") < 5, lit("<5")).otherwise(col("count"))
    ).withColumn(
        "percentage",
        when(col("percentage") < 5, lit("<5")).otherwise(col("percentage"))
    )
    
    # Pivot the table
    result_pivot = result.groupBy(input_col, "total_count").pivot(wait_band_col).agg(
        first("count").alias("count"), first("percentage").alias("percentage")
    )
    
    return result_pivot


# TABLE 1.2 FUNCTION
def calculate_cross_wait_band_distribution(df: DataFrame, input_col1: str, input_col2: str) -> DataFrame:
    """
    Calculate the distribution of wait bands for the given input columns,
    including counts and percentages for each wait band value, and total counts
    for each combination of input column values, renaming null values in the input columns to "unknown".

    Args:
        df (DataFrame): Input DataFrame.
        input_col1 (str): First column name for the input grouping.
        input_col2 (str): Second column name for the input grouping.

    Returns:
        DataFrame: A DataFrame containing input column values, total counts,
                   and affixed columns with each wait band value: count and percentage.
    """
    wait_band_col = "ndl_wait_band"
    
    df = df.withColumn(input_col1, when(col(input_col1).isNull(), "unknown").otherwise(col(input_col1)))
    df = df.withColumn(input_col2, when(col(input_col2).isNull(), "unknown").otherwise(col(input_col2)))
    
    # Group by input columns and wait band column to get counts
    wait_band_counts = df.groupBy(input_col1, input_col2, wait_band_col).agg(
        count("*").alias("count")
    )
    
    # Calculate the total count for each combination of input column values
    input_totals = df.groupBy(input_col1, input_col2).agg(
        count("*").alias("total_count")
    )
    
    result = wait_band_counts.join(input_totals, on=[input_col1, input_col2], how="left")
    
    result = result.withColumn(
        "percentage",
        round((col("count") / col("total_count")) * 100, 2)
    )
    
    result = result.withColumn(
        "count",
        when(col("count") < 5, lit("<5")).otherwise(col("count"))
    ).withColumn(
        "percentage",
        when(col("percentage") < 5, lit("<5")).otherwise(col("percentage"))
    )
    
    # Pivot the table
    result_pivot = result.groupBy(input_col1, input_col2, "total_count").pivot(wait_band_col).agg(
        first("count").alias("count"), first("percentage").alias("percentage")
    )
    
    return result_pivot


# TABLE 1.3 FUNCTION
def calculate_wait_length_statistics(
    df: DataFrame, 
    wait_length_col: str, 
    investigation_col: str
) -> DataFrame:
    """
    Calculate the mean, median, standard deviation, and interquartile range of wait lengths 
    grouped by the investigation column, handling null values and converting small values.

    Args:
        df (DataFrame): Input DataFrame.
        wait_length_col (str): Column name for the wait length.
        investigation_col (str): Column name for the investigation grouping.

    Returns:
        DataFrame: A DataFrame containing mean, median, standard deviation, and interquartile range
                   for each investigation group.
    """
    df = df.withColumn(investigation_col, coalesce(col(investigation_col), lit("unknown")))
    
    stats_df = df.groupBy(investigation_col).agg(
        round(mean(col(wait_length_col)), 2).alias("mean"),
        round(percentile_approx(col(wait_length_col), 0.5), 2).alias("median"),
        round(stddev(col(wait_length_col)), 2).alias("SD"),
        # Calculate IQR: P75 - P25
        round((percentile_approx(col(wait_length_col), 0.75) - 
               percentile_approx(col(wait_length_col), 0.25)), 2).alias("IQR")
    )
    
    stats_df = stats_df.select(
        col(investigation_col),
        when(col("mean") < 5, lit("<5")).otherwise(col("mean")).alias("mean"),
        when(col("median") < 5, lit("<5")).otherwise(col("median")).alias("median"),
        when(col("SD") < 5, lit("<5")).otherwise(col("SD")).alias("SD"),
        when(col("IQR") < 5, lit("<5")).otherwise(col("IQR")).alias("IQR")
    )
    
    return stats_df

# TABLE 1.4 FUNCTION
def calculate_wait_length_statistics_cross(
    df: DataFrame, 
    wait_length_col: str, 
    investigation_col1: str, 
    investigation_col2: str
) -> DataFrame:
    """
    Calculate the mean, median, standard deviation, and interquartile range of wait lengths 
    grouped by two investigation columns, converting null values to "unknown" and 
    replacing calculated values less than 5 with "<5".

    Args:
        df (DataFrame): Input DataFrame.
        wait_length_col (str): Column name for the wait length.
        investigation_col1 (str): First column name for the investigation grouping.
        investigation_col2 (str): Second column name for the investigation grouping.

    Returns:
        DataFrame: A DataFrame containing mean, median, standard deviation, and interquartile range
                   for each cross-tabulated group of investigation columns.
    """
    
    df = df.withColumn(investigation_col1, when(col(investigation_col1).isNull(), "unknown").otherwise(col(investigation_col1)))
    df = df.withColumn(investigation_col2, when(col(investigation_col2).isNull(), "unknown").otherwise(col(investigation_col2)))
    
    stats_df = df.groupBy(investigation_col1, investigation_col2).agg(
        round(mean(col(wait_length_col)), 2).alias("mean"),
        round(percentile_approx(col(wait_length_col), 0.5), 2).alias("median"),
        round(stddev(col(wait_length_col)), 2).alias("SD"),
        # Calculate IQR: P75 - P25
        round((percentile_approx(col(wait_length_col), 0.75) - 
         percentile_approx(col(wait_length_col), 0.25)), 2).alias("IQR")
    )
    
    stats_df = stats_df.withColumn("mean", when(col("mean") < 5, lit("<5")).otherwise(col("mean")))
    stats_df = stats_df.withColumn("median", when(col("median") < 5, lit("<5")).otherwise(col("median")))
    stats_df = stats_df.withColumn("SD", when(col("SD") < 5, lit("<5")).otherwise(col("SD")))
    stats_df = stats_df.withColumn("IQR", when(col("IQR") < 5, lit("<5")).otherwise(col("IQR")))
    
    return stats_df


# TABLE 2.1 FUNCTION
def objective2_table(df: DataFrame, investigation_col: str) -> DataFrame:
    """
    Calculates counts and percentages of various categories for the given investigation column.

    Args:
        df (DataFrame): Input DataFrame.
        investigation_col (str): Column name for the investigation grouping (e.g., "Specialty").

    Returns:
        DataFrame: A DataFrame containing counts and percentages for each category grouped by the investigation column.
    """

    df = df.withColumn(investigation_col, coalesce(col(investigation_col), lit("unknown")))

    conditions = [
        (df.wlmds_status == 30) & (col("wlmds_type_changes_last").contains("IRTT")),    # treatment + admitted                  = 30 & IRTT
        (df.wlmds_status == 30) & (~col("wlmds_type_changes_last").contains("IRTT")),   # treatment + not admitted              = 31 & no IRTT
        (df.wlmds_status == 31) | (df.wlmds_status == 32),                              # non treatment, start monitoring       = 31 or 32
        df.wlmds_status == 35,                                                          # non treatment, patient declines       = 35
        df.wlmds_status == 34,                                                          # non treatment, decision to not treat  = 34
        df.wlmds_status == 33,                                                          # non treatment, DNA                    = 33
        (~df.wlmds_status.isin(30, 31, 32, 33, 34, 35, 36, 99)),                        # non treatment, other                  = not 30, 31, 32, 33, 34, 35, 36, 99
        df.wlmds_status == 36,                                                          # non treatment, death                  = 36 
        df.wlmds_status == 99,                                                          # unknown                               = 99
        (df.wlmds_status == 99) & (col("wlmds_end_date_type_last") == "week_end_date")  # unknown, imputed end                  = 99 and week_end_date
    ]

    labels = [
        "treatment_admitted",
        "treatment_non_admitted",
        "non_treatment_monitoring",
        "non_treatment_patient_declines",
        "non_treatment_decision_not_to_treat",
        "non_treatment_dna",
        "non_treatment_other",
        "non_treatment_death",
        "null",
        "null_imputed_end"
    ]

    for cond, label in zip(conditions, labels):
        df = df.withColumn(label, when(cond, 1).otherwise(0))

    agg_exprs = [count(when(col(label) == 1, 1)).alias(f"count_{label}") for label in labels]
    total_expr = [count(lit(1)).alias("total_count")]

    df_aggregated = df.groupBy(investigation_col).agg(*(agg_exprs + total_expr))

    for label in labels:
        df_aggregated = df_aggregated.withColumn(
            f"percentage_{label}", 
            round((col(f"count_{label}") / col("total_count")) * 100, 2)
        )

    for label in labels:
        df_aggregated = df_aggregated.withColumn(
            f"count_{label}", 
            when(col(f"count_{label}") < 5, lit("<5")).otherwise(col(f"count_{label}"))
        ).withColumn(
            f"percentage_{label}", 
            when(col(f"percentage_{label}") < 5, lit("<5")).otherwise(col(f"percentage_{label}"))
        )

    return df_aggregated


def calculate_wait_band_distribution_report(df: DataFrame) -> DataFrame:
    """
    Calculate the distribution of wait bands, including counts and percentages
    for each wait band value, renaming null values in the wait band column to "unknown".

    Args:
        df (DataFrame): Input DataFrame.

    Returns:
        DataFrame: A DataFrame containing wait band values, counts, and percentages.
    """
    wait_band_col = "ndl_wait_band_granular_c1"
    
    df = df.withColumn(wait_band_col, when(col(wait_band_col).isNull(), "unknown").otherwise(col(wait_band_col)))
    
    wait_band_counts = df.groupBy(wait_band_col).agg(
        count("*").alias("count")
    )
    
    total_count = df.count()
    
    result = wait_band_counts.withColumn(
        "percentage",
        round((col("count") / total_count) * 100, 2)
    )
    
    result = result.withColumn(
        "percentage",
        when(col("count") < 5, lit("<5")).otherwise(col("percentage"))    
    ).withColumn(
        "count",
        when(col("count") < 5, lit("<5")).otherwise(col("count"))     
    )
    
    return result

def calculate_wait_band_distribution_waiting_group(df, input_cols):
    """
    Calculate the distribution of wait bands for the given input columns,
    including counts and percentages for each wait band value, and total counts
    for each input column value, renaming null values in the input columns to "unknown".
 
    Args:
        df (DataFrame): Input DataFrame.
        input_cols (list): List of column names for the input grouping.
 
    Returns:
        DataFrame: A DataFrame containing input column values, total counts,
                   and affixed columns with each wait band value: count and percentage.

    """

    personal_ch = df.groupBy(input_cols+["epp_pid", "epp_tfc", "epp_rtt_start_date", "group"]).agg(count("*").alias("count"))

    grouped_counts_list = []

    for var in input_cols:
        personal_ch = personal_ch.withColumn(var, when(col(var).isNull(), "unknown").otherwise(col(var)))
        grouped_counts = personal_ch.groupBy(var).pivot("group").agg(count("*").alias("count")).withColumnRenamed(var, "value")
        total_counts = grouped_counts.select([sum(col(c)).alias(c) for c in grouped_counts.columns if c != "value"])
        for c in grouped_counts.columns:
            if c != "value":
                grouped_counts = grouped_counts.withColumn(f"{c}_percentage", (coalesce(col(c), lit(0)) / total_counts.first()[c]) * 100)
                grouped_counts = grouped_counts.withColumn(c, round(coalesce(col(c), lit(0)), 2))
                grouped_counts = grouped_counts.withColumn(f"{c}_percentage", round(col(f"{c}_percentage"), 2))
        grouped_counts = grouped_counts.withColumn("Variable", lit(var))
        grouped_counts_list.append(grouped_counts)

    grouped_counts = reduce(lambda df1, df2: df1.unionByName(df2), grouped_counts_list)
        
    return grouped_counts

def save_file(table, group, end_pint):
    name="gall_bladder_"+end_pint+"_"+group
    path = "../File"+name
    table.write.format('parquet').mode('overwrite').option('overwriteSchema','True').save(path)

def suppress_values(row):
    total = pd.to_numeric(row["Total"], errors="coerce")  # Convert to numeric, ignore errors
    if total < 5:
        row[columns_to_check] = "< 5"
    else:
        for n_col, pct_col in zip(columns_to_check[1::2], columns_to_check[2::2]):  # Check (n, %) pairs
            n_value = pd.to_numeric(row[n_col], errors="coerce")  # Convert safely to numeric
            if n_value < 5:
                row[n_col] = "< 5"
                row[pct_col] = "< 5"
    return row

def plot_coefficients(data_file, title=''):
    """
    Plots the coefficients with 95% confidence intervals for different delivery points and groups.

    Args:
        data_file (str): Path to the CSV file containing the data.
        title (str): Title of the plot.
    """
    # Calculate errors (coef - lower bound of conf interval)
    data_file['errors'] = data_file['SE (DiD Estimator)'] * 1.96

    # Set plot style
    sns.set_context("poster")

    # Define figure and axes
    fig, ax = plt.subplots(figsize=(15, 10))

    # Create a color palette and map each group to a specific color
    groups = data_file['Group'].unique()
    group_colors = sns.color_palette("husl", len(groups))
    color_map = {group: group_colors[i] for i, group in enumerate(groups)}

    # Plot coefficients with error bars
    datasets = data_file['Dataset'].unique()
    for i, dataset in enumerate(datasets):
        dataset_df = data_file[data_file['Dataset'] == dataset]
        for j, group in enumerate(dataset_df['Group'].unique()):
            group_df = dataset_df[dataset_df['Group'] == group]
            ax.errorbar(
                i + j * 0.2, 
                group_df['Excess Healthcare Use (DiD Estimator)'], 
                yerr=group_df['errors'], 
                fmt='o', 
                color=color_map[group], 
                label=group if i == 0 else ""  # Only label in the first dataset to avoid duplicates
            )

    # Set title and labels
   # plt.title(f'Coefficients of Features w/ 95% Confidence Intervals - {title}', fontsize=30)
    ax.set_ylabel('Coefficients', fontsize=24)
    ax.set_xlabel('Delivery Points', fontsize=24)
    ax.axhline(y=0, linestyle='--', color='red', linewidth=1)

    # Set x-axis ticks and labels
    ax.set_xticks([i + 0.1 for i in range(len(datasets))])
    ax.set_xticklabels(datasets, rotation=45, ha='right')

    # Add legend with consistent group colors
    handles, labels = [], []
    for group, color in color_map.items():
        handles.append(plt.Line2D([0], [0], marker='o', color=color, label=group, linestyle=''))
        labels.append(group)
    ax.legend(handles, labels, title='Group', fontsize=18, title_fontsize='17')

    # Show plot
    plt.show()