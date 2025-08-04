# Databricks notebook source
# MAGIC %pip install openpyxl

# COMMAND ----------

# MAGIC %pip install --upgrade typing_extensions
# MAGIC %pip install pyfixest

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j").setLevel(logging.ERROR)

# COMMAND ----------

import sys
import os
#import pyfixest as pf
#import formulaic

sys.path.append("../src")

# COMMAND ----------

from utils_mod_lib import *

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Read Data

# COMMAND ----------

gb_data=simulate_waiting_list_data()

# COMMAND ----------

# Calculate waiting time for gb_data as the difference between epp_rtt_end_date and epp_rtt_start_date
gb_data = gb_data.withColumn("wt", datediff(col("epp_rtt_end_date"), col("epp_rtt_start_date")))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Waiting Bins

# COMMAND ----------

waiting_list_df = gb_data.withColumn(
    "group",
    when( col("wt") <= 126, "<= 18 weeks")
    .otherwise("> 18 weeks")
)
display(waiting_list_df.groupby("group").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Data for DID

# COMMAND ----------

# Filter data where wait_times > 84
waiting_list_filtered_df = waiting_list_df.filter(col("wt") > 72)

waiting_list_filtered_df = waiting_list_filtered_df.withColumn(
    "after_clock_start",
    when(col("days") >= col("epp_rtt_start_date"), 1).otherwise(0)
).filter(col("after_clock_start") == 1)


# Add 'washout_period_end_date', 'follow_up_end_date', 'washout_period_end_days', 'follow_up_end_days' columns
waiting_list_filtered_df = waiting_list_filtered_df.withColumn(
    "washout_period_end_date",
    F.col("epp_rtt_end_date") + F.lit(28)
).withColumn(
    "follow_up_end_date",
    F.col("washout_period_end_date") + F.col("wt")
).withColumn(
    "washout_period_end_days",
    F.col("wt") + F.lit(28)
).withColumn(
    "follow_up_end_days",
    F.col("washout_period_end_days") + F.col("wt")
)

# Count percentage
total_count = waiting_list_filtered_df.count()
percentage_df = waiting_list_filtered_df.groupBy("group").agg(
    (count("epp_pid") / total_count * 100).alias("percentage")
)

display(percentage_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter Death Cases

# COMMAND ----------

#how many death while waiting 
display(waiting_list_filtered_df.filter(col("REG_DATE_OF_DEATH") == col("follow_up_end_date")))
#how many death in follow up remove who died in follow up
display(waiting_list_filtered_df.count())
deaths_df = waiting_list_filtered_df.filter(col("REG_DATE_OF_DEATH") <= col("follow_up_end_date"))
distinct_patient_count = deaths_df.select("epp_pid").distinct().count()

display(distinct_patient_count)

waiting_list_filtered_df = waiting_list_filtered_df.filter(col("REG_DATE_OF_DEATH").isNull() | (col("REG_DATE_OF_DEATH") > col("follow_up_end_date")))
display(waiting_list_filtered_df.count())



# COMMAND ----------

deaths_df.groupBy("group").agg(
    countDistinct("epp_pid").alias("unique_patient_count")
).display()

waiting_list_filtered_df.groupBy("group").agg(
    countDistinct("epp_pid", "epp_tfc", "epp_rtt_start_date").alias("unique_patient_count")
).display()

# COMMAND ----------

activity_variable=[
    "gp_healthcare_use","gp_Total_cost","u111_healthcare_use","u999_healthcare_use","u00H_healthcare_use","ae_healthcare_use","ae_Total_cost","nel_healthcare_use", "nel_Total_cost","el_healthcare_use","el_Total_cost","op_healthcare_use","op_Total_cost","all_pres_count","antib_pres_count","antipres_pres_count","pain_pres_count","sick_note"
]

# COMMAND ----------

for col_name in activity_variable:
    gb_data_final = waiting_list_filtered_df.withColumn(col_name, when(col(col_name).isNull(), "unknown").otherwise(when(col(col_name) > 0, 1).otherwise(col(col_name))))

#display(gb_data_final)

# COMMAND ----------

# MAGIC %md
# MAGIC # Objectives 4.1 and 4.2

# COMMAND ----------

df_weeks_filtered=filter_data_function_HF_all_cohort(gb_data_final)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Models for All Activities

# COMMAND ----------

cost_var=["gp_Total_cost","ae_Total_cost","nel_Total_cost","el_Total_cost","op_Total_cost"]
for c in cost_var:
    gb_data_final = gb_data_final.withColumn(c, round(col(c)).cast("int"))

# COMMAND ----------

results_t = []
columns = ["ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc", "Sex", "Frailty_level"]
names = ["GP", "GP_Cost", "U111", "U999","UOoH","AE","AE_Cost", "NEl", "Nel_Cost", "EL", "EL_Cost","OP","OP_Cost", "All_Pres", "AtiB_Pres", "Depres_Pres", "Pain_Pres", "Sick_note"]
#names = ["GP", "U111", "U999","UOoH", "AE", "NEl",  "EL","OP", "All_Pres", "AtiB_Pres", "Depres_Pres", "Pain_Pres", "Sick_note"]
for dataset, name in zip(activity_variable, names):
        print(name)
        intervention_groups = ["> 18 weeks"]
        for group in intervention_groups:
                try:
                        display(group)   
                        df_weeks_filtered = filter_data_function_HF(
                               df=gb_data_final, 
                               intervention_group=group, 
                                control_group="<= 18 weeks"
                        )
                        #group_stats=calculate_wait_band_distribution(df_weeks_filtered,columns)
                        #save_file(group_stats, group, name)
                        df_weeks_grouped = group_data_function_HF(
                                df_filtered=df_weeks_filtered, 
                                intervention_group=group, 
                                control_group="<= 18 weeks",
                                var_hc=dataset
                        )
                       
                        #display(df_weeks_grouped)
                        #display(df_weeks_grouped.groupBy("group").agg(countDistinct("epp_pid").alias("unique_patient_count")))
                        unique_counts = df_weeks_grouped.groupBy("group").agg(countDistinct("epp_pid", "epp_tfc", "epp_rtt_start_date").alias("unique_patient_count")).collect()
                        #display(len(unique_counts))

                        if len(unique_counts) ==2 :
                                totals_weeks = totals_table_function_HF_avg( #total agregates
                                        df=df_weeks_grouped,
                                        control_group="<= 18 weeks",
                                        intervention_group=group
                                )
                                display(totals_weeks)
                                ref_period_control_total=totals_weeks.collect()[0][1]
                                ref_period_treated_total=totals_weeks.collect()[0][2]
                                inter_period_control_total=totals_weeks.collect()[1][1]
                                inter_period_treated_total=totals_weeks.collect()[1][2]
                                control_person_week=totals_weeks.collect()[3][1]
                                interv_person_week=totals_weeks.collect()[3][2]
                                total_exess=totals_weeks.collect()[2][2]
                                results_t.append({
                                        "Dataset": name,
                                        "Group": group,
                                        "Unique Patients (Control)": unique_counts[0]["unique_patient_count"],
                                        "Unique Patients (Intervention)": unique_counts[1]["unique_patient_count"],
                                        "Ref Period Total (Control)": ref_period_control_total,
                                        "Ref Period Total (Intervention)":ref_period_treated_total,
                                        "Intervention Period Total (Control)":inter_period_control_total,
                                        "Intervention Period Total (Interbention)": inter_period_treated_total,
                                        "Person-week (Control)":control_person_week,
                                        "Person-week (Intervention)":interv_person_week,
                                        "Total Excess":total_exess  
                                        })
                except Exception as e:
                        print(f"An error occurred for group {group} and dataset {name}: {e}")

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

total_results = pd.DataFrame(results_t)
total_results["Total Excess"] = total_results["Total Excess"] * 100 * 4.3

# Ensure 'heatmap_data' is already pivoted in the correct format
heatmap_data = total_results.pivot_table(
    index="Dataset", 
    columns="Group", 
    values="Total Excess", 
    aggfunc='mean'
)

# Define a colormap with only two colors: green for negative and red for positive
cmap = ListedColormap(["green", "red"])

# Normalize color scaling to center at 0 with discrete boundaries
bounds = [-np.inf, 0, np.inf]
norm = BoundaryNorm(bounds, cmap.N)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    heatmap_data, 
    annot=True, 
    fmt=".2f", 
    cmap=cmap, 
    norm=norm, 
    cbar=False  # Remove the color bar
)

# Labels and title
plt.title("Health Care Rate per 100 patient per one month. ")
plt.xlabel("Group")
plt.ylabel("Delivery Points")
plt.show()

# COMMAND ----------

results = []
columns = ["ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc", "Sex", "Frailty_level"]
names = ["GP", "GP_Cost", "U111", "U999","UOoH","AE","AE_Cost", "NEl", "Nel_Cost", "EL", "EL_Cost","OP","OP_Cost", "All_Pres", "AtiB_Pres", "Depres_Pres", "Pain_Pres", "Sick_note"]
#names = ["GP", "U111", "U999","UOoH", "AE", "NEl",  "EL","OP", "All_Pres", "AtiB_Pres", "Depres_Pres", "Pain_Pres", "Sick_note"]
for dataset, name in zip(activity_variable, names):
        print(name)
        intervention_groups = [ "> 18 weeks"]
        for group in intervention_groups:
                try:
                        display(group)   
                        df_weeks_filtered = filter_data_function_HF(
                               df=gb_data_final, 
                               intervention_group=group, 
                                control_group="<= 18 weeks"
                        )
                        #group_stats=calculate_wait_band_distribution(df_weeks_filtered,columns)
                        #save_file(group_stats, group, name)
                        df_weeks_grouped = group_data_function_HF(
                                df_filtered=df_weeks_filtered, 
                                intervention_group=group, 
                                control_group="<= 18 weeks",
                                var_hc=dataset
                        )
                       
                        #display(df_weeks_grouped)
                        #display(df_weeks_grouped.groupBy("group").agg(countDistinct("epp_pid").alias("unique_patient_count")))
                        unique_counts = df_weeks_grouped.groupBy("group").agg(countDistinct("epp_pid", "epp_tfc", "epp_rtt_start_date").alias("unique_patient_count")).collect()
                        #display(len(unique_counts))

                        if len(unique_counts) ==2 :
                                totals_weeks = totals_table_function_HF2( #total agregates
                                        df=df_weeks_grouped,
                                        control_group="<= 18 weeks",
                                        intervention_group=group
                                )
                                display(totals_weeks)
                                fe_twfe = run_fixed_effects_HF_total(df_weeks_grouped) #twfe on totals
                                display(fe_twfe.summary())            
                                # Extract summary data
                                p_value = float(fe_twfe.pvalue().iloc[0])
                                coef_value = float(fe_twfe.coef().values[0])
                                se = float(fe_twfe.se().iloc[0])
                                ref_period_control_total=totals_weeks.collect()[0][1]
                                ref_period_treated_total=totals_weeks.collect()[0][2]
                                inter_period_control_total=totals_weeks.collect()[1][1]
                                inter_period_treated_total=totals_weeks.collect()[1][2]
                                control_person_week=totals_weeks.collect()[3][1]
                                inter_person_week=totals_weeks.collect()[3][2]
                                total_exess=totals_weeks.collect()[2][2]

                                #print(p_value, coef_value, se )
                                results.append({
                                        "Dataset": name,
                                        "Group": group,
                                        "Unique Patients (Control)": unique_counts[0]["unique_patient_count"],
                                        "Unique Patients (Intervention)": unique_counts[1]["unique_patient_count"],
                                        "Ref Period Total (Control)": ref_period_control_total,
                                        "Ref Period Total (Intervention)":ref_period_treated_total,
                                        "Intervention Period Total (Control)":inter_period_control_total,
                                        "Intervention Period Total (Interbention)": inter_period_treated_total,
                                        "Person-week (Control)":control_person_week,
                                        "Person-week (Intervention)":inter_person_week,
                                        "Total Excess":total_exess,
                                        "Excess Healthcare Use (DiD Estimator)": coef_value,
                                        "p-value (DiD Estimator)": p_value,
                                        "SE (DiD Estimator)": se    
                                        })
                except Exception as e:
                        print(f"An error occurred for group {group} and dataset {name}: {e}")
#print(results)

# COMMAND ----------

# Assuming results is a list of dictionaries, convert it to a DataFrame
results_pd = pd.DataFrame(results)

# Define the directory and file path
#directory = "../Files"
#file_path = os.path.join(directory, "Table4_1_Table_4_2gb_two_groups.xlsx")

# Create the directory if it does not exist
#os.makedirs(directory, exist_ok=True)

# Write the DataFrame to an Excel file
#results_pd.to_excel(file_path, index=False)

# COMMAND ----------

total_results = pd.DataFrame(results)
# Round all floats to 2 decimal places
total_results = total_results.round(2)



# COMMAND ----------

display(results)

# COMMAND ----------

# Ensure results is a DataFrame
results_df = spark.createDataFrame(results)

# Select and round only numerical columns
numerical_cols = [col for col, dtype in results_df.dtypes if dtype in ('int', 'double')]
a = results_df.select(*[round(results_df[col], 2).alias(col) if col in numerical_cols else results_df[col] for col in results_df.columns])

# Convert DataFrame to comma-separated string
a_str = a.toPandas().to_csv(index=False)
a_str

# COMMAND ----------

total_results_filtered = total_results[~total_results['Dataset'].str.contains('_Cost', case=False)]
plot_coefficients(total_results_filtered, title='')

total_results_filtered = total_results[total_results['Dataset'].str.contains('_Cost', case=False)]
plot_coefficients(total_results_filtered, title='')

# COMMAND ----------

# Sample data (replace with your actual data)
# Ensure 'heatmap_data' is already pivoted in the correct format
heatmap_data = total_results.pivot_table(index="Dataset", columns="Group", values="Total Excess", aggfunc='mean')

# Define a colormap with only two colors: green for negative and red for positive
cmap = ListedColormap(["green", "red"])

# Normalize color scaling to center at 0 with discrete boundaries
bounds = [-np.inf, 0, np.inf]
norm = BoundaryNorm(bounds, cmap.N)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    heatmap_data, 
    annot=True, 
    fmt=".2f", 
    cmap=cmap, 
    norm=norm, 
    cbar=False  # Remove the color bar
)

# Labels and title
#plt.xticks(rotation=45)
plt.title("Total of Healthcare Use Changes Across Delivery Points")
plt.xlabel("Group")
plt.ylabel("Delivery Points")
plt.show()

# COMMAND ----------

# Sample data (replace with your actual data)
# Ensure 'heatmap_data' is already pivoted in the correct format
heatmap_data = total_results.pivot_table(index="Dataset", columns="Group", values="Excess Healthcare Use (DiD Estimator)", aggfunc='mean')

# Define a colormap with only two colors: green for negative and red for positive
cmap = ListedColormap(["green", "red"])

# Normalize color scaling to center at 0 with discrete boundaries
bounds = [-np.inf, 0, np.inf]
norm = BoundaryNorm(bounds, cmap.N)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    heatmap_data, 
    annot=True, 
    fmt=".2f", 
    cmap=cmap, 
    norm=norm, 
    cbar=False  # Remove the color bar
)

# Labels and title
#plt.xticks(rotation=45)
plt.title("Total of Healthcare Use Changes Across Delivery Points")
plt.xlabel("Group")
plt.ylabel("Delivery Points")
plt.show()

# COMMAND ----------

tmp.columns

# COMMAND ----------

# MAGIC %md
# MAGIC # Objective 4.4

# COMMAND ----------

import pyspark.sql.functions as F
import pandas as pd
from scipy.stats import chi2_contingency

# Step 1: Aggregate gp_healthcare_use by categorical groups
tmp_grouped = tmp.groupBy("sex", "ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc","epp_pid", "epp_tfc", "epp_rtt_start_date", "wt").agg(F.sum("gp_healthcare_use").alias("gp_healthcare_use_s"))

# Convert to weekly usage
tmp_grouped = tmp_grouped.withColumn("gp_healthcare_use_weekly", F.col("gp_healthcare_use_s") / (F.col("wt") / 7))

# Step 2: Define Top 10% and Bottom 90%
percentile_90 = tmp_grouped.approxQuantile("gp_healthcare_use_weekly", [0.9], 0.01)[0]

df_filtered = tmp_grouped.withColumn(
        "GP_usage_group",
        F.when(F.col("gp_healthcare_use_weekly") >= percentile_90, "Top 10%").otherwise("Bottom 90%")
    )



# Step 1: Aggregate counts for Top 10% and Bottom 90%
categories = ["sex", "ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc"]

# Convert to long format (stack all categorical variables into two columns: "Category" and "Value")
long_df = df_filtered.select(
    F.expr("stack({}, {})".format(
        len(categories),
        ", ".join([f"'{col}', {col}" for col in categories])
    )).alias("Category", "Value"),
    "GP_usage_group"
)

# Step 2: Compute counts for each category-value pair
summary_df = (
    long_df.groupBy("Category", "Value")
    .agg(
        F.count("*").alias("Total"),
        F.sum(F.when(F.col("GP_usage_group") == "Bottom 90%", 1).otherwise(0)).alias("Bottom 90%"),
        F.sum(F.when(F.col("GP_usage_group") == "Top 10%", 1).otherwise(0)).alias("Top 10%")
    )
    .withColumn("Bottom 90% %", (F.col("Bottom 90%") / F.col("Total")) * 100)
    .withColumn("Top 10% %", (F.col("Top 10%") / F.col("Total")) * 100)
)

# Convert Spark DataFrame to Pandas for Chi-square calculations
summary_pd = summary_df.toPandas()

# Step 3: Compute p-values row-wise with a proper 2D contingency table
def compute_p_value(row):
    """Perform Chi-square test for each row with proper 2x2 contingency table."""
    contingency_table = [[row["Top 10%"], row["Bottom 90%"]]]  # <-- Issue here: This is 1 row, not 2D
    
    # Ensure we have a valid 2x2 table by checking across all rows in the same category
    category_data = summary_pd[summary_pd["Category"] == row["Category"]][["Top 10%", "Bottom 90%"]]
    
    if category_data.shape[0] > 1:  # Ensure we have multiple values to compare
        chi2, p, _, _ = chi2_contingency(category_data)
        return float(p)
    else:
        return None  # Not enough data to run a test

# Apply the test across all rows
summary_pd["p-value"] = summary_pd.apply(compute_p_value, axis=1)

# Convert back to Spark DataFrame
summary_spark = spark.createDataFrame(summary_pd)

# Display results
display(summary_spark)


# COMMAND ----------

summary_spark_pd = summary_spark.toPandas()
print(summary_spark_pd.to_string())

# COMMAND ----------

# Convert the Spark DataFrame to a Pandas DataFrame
summary_spark_pd = summary_spark.toPandas()

# Define the directory and file path
#directory = "../Files"
#file_path = os.path.join(directory, "Table4_4gb.xlsx")

# Create the directory if it does not exist
#os.makedirs(directory, exist_ok=True)

# Write the Pandas DataFrame to an Excel file
#summary_spark_pd.to_excel(file_path, index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Objective 4.5

# COMMAND ----------

import pyspark.sql.functions as F
import pandas as pd
from scipy.stats import chi2_contingency

# Step 1: Aggregate gp_healthcare_use by categorical groups
tmp_grouped = tmp.groupBy("sex", "ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc","epp_tfc", "epp_rtt_start_date", "wt").agg(F.sum("ae_healthcare_use").alias("ae_healthcare_use_s"))

# Convert to weekly usage
tmp_grouped = tmp_grouped.withColumn("ae_healthcare_use_weekly", F.col("ae_healthcare_use_s") / (F.col("wt") / 7))

# Step 2: Define Top 10% and Bottom 90%
percentile_90 = tmp_grouped.approxQuantile("ae_healthcare_use_weekly", [0.9], 0.01)[0]

df_filtered = tmp_grouped.withColumn(
        "GP_usage_group",
        F.when(F.col("ae_healthcare_use_weekly") >= percentile_90, "Top 10%").otherwise("Bottom 90%")
    )



# Step 1: Aggregate counts for Top 10% and Bottom 90%
categories = ["sex", "ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc"]

# Convert to long format (stack all categorical variables into two columns: "Category" and "Value")
long_df = df_filtered.select(
    F.expr("stack({}, {})".format(
        len(categories),
        ", ".join([f"'{col}', {col}" for col in categories])
    )).alias("Category", "Value"),
    "GP_usage_group"
)

# Step 2: Compute counts for each category-value pair
summary_df = (
    long_df.groupBy("Category", "Value")
    .agg(
        F.count("*").alias("Total"),
        F.sum(F.when(F.col("GP_usage_group") == "Bottom 90%", 1).otherwise(0)).alias("Bottom 90%"),
        F.sum(F.when(F.col("GP_usage_group") == "Top 10%", 1).otherwise(0)).alias("Top 10%")
    )
    .withColumn("Bottom 90% %", (F.col("Bottom 90%") / F.col("Total")) * 100)
    .withColumn("Top 10% %", (F.col("Top 10%") / F.col("Total")) * 100)
)

# Convert Spark DataFrame to Pandas for Chi-square calculations
summary_pd = summary_df.toPandas()

# Step 3: Compute p-values row-wise with a proper 2D contingency table
def compute_p_value(row):
    """Perform Chi-square test for each row with proper 2x2 contingency table."""
    contingency_table = [[row["Top 10%"], row["Bottom 90%"]]]  # <-- Issue here: This is 1 row, not 2D
    
    # Ensure we have a valid 2x2 table by checking across all rows in the same category
    category_data = summary_pd[summary_pd["Category"] == row["Category"]][["Top 10%", "Bottom 90%"]]
    
    if category_data.shape[0] > 1:  # Ensure we have multiple values to compare
        chi2, p, _, _ = chi2_contingency(category_data)
        return float(p)
    else:
        return None  # Not enough data to run a test

# Apply the test across all rows
summary_pd["p-value"] = summary_pd.apply(compute_p_value, axis=1)

# Convert back to Spark DataFrame
summary_spark = spark.createDataFrame(summary_pd)

# Display results
display(summary_spark)


# COMMAND ----------

# Convert the Spark DataFrame to a Pandas DataFrame
summary_spark_pd = summary_spark.toPandas()

# Define the directory and file path
#directory = "../Files"
#file_path = os.path.join(directory, "Table4_5gb.xlsx")

# Create the directory if it does not exist
#os.makedirs(directory, exist_ok=True)

# Write the Pandas DataFrame to an Excel file
#summary_spark_pd.to_excel(file_path, index=False)

# COMMAND ----------

# Step 6: Add p-values to tables
for col in categories:
    summary_tables[col] = summary_tables[col].withColumn("p-value", F.when(p_values[col] == "Not enough data", None).otherwise(p_values[col]))

# Step 7: Show results
for col, table in summary_tables.items():
    display(table) 

# COMMAND ----------

from pyspark.sql.functions import col, when, count, lit, sum as spark_sum
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType
from scipy.stats import chi2_contingency
import pandas as pd

# Define the columns to be analyzed
columns = ["ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc", "Sex", "Frailty_level"]
personal_ch = tmp.groupBy(columns+["epp_pid", "group"]).agg(count("*").alias("count"))

# Calculate total for the total_healthcare_use
total_healthcare_use = personal_ch.agg(spark_sum("count").alias("total_healthcare_use")).collect()[0]["total_healthcare_use"]

# Function to calculate chi-square test and format the results


# Initialize an empty list to store results
results_list = []

# Calculate chi-square test for each column and append the results
for column in columns:
    result = calculate_chisquare(personal_ch, column)
    results_list.append(result)

# Concatenate all results into a single DataFrame
final_results = pd.concat(results_list, ignore_index=True)

# Display the final results
display(spark.createDataFrame(final_results))

# COMMAND ----------

# Group by the specified columns and aggregate
personal_ch = df_weeks_filtered.groupBy(
    "epp_pid", "group", "ndl_age_band", "ndl_imd_quantile", 
    "ndl_ethnicity", "ndl_ltc", "Sex", "Frailty_level"
).agg(count("*").alias("count"))

variables = ["ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc", "Sex", "Frailty_level"]
grouped_counts_list = []

for var in variables:
    df = personal_ch.withColumn(var, when(col(var).isNull(), "unknown").otherwise(col(var)))
    grouped_counts = df.groupBy(var).pivot("group").agg(count("*").alias("count")).withColumnRenamed(var, "value")
    total_counts = grouped_counts.select([sum(col(c)).alias(c) for c in grouped_counts.columns if c != "value"])
    for c in grouped_counts.columns:
        if c != "value":
            grouped_counts = grouped_counts.withColumn(f"{c}_percentage", (col(c) / total_counts.first()[c]) * 100)
    grouped_counts = grouped_counts.withColumn("Variable", lit(var))
    grouped_counts_list.append(grouped_counts)

grouped_counts = reduce(lambda df1, df2: df1.unionByName(df2), grouped_counts_list)

display(grouped_counts)

display(calculate_wait_band_distribution(personal_ch,variables))

# COMMAND ----------

# MAGIC %md
# MAGIC # Average weekly usage

# COMMAND ----------

from pyspark.sql.functions import countDistinct

results = []
names = ["GP", "GP_Cost", "U111", "U999","UOoH","AE","AE_Cost", "NEl", "Nel_Cost", "EL", "EL_Cost","OP","OP_Cost", "All_Pres", "AtiB_Pres", "Depres_Pres", "Pain_Pres", "Sick_note"]
#names = ["GP", "U111", "U999","UOoH", "AE", "NEl",  "EL","OP", "All_Pres", "AtiB_Pres", "Depres_Pres", "Pain_Pres", "Sick_note"]
for dataset, name in zip(activity_variable, names):
        print(name)
        intervention_groups = ["19-30 weeks", "31-42 weeks", "> 42 weeks"]
        for group in intervention_groups:
                try:
                        display(group)   
                        df_weeks_filtered = filter_data_function_HF(
                                df=gb_data_final, 
                                intervention_group=group, 
                                control_group="<= 18 weeks"
                        )
                        df_weeks_grouped = group_data_function_HF(
                                df_filtered=df_weeks_filtered, 
                                intervention_group=group, 
                                control_group="<= 18 weeks",
                                var_hc=dataset
                        )
                        #display(df_weeks_grouped)
                        #display(df_weeks_grouped.groupBy("group").agg(countDistinct("epp_pid").alias("unique_patient_count")))
                        unique_counts = df_weeks_grouped.groupBy("group").agg(countDistinct("epp_pid").alias("unique_patient_count")).collect()
                        #display(len(unique_counts))

                        if len(unique_counts) ==2 :
                                totals_weeks = totals_table_function_HF_avg(
                                        df=df_weeks_grouped,
                                        control_group="<= 18 weeks",
                                        intervention_group=group
                                )
                               # display(totals_weeks)
                                fe_twfe = run_fixed_effects_HF(df_weeks_grouped)
                                #display(fe_twfe.summary())            
                                # Extract summary data
                                p_value = float(fe_twfe.pvalue().iloc[0])
                                coef_value = float(fe_twfe.coef().values[0])
                                se = float(fe_twfe.se().iloc[0])
                                ref_period_control_total=totals_weeks.collect()[0][1]
                                ref_period_treated_total=totals_weeks.collect()[0][2]
                                inter_period_control_total=totals_weeks.collect()[1][1]
                                inter_period_treated_total=totals_weeks.collect()[1][2]
                                control_person_week=totals_weeks.collect()[3][1]
                                control_person_week=totals_weeks.collect()[3][2]
                                total_exess=totals_weeks.collect()[2][2]

                                #print(p_value, coef_value, se )
                                results.append({
                                        "Dataset": name,
                                        "Group": group,
                                        "Unique Patients (Control)": unique_counts[0]["unique_patient_count"],
                                        "Unique Patients (Intervention)": unique_counts[1]["unique_patient_count"],
                                        "Ref Period Total (Control)": ref_period_control_total,
                                        "Ref Period Total (Intervention)":ref_period_treated_total,
                                        "Intervention Period Total (Control)":inter_period_control_total,
                                        "Intervention Period Total (Interbention)": inter_period_treated_total,
                                        "Person-week (Control)":control_person_week,
                                        "Person-week (Intervention)":control_person_week,
                                        "Total Excess":total_exess,
                                        "Excess Healthcare Use (DiD Estimator)": coef_value,
                                        "p-value (DiD Estimator)": p_value,
                                        "SE (DiD Estimator)": se    
                                        })
                except Exception as e:
                        print(f"An error occurred for group {group} and dataset {name}: {e}")
#print(results)

# COMMAND ----------

print(results)

# COMMAND ----------

avg_results = pd.DataFrame(results)
avg_results = avg_results.round(2)
display(avg_results)

# COMMAND ----------

avg_results_filtered = avg_results[~avg_results['Dataset'].str.contains('_Cost', case=False)]
plot_coefficients(avg_results_filtered, title='')

total_results_filtered = avg_results[avg_results['Dataset'].str.contains('_Cost', case=False)]
plot_coefficients(total_results_filtered, title='')

# COMMAND ----------

# Pivot data for heatmap
heatmap_data = avg_results.pivot_table(index="Dataset", columns="Group", values="Excess Healthcare Use (DiD Estimator)", aggfunc='mean')

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
plt.title("Heatmap of Healthcare Use Changes Across Delivery Points")
plt.xlabel("Group")
plt.ylabel("Delivery Points")
plt.show()

# COMMAND ----------

# Pivot data for heatmap
heatmap_data = avg_results.pivot_table(index="Dataset", columns="Group", values="Total Excess", aggfunc='mean')

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
plt.title("Heatmap of Healthcare Use Changes Across Delivery Points")
plt.xlabel("Group")
plt.ylabel("Delivery Points")
plt.show()

# COMMAND ----------

# Compute the difference
results_df = spark.createDataFrame(results)
results_df = results_df.withColumn("Difference", col("Intervention Period Total (Interbention)") - col("Ref Period Total (Intervention)"))

# Convert to Pandas DataFrame for plotting
pdf = results_df.toPandas()

# Pivot for heatmap
heatmap_data = pdf.pivot(index="Dataset", columns="Group", values="Difference")

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", center=0, annot_kws={"size": 8})
plt.title("Heatmap of Difference Between Intervention and Reference Periods", fontsize=10)
plt.xticks(fontsize=8, rotation=45, ha='right')
plt.yticks(fontsize=8, rotation=0)
plt.show()

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
plt.title("Heatmap of Healthcare Use Changes Across Delivery Points")
plt.xlabel("Group")
plt.ylabel("Dataset")
plt.show()

# COMMAND ----------

avg_results_6m = pd.DataFrame(results)
avg_results_6m = avg_results.round(2)
display(avg_results_6m)

# COMMAND ----------

# MAGIC %md
# MAGIC # 6 Month after  - Fixed post treatment 

# COMMAND ----------

#Run for fixed 6 months after recovery 
waiting_list_filtered_df = waiting_list_df.filter(col("wt") > 72)

waiting_list_filtered_df = waiting_list_filtered_df.withColumn(
    "after_clock_start",
    when(col("days") >= col("epp_rtt_start_date"), 1).otherwise(0)
).filter(col("after_clock_start") == 1)


# Add 'washout_period_end_date', 'follow_up_end_date', 'washout_period_end_days', 'follow_up_end_days' columns
waiting_list_filtered_df = waiting_list_filtered_df.withColumn(
    "washout_period_end_date",
    F.col("epp_rtt_end_date") + F.lit(28)
).withColumn(
    "follow_up_end_date",
    F.col("washout_period_end_date") +F.lit(180)
).withColumn(
    "washout_period_end_days",
    F.col("wt") + F.lit(28)
).withColumn(
    "follow_up_end_days",
    F.col("washout_period_end_days") + F.lit(180)
)

display(waiting_list_filtered_df.count())
deaths_df = waiting_list_filtered_df.filter(col("REG_DATE_OF_DEATH") <= col("follow_up_end_date"))
distinct_patient_count = deaths_df.select("epp_pid").distinct().count()

display(distinct_patient_count)

waiting_list_filtered_df = waiting_list_filtered_df.filter(col("REG_DATE_OF_DEATH").isNull() | (col("REG_DATE_OF_DEATH") > col("follow_up_end_date")))
display(waiting_list_filtered_df.count())

for col_name in activity_variable:
    gb_data_final = waiting_list_filtered_df.withColumn(col_name, when(col(col_name) > 0, 1).otherwise(0))

results = []
#names = ["GP"]#, "GP_Cost", "U111","U111_Cost", "U999","U999_Cost", "UOoH","UooH_Cost", "AE","AE_Cost", "NEl", "Nel_Cost", "EL", "EL_Cost","OP","OP_Cost", "All_Pres", "AtiB_Pres", "Depres_Pres", "Pain_Pres", "Sick_note"]
names = ["GP", "U111", "U999","UOoH", "AE", "NEl",  "EL","OP", "All_Pres", "AtiB_Pres", "Depres_Pres", "Pain_Pres", "Sick_note"]
for dataset, name in zip(activity_variable, names):
        print(name)
        intervention_groups = ["19-30 weeks", "31-42 weeks", "> 42 weeks"]
        for group in intervention_groups:
                try:
                        display(group)   
                        df_weeks_filtered = filter_data_function_HF(
                                df=gb_data_final, 
                                intervention_group=group, 
                                control_group="<= 18 weeks"
                        )
                        df_weeks_grouped = group_data_function_HF(
                                df_filtered=df_weeks_filtered, 
                                intervention_group=group, 
                                control_group="<= 18 weeks",
                                var_hc=dataset
                        )
                        #display(df_weeks_grouped)
                        #display(df_weeks_grouped.groupBy("group").agg(countDistinct("epp_pid").alias("unique_patient_count")))
                        unique_counts = df_weeks_grouped.groupBy("group").agg(countDistinct("epp_pid").alias("unique_patient_count")).collect()
                        #display(len(unique_counts))

                        if len(unique_counts) ==2 :
                                totals_weeks = totals_table_function_HF_avg(
                                        df=df_weeks_grouped,
                                        control_group="<= 18 weeks",
                                        intervention_group=group
                                )
                               # display(totals_weeks)
                                fe_twfe = run_fixed_effects_HF(df_weeks_grouped)
                                #display(fe_twfe.summary())            
                                # Extract summary data
                                p_value = float(fe_twfe.pvalue().iloc[0])
                                coef_value = float(fe_twfe.coef().values[0])
                                se = float(fe_twfe.se().iloc[0])
                                ref_period_control_total=totals_weeks.collect()[0][1]
                                ref_period_treated_total=totals_weeks.collect()[0][2]
                                inter_period_control_total=totals_weeks.collect()[1][1]
                                inter_period_treated_total=totals_weeks.collect()[1][2]
                                control_person_week=totals_weeks.collect()[3][1]
                                control_person_week=totals_weeks.collect()[3][2]
                                total_exess=totals_weeks.collect()[2][2]

                                #print(p_value, coef_value, se )
                                results.append({
                                        "Dataset": name,
                                        "Group": group,
                                        "Unique Patients (Control)": unique_counts[0]["unique_patient_count"],
                                        "Unique Patients (Intervention)": unique_counts[1]["unique_patient_count"],
                                        "Ref Period Total (Control)": ref_period_control_total,
                                        "Ref Period Total (Intervention)":ref_period_treated_total,
                                        "Intervention Period Total (Control)":inter_period_control_total,
                                        "Intervention Period Total (Interbention)": inter_period_treated_total,
                                        "Person-week (Control)":control_person_week,
                                        "Person-week (Intervention)":control_person_week,
                                        "Total Excess":total_exess,
                                        "Excess Healthcare Use (DiD Estimator)": coef_value,
                                        "p-value (DiD Estimator)": p_value,
                                        "SE (DiD Estimator)": se    
                                        })
                except Exception as e:
                        print(f"An error occurred for group {group} and dataset {name}: {e}")
#print(results)

# COMMAND ----------

display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC # OLS

# COMMAND ----------

# MAGIC %md
# MAGIC Difference in Difference 
# MAGIC
# MAGIC Treatment group – patient who waited longer than 18th weeks  for their treatment, hese patients are exposed to the "treatment" of longer waiting times, which may influence healthcare utilization, outcomes, or other measures during or after the waiting period
# MAGIC
# MAGIC Control group – patients who wait 18 weeks or less for their treatment, they are not exposed for a treatment of extended waiting
# MAGIC
# MAGIC Treatment Timing:
# MAGIC
# MAGIC Pre-Treatment Period: Before the 18-week threshold is reached.
# MAGIC For both groups, this could include the initial weeks of referral-to-treatment (RTT) waiting time.
# MAGIC Post-Treatment Period: After the 18-week threshold is crossed.
# MAGIC For the treatment group, this includes additional weeks of waiting beyond 18 weeks or the time post-procedure.
# MAGIC For the control group, this includes the time post-procedure after their shorter waiting period.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col, when

df = gb_data_final.withColumn(
    "group",
    when((col("wt") <= 84), "<= 12 weeks")
    .when((col("wt") > 84) & (col("wt") <= 126), "<= 18 weeks")
    .otherwise("> 18 weeks")
)
df = df.filter((col("wt") > 84) & (col("wt") != 126))
display(df.groupby("group").count())

# COMMAND ----------

from pyspark.sql.functions import col, when, date_add
def filter_data_function_M(df, intervention_group, control_group, recovery):
    # Convert max follow-up weeks to days
    
    # Filter and add time periods based on individual waiting time
    df_filtered = (
        df.filter((col("group") == intervention_group) | (col("group") == control_group))
        .withColumn("start_of_reference_period", when(col("epp_rtt_end_date") < date_add(col("epp_rtt_start_date"),85), col("epp_rtt_end_date")).otherwise(date_add(col("epp_rtt_start_date"), 85)))
        .withColumn("end_of_reference_period", date_add(col("start_of_reference_period"), 360))  # Define recovery period if needed
        .filter((col("days") >= col("epp_rtt_start_date"))  & (col("days")<=col("end_of_reference_period")))
        .withColumn(
            "time_period",
            when(col("days") <= col("start_of_reference_period"), 0)  # Waiting period
            .otherwise(1)  # Post-recovery period
        )
    )

    return df_filtered

# COMMAND ----------

def group_data_function(df_filtered, intervention_group, control_group, colum_hc, col_list):
    # Group and aggregate
    df_grouped_by_time_period = (
        df_filtered.groupBy("epp_pid", "time_period", "group", *col_list)
        .agg(F.sum(colum_hc).alias('total_hc_use'))
        .withColumn(
            "treated",
            when(col("group") == control_group, 0).when(col("group") == intervention_group, 1)
        )
    )

    df_grouped = df_grouped_by_time_period.withColumn(
        'avg_weekly_use', 
        (F.col('total_hc_use') / 372 * 7).cast('double')
    )

    return df_grouped

# COMMAND ----------

inter_group="> 18 weeks"
df_30_weeks_filtered = filter_data_function_M(
    df=df, 
    intervention_group=inter_group, 
    control_group="<= 18 weeks",
    recovery=28  # Default recovery period
)

# COMMAND ----------

from pyspark.sql.functions import col, when
columns = ["ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc", "Sex", "Frailty_level"]

# Replace null values with "unknown" in the specified columns
for column in columns:
    df_30_weeks_filtered = df_30_weeks_filtered.withColumn(column, when(col(column).isNull(), "unknown").otherwise(col(column)))

df_grouped = group_data_function(
    df_filtered=df_30_weeks_filtered,
    intervention_group=inter_group,
    control_group="<= 18 weeks", 
    colum_hc="gp_healthcare_use",
    col_list=columns
)
display(df_grouped)

# COMMAND ----------

import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd


columns2 = ["ndl_ethnicity", "ndl_ltc", "Sex", "Frailty_level"]
#columns2 = ["ndl_ethnicity", "Sex", "Frailty_level"]
#columns2 = ["ndl_ethnicity", "ndl_ltc", "Frailty_level"]

# Define your activity variables
#activity_variable = [
#    "gp_healthcare_use", "u111_healthcare_use", "u999_healthcare_use", "u00H_healthcare_use", 
#    "ae_healthcare_use", "nel_healthcare_use", "el_healthcare_use", "op_healthcare_use", 
#    "all_pres_count", "antib_pres_count", "antipres_pres_count", "pain_pres_count", "sick_note"
#]

activity_variable = [
    "gp_healthcare_use",  
    "ae_healthcare_use", "nel_healthcare_use", "el_healthcare_use", "op_healthcare_use", 
    "antib_pres_count", "antipres_pres_count", "pain_pres_count", 
]

# Initialize empty lists to collect the estimates and standard errors
all_estimates = []

# Loop over each activity variable
for activity in activity_variable:
    # Filter data (assuming the filter function and group_data_function work as intended)
    df_30_weeks_filtered = filter_data_function_M(df=df, intervention_group='> 18 weeks', control_group='<= 18 weeks', recovery=28)

    # Group data
    df_grouped = group_data_function(
        df_filtered=df_30_weeks_filtered,
        intervention_group="> 18 weeks",
        control_group="<= 18 weeks", 
        colum_hc=activity,  # Use the current activity variable
        col_list=columns2
    )

    # Check if all columns exist in the DataFrame
    missing_columns = [col for col in columns2 if col not in df_grouped.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in DataFrame: {missing_columns}")

    # Define the formula for the regression model
    base_formula = f"avg_weekly_use ~ C(group) * C(time_period)"

    # Add columns from the list
    full_formula = base_formula + " + " + " + ".join(columns2)

    # Fit the model for the current activity variable
    did_model = smf.ols(formula=full_formula, data=df_grouped.toPandas()).fit()
    print(did_model.summary())

    # Collect all estimates and standard errors
    estimates = did_model.params
    se = did_model.bse

    # Append estimates and standard errors to the list
    for term in estimates.index:
        all_estimates.append({
            'Activity': activity,
            'Term': term,
            'Estimate': estimates[term],
            'Standard Error': se[term]
        })



# COMMAND ----------

display(pd.DataFrame(all_estimates))

# COMMAND ----------

# Create a DataFrame to store all estimates and standard errors
results_df = pd.DataFrame(all_estimates)

# Plot the estimates with error bars
fig, axes = plt.subplots(4, 2, figsize=(20, 20), sharex=True)
axes = axes.flatten()

for i, activity in enumerate(activity_variable[:len(axes)]):
    activity_df = results_df[results_df['Activity'] == activity]
    axes[i].errorbar(activity_df['Estimate'], activity_df['Term'], xerr=activity_df['Standard Error'], fmt='o', markersize=4)
    axes[i].axvline(0, color='black', linestyle='--', linewidth=0.8)  # Add vertical line at x=0
    axes[i].set_title(f'Estimates for {activity}', fontsize=14)
    axes[i].set_xlabel('Estimate', fontsize=14)
    axes[i].tick_params(axis='y', labelsize=12)
    if i % 2 != 0:
        axes[i].set_yticklabels([])

plt.suptitle('Estimates of Healthcare Use with Standard Errors for Different Activities', fontsize=16)
plt.subplots_adjust(bottom=0.05, left=0.05, right=0.95, top=0.95, hspace=0.3, wspace=0.3)
plt.show()

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Filter the DataFrame for each term
terms = results_df['Term'].unique()

for term in terms:
    filtered_df = results_df[results_df['Term'] == term]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Scatter plot with error bars
    sns.scatterplot(
        data=filtered_df,
        x='Estimate',
        y='Activity',
        marker='o'
    )
    plt.errorbar(
        filtered_df['Estimate'], 
        filtered_df['Activity'], 
        xerr=filtered_df['Standard Error'], 
        fmt='none', 
        capsize=3, 
        color='blue'
    )
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(f'Estimates for {term}', fontsize=14)
    plt.xlabel('Estimate Value', fontsize=12)
    plt.ylabel('Activity', fontsize=12)

    plt.tight_layout()
    plt.show()

# COMMAND ----------

display(results.summary().tables[1].data)
