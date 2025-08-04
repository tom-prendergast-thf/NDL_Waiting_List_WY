# Databricks notebook source
# MAGIC %pip install --upgrade typing_extensions
# MAGIC %pip install pyfixest

# COMMAND ----------

# MAGIC %pip install openpyxl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j").setLevel(logging.ERROR)

# COMMAND ----------

import sys
import os
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

waiting_list_df=waiting_list_df.filter(~(col('epp_referral_priority') == 'cancer'))

# COMMAND ----------

# DBTITLE 1,percentage
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

# Plot percentage
percentage_pandas_df = percentage_df.toPandas()
percentage_pandas_df.plot(kind='bar', x='group', y='percentage', legend=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter Death Cases

# COMMAND ----------

#how many death in follow up remove who died in follow up
display(waiting_list_filtered_df.count())
deaths_df = waiting_list_filtered_df.filter(col("REG_DATE_OF_DEATH") <= col("follow_up_end_date"))
distinct_patient_count = deaths_df.select("epp_pid").distinct().count()

display(distinct_patient_count)

waiting_list_filtered_df = waiting_list_filtered_df.filter(col("REG_DATE_OF_DEATH").isNull() | (col("REG_DATE_OF_DEATH") > col("follow_up_end_date")))
display(waiting_list_filtered_df.count())

# COMMAND ----------

from pyspark.sql.functions import countDistinct


deaths_df.groupBy("group").agg(
    countDistinct("epp_pid").alias("unique_patient_count")
).display()

waiting_list_filtered_df.groupBy("group").agg(
    countDistinct("epp_pid").alias("unique_patient_count")
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
# MAGIC ## Filter Modeling Samples

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Models for All Activities

# COMMAND ----------

cost_var=["gp_Total_cost","ae_Total_cost","nel_Total_cost","el_Total_cost","op_Total_cost"]
for c in cost_var:
    gb_data_final = gb_data_final.withColumn(c, round(col(c)).cast("int"))

# COMMAND ----------

results_t = []
columns = ["ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc", "Sex", "Frailty_level"]
names = ["GP", "GP_Cost", "U111", "U999","UOoH","AE","AE_Cost", "NEl", "Nel_Cost", "EL", "EL_Cost","OP","OP_Cost", "All_Pres", "AtiB_Pres", "Depres_Pres", "Pain_Pres", "Sick_note"]
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
                        unique_counts = df_weeks_grouped.groupBy("group").agg(countDistinct("epp_pid").alias("unique_patient_count")).collect()
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

# Convert results to a DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns

total_results_h = pd.DataFrame(results_t)
total_results_h["Total Excess"] = total_results_h["Total Excess"] * 100 * 4.3

# Ensure 'heatmap_data' is already pivoted in the correct format
heatmap_dat_h = total_results_h.pivot_table(
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
    heatmap_dat_h, 
    annot=True, 
    fmt=".2f", 
    cmap=cmap, 
    norm=norm, 
    cbar=False  # Remove the color bar
)

# Labels and title
plt.title("Health Care Rate per 100 patients per one month ")
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
                                interv_person_week=totals_weeks.collect()[3][2]
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
                                        "Person-week (Intervention)":interv_person_week,
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
#file_path = os.path.join(directory, "Table4_1_Table_4_2hyster_two_groups.xlsx")

# Create the directory if it does not exist
#os.makedirs(directory, exist_ok=True)

# Write the DataFrame to an Excel file
#results_pd.to_excel(file_path, index=False)

# COMMAND ----------

total_results = pd.DataFrame(results)
# Round all floats to 2 decimal places
total_results = total_results.round(2)


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

# Pivot data for heatmap
heatmap_data = total_results.pivot_table(index="Dataset", columns="Group", values="Excess Healthcare Use (DiD Estimator)", aggfunc='mean')

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
plt.title("Heatmap of Healthcare Use Changes Across Delivery Points")
plt.xlabel("Group")
plt.ylabel("Delivery Points")
plt.show()

# COMMAND ----------

print(tmp.toPandas())

# COMMAND ----------

from pyspark.sql.functions import col, when, count, lit
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType
from scipy.stats import chi2_contingency
import pandas as pd

# Define the columns to be analyzed
columns = ["ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc", "Sex", "Frailty_level"]
personal_ch = tmp.groupBy(columns + ["epp_pid", "group"]).agg(count("*").alias("count"))

# Function to calculate chi-square test and format the results
def calculate_chisquare(df: DataFrame, column: str) -> pd.DataFrame:
    # Convert Spark DataFrame to Pandas DataFrame
    pdf = df.toPandas()
    
    # Check if the column exists in the DataFrame
    if column not in pdf.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    
    # Create a contingency table
    contingency_table = pd.crosstab(pdf[column], pdf['group'])
    
    # Perform chi-square test
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    
    # Format the results
    result = pd.DataFrame({
        'Patient pathway characteristics': [column],
        'Total': [pdf.shape[0]],
        'Bottom 90%': [contingency_table.iloc[:, 0].sum()],
        'Bottom 90% %': [contingency_table.iloc[:, 0].sum() / pdf.shape[0] * 100],
        'Top 10%': [contingency_table.iloc[:, 1].sum()],
        'Top 10% %': [contingency_table.iloc[:, 1].sum() / pdf.shape[0] * 100],
        'p-value': [p]
    })
    
    return result

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

# Convert the list of dictionaries into a DataFrame


display(total_results)

# COMMAND ----------


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

avg_results = pd.DataFrame(results)
avg_results = avg_results.round(2)
display(avg_results)

# COMMAND ----------

avg_results_6m = pd.DataFrame(results)
avg_results_6m = avg_results.round(2)
display(avg_results_6m)

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
