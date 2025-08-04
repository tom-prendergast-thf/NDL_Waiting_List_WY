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

sys.path.append("../src")

# COMMAND ----------

from utils_mod_lib import *


# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Read Data

# COMMAND ----------

#read GB data  - see GallBLADDER dATA FOR DATA PREP
gb_data=simulate_waiting_list_data()
#gb_data=spark.read.format("parquet").load("path_to_data")

# COMMAND ----------

# Calculate waiting time for gb_data as the difference between epp_rtt_end_date and epp_rtt_start_date
gb_data = gb_data.withColumn("wt", datediff(col("epp_rtt_end_date"), col("epp_rtt_start_date")))


# COMMAND ----------

#add detailed bang group fort table 4.6
gb_data = gb_data.withColumn(
    "ndl_wait_band_granular_c1",
    when((col("wt") >= 0) & (col("wt") <= 21), "0-3 weeks")
    .when((col("wt") >= 22) & (col("wt") <= 42), "4-6 weeks")
    .when((col("wt") >= 43) & (col("wt") <= 63), "7-9 weeks")
    .when((col("wt") >= 64) & (col("wt") <= 84), "10-12 weeks")
    .when((col("wt") >= 85) & (col("wt") <= 105), "13-15 weeks")
    .when((col("wt") >= 106) & (col("wt") <= 126), "16-18 weeks")
    .when((col("wt") >= 127) & (col("wt") <= 147), "19-21 weeks")
    .when((col("wt") >= 148) & (col("wt") <= 168), "22-24 weeks")
    .when((col("wt") >= 169) & (col("wt") <= 189), "25-27 weeks")
    .when((col("wt") >= 190) & (col("wt") <= 210), "28-30 weeks")
    .when((col("wt") >= 211) & (col("wt") <= 231), "31-33 weeks")
    .when((col("wt") >= 232) & (col("wt") <= 252), "34-36 weeks")
    .when((col("wt") >= 253) & (col("wt") <= 273), "37-39 weeks")
    .when((col("wt") >= 274) & (col("wt") <= 294), "40-42 weeks")
    .when((col("wt") >= 295) & (col("wt") <= 315), "43-45 weeks")
    .when((col("wt") >= 316) & (col("wt") <= 336), "46-48 weeks")
    .when((col("wt") >= 337) & (col("wt") <= 357), "49-51 weeks")
    .when(col("wt") >= 358, "52+ weeks")
    .otherwise(lit("unknown"))
)

# COMMAND ----------


# Calculate unique number of pathways
unique_pathways = gb_data.select("epp_pid", "epp_tfc", "epp_rtt_start_date").distinct().count()

# Calculate unique number of patients
unique_patients = gb_data.select("epp_pid").distinct().count()

# Display the results
display({"Unique Pathways": unique_pathways, "Unique Patients": unique_patients})

# COMMAND ----------

tnp_gb_data=gb_data.groupBy("epp_pid", "epp_tfc", "epp_rtt_start_date", "wt", "ndl_wait_band_granular_c1").agg(count("*").alias("count"))
cohort_stats = calculate_wait_band_distribution_report(tnp_gb_data)

# COMMAND ----------

display(cohort_stats)

# COMMAND ----------

# Convert the list of dictionaries to a DataFrame
cohort_stats_pd = cohort_stats.toPandas()

# Define the directory and file path
#directory = "../Files"
#file_path = os.path.join(directory, "Table4_6gb.xlsx")

# Create the directory if it does not exist
#os.makedirs(directory, exist_ok=True)

# Write the DataFrame to an Excel file
#cohort_stats_pd.to_excel(file_path, index=False)

# COMMAND ----------

total_count = cohort_stats.agg(spark_sum("count")).collect()[0][0]
cohort_stats = cohort_stats.withColumn("total_count", lit(total_count))
display(cohort_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Waiting Bins

# COMMAND ----------

waiting_list_df = gb_data.withColumn(
    "group",
    when(col("wt") <= 72,  "<= 6 weeks")
    .when((col("wt") > 72) & (col("wt") <= 126), "<= 18 weeks")
    .when((col("wt") > 126) & (col("wt") <= 210), "19-30 weeks")
    .when((col("wt") > 210) & (col("wt") <= 294), "31-42 weeks")
    .otherwise("> 42 weeks")
)

# COMMAND ----------

# Replace None values with a string 'Unknown' in 'epp_referral_priority' column
waiting_list_prority = waiting_list_df.fillna({'epp_referral_priority': 'Unknown'}).groupBy("epp_pid", "epp_tfc", "epp_rtt_start_date", "epp_referral_priority", "wt", "group").count()

display(waiting_list_prority.groupBy("group", "epp_referral_priority").count())
total_count = waiting_list_prority.count()
percentage_df = waiting_list_prority.groupBy("group").agg(
    (count("epp_pid") / total_count * 100).alias("percentage")
)
display(percentage_df)

# Group by 'epp_referral_priority' and count the occurrences in each group
priority_counts_df = waiting_list_prority.groupBy("epp_referral_priority").count()

# Calculate total count for percentage calculation
total_count = priority_counts_df.agg({"count": "sum"}).collect()[0][0]

# Calculate percentage
priority_counts_df = priority_counts_df.withColumn("percentage", (col("count") / total_count) * 100)

# Convert to Pandas DataFrame for plotting
priority_counts_pd = priority_counts_df.toPandas()

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(priority_counts_pd["epp_referral_priority"], priority_counts_pd["percentage"], color='skyblue')
plt.xlabel('EPP Referral Priority', fontsize=16)
plt.ylabel('Percentage', fontsize=16)
plt.title('EPP Referral Priority Percentages in Each Group', fontsize=16)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.show()

# Plot distribution of the WT
wt_data = waiting_list_prority.select("wt").toPandas()
plt.figure(figsize=(10, 6))
plt.hist(wt_data["wt"], bins=30, edgecolor='black')
plt.title("Distribution of Waiting Time (WT)", fontsize=16)
plt.xlabel("Waiting Time (days)", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

# COMMAND ----------

waiting_list_others = waiting_list_df.fillna({'epp_referral_priority': 'Unknown'}).groupBy(
    "epp_pid", "epp_tfc", "epp_rtt_start_date", "epp_referral_priority", "wt", "group", "sus_primary_diagnosis", "wlmds_priority_set_last"
).count()

display(waiting_list_others.groupBy("sus_primary_diagnosis", "epp_referral_priority").count())

waiting_list_others = waiting_list_others.withColumn("is_urgent", col("epp_referral_priority") == 'urgent')
display(waiting_list_others.groupBy("wlmds_priority_set_last", "group").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Data for DID

# COMMAND ----------

#group characteristics for all group waiters
all_waiter= waiting_list_df
all_waiter_group = all_waiter.groupBy("epp_pid", "epp_tfc", "epp_rtt_start_date", "wt", "group", "ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc", "Sex", "Frailty_level").count()
columns = ["ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc", "Sex", "Frailty_level"]
all_waiters_group_stats = calculate_wait_band_distribution_characteristics(all_waiter_group, columns)
display(all_waiters_group_stats)

# Convert the list of dictionaries to a DataFrame
#all_waiters_group_stats_pd = all_waiters_group_stats.toPandas()

# Define the directory and file path
#directory = "../Files"
#file_path = os.path.join(directory, "Table4_3gb_all_group.xlsx")

# Create the directory if it does not exist
#os.makedirs(directory, exist_ok=True)

# Write the DataFrame to an Excel file#
#all_waiters_group_stats_pd.to_excel(file_path, index=False)

# COMMAND ----------

# Filter data where wait_times > 6 weeks
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
# MAGIC ## Filter Modeling Samples

# COMMAND ----------

columns = ["ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc", "Sex", "Frailty_level"]
tmp=filter_data_function_HF_all_cohort(gb_data_final)
group_stats=calculate_wait_band_distribution_waiting_group(tmp,columns)

# COMMAND ----------

grouped_counts = tmp.groupBy("group").count()
total_count = tmp.count()
grouped_counts = grouped_counts.withColumn("percentage", (col("count") / total_count) * 100)
display(grouped_counts)

# COMMAND ----------

#did cohort stats
columns = ["ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc", "Sex", "Frailty_level"]
in_did = filter_data_function_HF_all_cohort(gb_data_final)
group_stats = calculate_wait_band_distribution_characteristics(in_did, columns)
display(group_stats)

#not in did cohort stats
not_in_did = gb_data_final.join(in_did, on=["epp_pid", "epp_tfc", "epp_rtt_start_date"], how="left_anti")
not_in_did_group = not_in_did.groupBy("epp_pid", "epp_tfc", "epp_rtt_start_date", "wt", "group", "ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc", "Sex", "Frailty_level").count()
not_in_did_group_stats = calculate_wait_band_distribution_characteristics(not_in_did_group, columns)



# COMMAND ----------

group_stats_str = group_stats.toPandas().to_csv(index=False, sep=',', lineterminator='')
display(group_stats_str)

# COMMAND ----------

# Rename columns for clarity
group_stats = group_stats.withColumnRenamed("<= 18 weeks", "18_weeks_n") \
                         .withColumnRenamed("19-30 weeks", "19_30_weeks_n") \
                         .withColumnRenamed("31-42 weeks", "31_42_weeks_n") \
                         .withColumnRenamed("> 42 weeks", "43_plus_weeks_n") \
                         .withColumnRenamed("<= 18 weeks_percentage", "18_weeks_pct") \
                         .withColumnRenamed("19-30 weeks_percentage", "19_30_weeks_pct") \
                         .withColumnRenamed("31-42 weeks_percentage", "31_42_weeks_pct") \
                         .withColumnRenamed("> 42 weeks_percentage", "43_plus_weeks_pct") \
                         .withColumnRenamed("Variable", "Category")

# Define a mapping of categories
category_mapping = {
    "Sex": ["M", "F"],
    "Age band": ["11-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "84+"],
    "IMD quintile": ["1", "2", "3", "4", "5", "unknown"],
    "Ethnicity": ["asian_background", "black_background", "mixed_background", "white_background", "unknown"],
    "Frailty_level":["Fit", "Mild", "Moderate", "Severe", "unknown"],
    "LTC": ["No LTCs","Single LTC","Comorbidities","Multimorbidities", "unknown"]
}

# Reformat the table
table_4_3 = []
for category, values in category_mapping.items():
    table_4_3.append([category, "", "", "", "", "", "", "", "", ""])  # Category header row with empty columns
    for value in values:
        row = group_stats.filter(group_stats["value"] == value).limit(1).collect()[0] if group_stats.filter(group_stats["value"] == value).count() > 0 else None
        if row is not None:
            total = row["18_weeks_n"] + row["19_30_weeks_n"] + row["31_42_weeks_n"] + row["43_plus_weeks_n"]
            table_4_3.append([value, total, 
                               row["18_weeks_n"], row["18_weeks_pct"], 
                               row["19_30_weeks_n"], row["19_30_weeks_pct"],
                               row["31_42_weeks_n"], row["31_42_weeks_pct"],
                               row["43_plus_weeks_n"], row["43_plus_weeks_pct"]])

# Convert to DataFrame
columns = ["Patient pathway characteristics", "Total", "<=18 weeks n", "<=18 weeks %", "19-30 week waiters n", "19-30 week waiters %", 
           "31-42 week waiters n", "31-42 week waiters %", "43+ week waiters n", "43+ week waiters %"]
final_df = pd.DataFrame(table_4_3, columns=columns)



#display(final_df)

# COMMAND ----------

# Apply suppression rules   
final_df_s=final_df

columns_to_check = ["Total", "<=18 weeks n", "<=18 weeks %", "19-30 week waiters n", "19-30 week waiters %", 
                    "31-42 week waiters n", "31-42 week waiters %", "43+ week waiters n", "43+ week waiters %"]

final_df_s[columns_to_check] = final_df_s[columns_to_check].apply(suppress_values, axis=1)

# COMMAND ----------


# Define the directory and file path
#directory = "../Files"
#file_path = os.path.join(directory, "Table4_3_gb.xlsx")

# Create the directory if it does not exist
#os.makedirs(directory, exist_ok=True)

# Write the DataFrame to an Excel file
#final_df_s.to_excel(file_path, index=False)

# COMMAND ----------

# Define the directory and file path
#directory = "../Files"
#file_path = os.path.join(directory, "Table4_3_gb_not_in_did.xlsx")

# Create the directory if it does not exist
#os.makedirs(directory, exist_ok=True)

# Write the DataFrame to an Excel file
#not_in_did_group_stats.toPandas().to_excel(file_path, index=False)


# COMMAND ----------

group_stats = group_stats.withColumn("value", when(col("value") == 0, "unknown").otherwise(col("value")))
group_stats = group_stats.fillna(0)
display(group_stats)

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Convert Spark DataFrame to Pandas DataFrame for plotting
group_stats_pd = group_stats.toPandas()

# Set the index to 'Variable' for easier plotting
group_stats_pd.set_index('value', inplace=True)

# Sort the DataFrame by 'Variable' and 'value'
group_stats_pd.sort_values(by=['Category', 'value'], inplace=True)

# Plotting the counts for each group
fig, ax = plt.subplots(figsize=(12, 8))
group_stats_pd[['18_weeks_pct', '19_30_weeks_pct', '31_42_weeks_pct', '43_plus_weeks_pct']].plot(kind='bar', ax=ax)
ax.set_title('Counts of Patients in Different Groups')
ax.set_xlabel('Variable and Value')
ax.set_ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Groups')
plt.tight_layout()
plt.show()

# Plotting the percentages for each group
fig, ax = plt.subplots(figsize=(12, 8))
group_stats_pd[['18_weeks_pct', '19_30_weeks_pct', '31_42_weeks_pct', '43_plus_weeks_pct']].plot(kind='bar', ax=ax)
ax.set_title('Percentage of Patients in Different Groups')
ax.set_xlabel('Variable and Value')
ax.set_ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Groups')
plt.tight_layout()
plt.show()

# COMMAND ----------

# Assuming the correct column name is 'category' instead of 'value'
heatmap_data = group_stats_pd[['18_weeks_pct', '19_30_weeks_pct', '31_42_weeks_pct', '43_plus_weeks_pct']]
#hetmap_data.set_index('value', inplace=True)

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={'label': 'Percentage'})
plt.title('Heatmap of Waiting Time Percentages Across Groups')
plt.xlabel('Waiting Time Groups')
plt.ylabel('Categories')
plt.tight_layout()
plt.show()

# COMMAND ----------

df_weeks_filtered=filter_data_function_HF_all_cohort(gb_data_final)

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

heatmap_data


# COMMAND ----------

results = []
columns = ["ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc", "Sex", "Frailty_level"]
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

import os
import pandas as pd

# Assuming results is a list of dictionaries, convert it to a DataFrame
results_pd = pd.DataFrame(results)

# Define the directory and file path
#directory = "../Files"
#file_path = os.path.join(directory, "Table4_1_Table_4_2gb.xlsx")

# Create the directory if it does not exist
#os.makedirs(directory, exist_ok=True)

# Write the DataFrame to an Excel file
#results_pd.to_excel(file_path, index=False)

# COMMAND ----------

total_results = pd.DataFrame(results)
# Round all floats to 2 decimal places
total_results = total_results.round(2)

# Save total_results to a temporary local file
#name = "gb_modelling_res.csv"
#local_path = f"/dbfs/tmp/{name}"
#total_results.to_csv(local_path, index=False)



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

# Import necessary libraries
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

import os


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
# MAGIC # Grouped >= 18 weeks 

# COMMAND ----------

from pyspark.sql.functions import col, when

grouped_more_18_weeks = gb_data_final.withColumn(
    "group",
    when( col("wt") <= 126, "<= 18 weeks")
    .otherwise("> 18 weeks")
)
display(grouped_more_18_weeks.groupby("group").count())

# COMMAND ----------

grouped_more_18_weeks = grouped_more_18_weeks.filter(col("wt") > 72)

grouped_more_18_weeks = grouped_more_18_weeks.withColumn(
    "after_clock_start",
    when(col("days") >= col("epp_rtt_start_date"), 1).otherwise(0)
).filter(col("after_clock_start") == 1)


# Add 'washout_period_end_date', 'follow_up_end_date', 'washout_period_end_days', 'follow_up_end_days' columns
grouped_more_18_weeks = grouped_more_18_weeks.withColumn(
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
total_count = grouped_more_18_weeks.count()
percentage_df = grouped_more_18_weeks.groupBy("group").agg(
    (count("epp_pid") / total_count * 100).alias("percentage")
)

display(percentage_df)

# COMMAND ----------

#how many death while waiting 
display(grouped_more_18_weeks.filter(col("REG_DATE_OF_DEATH") == col("follow_up_end_date")))
#how many death in follow up remove who died in follow up
display(grouped_more_18_weeks.count())
deaths_df = grouped_more_18_weeks.filter(col("REG_DATE_OF_DEATH") <= col("follow_up_end_date"))
distinct_patient_count = deaths_df.select("epp_pid").distinct().count()

display(distinct_patient_count)

grouped_more_18_weeks = grouped_more_18_weeks.filter(col("REG_DATE_OF_DEATH").isNull() | (col("REG_DATE_OF_DEATH") > col("follow_up_end_date")))
display(waiting_list_filtered_df.count())


# COMMAND ----------

for col_name in activity_variable:
    gb_data_final = grouped_more_18_weeks.withColumn(col_name, when(col(col_name).isNull(), "unknown").otherwise(when(col(col_name) > 0, 1).otherwise(col(col_name))))

# COMMAND ----------

cost_var=["gp_Total_cost","ae_Total_cost","nel_Total_cost","el_Total_cost","op_Total_cost"]
for c in cost_var:
    gb_data_final = gb_data_final.withColumn(c, round(col(c)).cast("int"))

# COMMAND ----------

from pyspark.sql.functions import countDistinct
results = []
columns = ["ndl_age_band", "ndl_imd_quantile", "ndl_ethnicity", "ndl_ltc", "Sex", "Frailty_level"]
names = ["GP", "GP_Cost", "U111", "U999","UOoH","AE","AE_Cost", "NEl", "Nel_Cost", "EL", "EL_Cost","OP","OP_Cost", "All_Pres", "AtiB_Pres", "Depres_Pres", "Pain_Pres", "Sick_note"]


for dataset, name in zip(activity_variable, names):
        print(name)
        #intervention_groups = ["19-30 weeks", "31-42 weeks", "> 42 weeks"]
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
                        unique_counts = df_weeks_grouped.groupBy("group").agg(countDistinct("epp_pid").alias("unique_patient_count")).collect()
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

total_results = pd.DataFrame(results)
# Round all floats to 2 decimal places
total_results = total_results.round(2)

# COMMAND ----------

total_results.to_string()

# COMMAND ----------

total_results_filtered = total_results[~total_results['Dataset'].str.contains('_Cost', case=False)]
plot_coefficients(total_results_filtered, title='')

total_results_filtered = total_results[total_results['Dataset'].str.contains('_Cost', case=False)]
plot_coefficients(total_results_filtered, title='')



# COMMAND ----------

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

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


# Pivot data for heatmap
heatmap_data = total_results.pivot_table(index="Dataset", columns="Group", values="Excess Healthcare Use (DiD Estimator)", aggfunc='mean')

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", center=0, cbar_kws={'label': 'p-value (DiD Estimator)'})
plt.title("Heatmap of Healthcare Use Changes Across Delivery Points")
plt.xlabel("Group")
plt.ylabel("Delivery Points")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # R testing- Pyfixes

# COMMAND ----------

df_weeks_grouped.createOrReplaceTempView("temp_table")

# COMMAND ----------

# MAGIC %r
# MAGIC # Load the SparkR package
# MAGIC library(SparkR)
# MAGIC
# MAGIC # Read the temporary table created in PySpark
# MAGIC df_30_weeks_grouped_u <- sql("SELECT * FROM temp_table")

# COMMAND ----------

# MAGIC %r
# MAGIC df_30_weeks_grouped_u

# COMMAND ----------

# MAGIC %r
# MAGIC # Install the fixest package if not already installed
# MAGIC if (!requireNamespace("fixest", quietly = TRUE)) {
# MAGIC   install.packages("fixest")
# MAGIC }

# COMMAND ----------

# MAGIC %r
# MAGIC df_30_weeks_grouped_l<-collect(df_30_weeks_grouped)

# COMMAND ----------

# MAGIC %r
# MAGIC # Load the fixest package
# MAGIC library(fixest)
# MAGIC
# MAGIC # Run the fixed effects regression
# MAGIC fe_ols_30weeks <- feols(
# MAGIC   avg_weekly_use ~ i(time_period, treated, ref = 0) | epp_pid + time_period,  # Interaction between time and treatment and Fixed effects
# MAGIC   data = df_30_weeks_grouped_l)
# MAGIC
# MAGIC # Display the summary of the regression
# MAGIC summary(fe_ols_30weeks)

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

# COMMAND ----------

dbutils.library.restartPython()
