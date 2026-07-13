import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("pipeline_results_az_az.tsv", sep="\t")

# 2. Simple logic to categorize errors
# Counting if there is a difference between Reference and Prediction
def categorize_error(row):
    asr_diff = row["1.Reference_Azerbaijani_Audio"] != row["2.Whisper_ASR_Prediction"]
    mt_diff = row["3.Reference_English_Target"] != row["4.MT_English_Prediction"]

    if asr_diff and mt_diff:
        return "Cascading Error (Both)"
    elif asr_diff:
        return "ASR Only"
    elif mt_diff:
        return "MT Only"
    else:
        return "Correct"

df["Error_Type"] = df.apply(categorize_error, axis=1)

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="Error_Type", palette="viridis")
plt.title("Error Distribution: Where is the Pipeline Failing?")
plt.ylabel("Number of Samples")
plt.xlabel("Error Category")
plt.tight_layout()

plt.savefig("error_analysis_chart_az.png")

summary = df["Error_Type"].value_counts(normalize=True) * 100
print("Error Summary")
print(summary)