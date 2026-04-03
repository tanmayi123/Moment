from google.cloud import bigquery
import os 

client = bigquery.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT", "moment-486719"))

destination = "moment-486719.moments_staging_manual__2026_04_03_15_09_38_480816_00_00.decompositions"
source = "moment-486719.new_moments_processed.decompositions"

job_config = bigquery.CopyJobConfig(
    write_disposition=bigquery.WriteDisposition.WRITE_APPEND
)

# Positional args: (source, destination)
job = client.copy_table(destination, source, job_config=job_config)
job.result()

print("Destination table appended into source table.")