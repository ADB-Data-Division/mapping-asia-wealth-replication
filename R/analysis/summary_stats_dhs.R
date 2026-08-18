library(arrow)
library(dplyr)


# Load DHS responses
dhs <- read_parquet("data-clean/dhs/dhs_survey_responses.parquet")


# ---------------------- BASIC SUMMARY STATS -------------------------

# Count
nrow(dhs)

# Count of clusters
length(unique(dhs$dhs_id))

# Count, by country
table(dhs$dhs_country_code)


