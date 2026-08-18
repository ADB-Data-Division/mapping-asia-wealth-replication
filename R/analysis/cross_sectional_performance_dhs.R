library(readr)
library(dplyr)
library(tidyr)
library(yardstick)
library(knitr)
library(ggplot2)
library(arrow)

rm(list = ls())



# --------------------------------------- DATA -------------------------------------------------


# Translate DHS country codes to proper country names
country_map <- c(BD = "Bangladesh", IA = "India", ID = "Indonesia", KH = "Cambodia", KY = "Kyrgz Republic",
                 MM = "Myanmar", NP = "Nepal", PH = "Phillipines", PK = "Pakistan", TL = "Timor Leste")

# Predictions from CNNs / XGBoost Models
DATASETS = list(
  landsat_spatial_only = read_csv("data-clean/outputs/predictions/xgboost-spatial-only-landsat-test-set-predictions.csv"),
  sentinel_spatial_only = read_csv("data-clean/outputs/predictions/xgboost-spatial-only-sentinel-test-set-predictions.csv"),
  landsat_embed = read_csv("data-clean/outputs/predictions/xgboost-embeddings-only-landsat-test-set-predictions.csv"),
  sentinel_embed = read_csv("data-clean/outputs/predictions/xgboost-embeddings-only-sentinel-test-set-predictions.csv"),
  aef_embed = read_csv("data-clean/outputs/predictions/xgboost-embeddings-only-aef-test-set-predictions.csv"),
  landsat_all = read_csv("data-clean/outputs/predictions/xgboost-all-features-landsat-test-set-predictions.csv"),
  sentinel_all = read_csv("data-clean/outputs/predictions/xgboost-all-features-sentinel-test-set-predictions.csv"),
  aef_all = read_csv("data-clean/outputs/predictions/xgboost-all-features-aef-test-set-predictions.csv"),
  landsat_resnet50 = read_parquet("data-clean/outputs/predictions/landsat-test-set-resnet50-predictions.parquet"),
  sentinel_resnet50 = read_parquet("data-clean/outputs/predictions/sentinel-test-set-resnet50-predictions.parquet"),
  sentinel_swinv2 = read_parquet("data-clean/outputs/predictions/sentinel-test-set-swinv2-predictions.parquet")
)

# Add model name as a column and merge
for (model_name in names(DATASETS)) {
  DATASETS[[model_name]] <- DATASETS[[model_name]] |>
    mutate(model_type = model_name)
}

model_predictions <- bind_rows(DATASETS) |>
  select(
    survey_id, country_code, patch_id, survey_year, model_type,
    fold_kmeans, wealth_index, labels, outputs, rural, lat, lon
  ) |>
  mutate(
    rural = if_else(rural == 1, "Rural", "Urban"),
    country = country_map[country_code],
    model_name = case_match(
      model_type,
      "sentinel_swinv2" ~ "SwinV2T - Sentinel",
      "landsat_resnet50" ~ "Resnet50 - Landsat",
      "sentinel_resnet50" ~ "Resnet50 - Sentinel",
      "aef_all" ~ "XGB - AEF + Covs",
      "aef_embed" ~ "XGB - AEF",
      "landsat_all" ~ "XGB - Landsat + Covs",
      "landsat_embed" ~ "XGB - Landsat",
      "landsat_spatial_only" ~ "XGB - Covs. (6270m)",
      "sentinel_spatial_only" ~ "XGB - Covs. (5200m)",
      "sentinel_embed" ~ "XGB - Sentinel",
      "sentinel_all" ~ "XGB - Sentinel + Covs."
    )
  ) |>
  mutate(
    true_wealth_quintile = ntile(wealth_index, 5),
    .by = survey_id
  )

# Sanity Check: Confirm all labels align with wealth index (should be small)
sum(abs(model_predictions$wealth_index - model_predictions$labels))



# ---------------------------------- CROSS-SECTIONAL ANALYSIS -------------------------------------

# We use 5-fold CV to train our models. Each model 'sees' 4 folds, and we test its performance on the 5th 'held-out'
# fold, given in the 'fold_kmeans' column (which really outght to be called the held_out_fold).


# --- Performance by Fold

fold_results <- model_predictions |>
  summarise(
    r2 = rsq_vec(truth = labels, estimate = outputs),
    mae = mae_vec(truth = labels, estimate = outputs),
    rmse = rmse_vec(truth = labels, estimate = outputs),
    .by = c(model_type, fold_kmeans)
  ) |>
  pivot_longer(
    cols = c(r2, mae, rmse),
    names_to = "metric",
    values_to = "value"
  ) |>
  select(model_type, fold_kmeans, metric, value) |>
  pivot_wider(
    names_from = fold_kmeans,
    values_from = value,
    names_prefix = "fold_"
  ) |>
  rowwise() |>
  mutate(all_fold_average = mean(c_across(starts_with("fold_")))) |>
  arrange(model_type, metric)


# --- Performance by Country

country_results <- model_predictions |>
  summarise(
    r2 = rsq_vec(truth = labels, estimate = outputs),
    mae = mae_vec(truth = labels, estimate = outputs),
    rmse = rmse_vec(truth = labels, estimate = outputs),
    .by = c(model_type, country, fold_kmeans)
  ) |>
  summarise(
    r2 = mean(r2),
    mae = mean(mae),
    rmse = mean(rmse),
    .by = c(model_type, country)
  ) |>
  pivot_longer(
    cols = c(r2, mae, rmse),
    names_to = "metric",
    values_to = "value"
  ) |>
  select(model_type, country, metric, value) |>
  pivot_wider(
    names_from = country,
    values_from = value
  ) |>
  arrange(model_type, metric)


# Print results
fold_results |>
  filter(metric == "r2") |>
  select(!metric) |>
  kable()

country_results |>
  filter(metric == "r2") |>
  select(!metric) |>
  kable()



# --- PERFORMANCE BY GROUPS

# By Urban - Rural
model_predictions |>
  summarise(
    rsq = rsq_vec(truth = wealth_index, estimate = outputs),
    .by = c(model_name, rural)
  ) |>
  kable()

# By True Wealth Quintile (within each survey)
model_predictions |>
  summarise(
    rsq = rsq_vec(truth = wealth_index, estimate = outputs),
    .by = c(model_name, true_wealth_quintile)
  ) |>
  kable()



# ------------------------------------ PLOTS --------------------------------------


# Plots -- XGB Models
model_predictions |>
  filter(!(model_type %in% c("sentinel_swinv2", "sentinel_resnet50", "landsat_resnet50"))) |>
  ggplot() +
  geom_point(aes(x = labels, y = outputs, colour = rural), alpha = 0.25) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.5) +
  facet_wrap(~ model_name, ncol = 4) +
  labs(
    x = "Observed Wealth",
    y = "Predicted Wealth",
    colour = ""
  ) +
  theme_minimal()


# Plots --- DL Models
model_predictions |>
  filter(model_type %in% c("sentinel_swinv2", "sentinel_resnet50", "landsat_resnet50")) |>
  ggplot() +
  geom_point(aes(x = labels, y = outputs, colour = rural), alpha = 0.25) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.5) +
  facet_wrap(~ model_name, ncol = 3) +
  labs(
    x = "Observed Wealth",
    y = "Predicted Wealth",
    colour = ""
  ) +
  theme_minimal()
