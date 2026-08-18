library(tidyverse)
library(broom)
library(arrow)
library(dplyr)
library(sf)
library(yardstick)
library(knitr)



rm(list = ls())

# Params
CENSUS_PATH <- "data-raw/census/PHL/census_wealth_panel_admin4_spatial.gpkg"
YEARS <- c(2010, 2020)
IMAGERY_CRS <- "epsg:32651"
GEOSAMPLED_MODEL <- FALSE
CENSUS_WEALTH_INDEX_NAME <- "wealth_index_common"

PREDS_STEMS <- c(
  landsat_cnn = "data-clean/outputs/model_predictions/country/phl-resnet50-best-resnet-model-",
  landsat_xgb = "data-clean/outputs/model_predictions/country/phl-resnet50-best-resnet-model-xgb-embed-",
  spatial_xgb = "data-clean/outputs/model_predictions/country/phl-resnet50-best-resnet-model-xgb-spatial-",
  all_xgb = "data-clean/outputs/model_predictions/country/phl-resnet50-best-resnet-model-xgb-all-"
)


# -------------- DATA PREP -----------------

# Load Fold Bounds
fold_bounds <- st_read("data-raw/census/PHL/phl_census_fold_bounds_2020.gpkg") |>
  st_transform(crs = IMAGERY_CRS) |>
  select(fold)


# Aggregate to ADM4
preds_list <- list()
for (year in YEARS) {

  # Load census data
  census <- st_read(CENSUS_PATH) |>
    st_transform(IMAGERY_CRS) |>
    filter(year == .env$year) |>
    select(adm4_id = ADM4_PCODE, adm3_id = ADM3_PCODE, adm2_id = ADM2_PCODE, adm1_id = ADM1_PCODE, wealth_index = CENSUS_WEALTH_INDEX_NAME)


  for (model in names(PREDS_STEMS)) {

    # Load predictions
    stem = PREDS_STEMS[[model]]
    preds <- read_parquet(paste0(stem, year, "-sampled-weighted.parquet")) |>
      st_set_geometry("geometry") |>
      st_set_crs(IMAGERY_CRS) |>
      st_join(census, predicate=st_intersects, left=FALSE) |>
      st_drop_geometry() |>
      summarise(
        ensemble = mean(outputs),
        labels = mean(labels),
        census = mean(wealth_index),
        year = .env$year,
        .by = c(adm4_id, adm3_id, adm2_id, adm1_id, sample_id)
      ) |>
      mutate(model_type = model)

    preds_list[[paste(model, year)]] <- preds

    # Performance
    print(paste("Model: ", model, "   Year: ", year))
    print(paste("Patch-level R2:", rsq(preds, truth = labels, estimate = ensemble)$.estimate))
    print(paste("Label-Census R2:", rsq(preds, truth = labels, estimate = census)$.estimate))

    }
  }

all_preds <- bind_rows(preds_list)


# ------------- ANALYSIS ----------------

# Predictiosn in cross-section, population-weighted
all_preds |>
  summarise(
    n = n(),
    pearson = cor(ensemble, labels, use = "complete.obs"),
    r2 = rsq_vec(truth = labels, estimate = ensemble),
    spearman = cor(ensemble, labels, method = "spearman", use = "complete.obs"),
    rmse = sqrt(mean((ensemble - labels)^2, na.rm = TRUE)),
    mae = mean(abs(ensemble - labels), na.rm = TRUE),
    bias = mean(ensemble - labels, na.rm = TRUE),
    sd_pred = sd(ensemble, na.rm = TRUE),
    sd_census = sd(labels, na.rm = TRUE),
    .by = model_type
  ) |>
  kable()



# ---- INTERTEMPORAL

# Calculate intertemporal changes at patch level
change_patch <- all_preds |>
  arrange(model_type, sample_id, year) |>
  mutate(
    lag_year = lag(year),
    lag_pred = lag(ensemble),
    lag_census = lag(census),
    lag_label = lag(labels),
    d_pred = ensemble - lag_pred,
    d_census = census - lag_census,
    d_label = labels - lag_label,
    period = paste0(lag_year, "-", year),
    .by = c(sample_id, model_type)
  ) |>
  filter(!is.na(d_pred), !is.na(d_census))


# Aggregate results and calculate R2 at various Admin levels
adm_levels <- c(patch = "sample_id", adm1 = "adm1_id", adm2 = "adm2_id", adm3 = "adm3_id", adm4 = "adm4_id")
adm_results <- list()
for (level_name in names(adm_levels)) {

  id_col <- adm_levels[[level_name]]
  adm_results[[level_name]] <- change_patch |>
    summarise(
      d_pred = mean(d_pred),
      d_census = mean(d_census),
      d_label = mean(d_label),
      .by = c(all_of(id_col), period, model_type)
    ) |>
    summarise(
      n = n(),
      rsq = rsq_vec(truth = d_label, estimate = d_pred),
      pearson_change = cor(d_pred, d_label, use = "complete.obs"),
      spearman_change = cor(d_pred, d_label, method = "spearman", use = "complete.obs"),
      .by = model_type
    ) |>
    mutate(level = level_name, .before = 1)

}

bind_rows(adm_results) |>
  kable()


# Insufficient shift in predicted distribution; doesn't keep up with census
ggplot(change_patch) +
  geom_density(aes(x = d_label), fill = "blue", alpha = 0.3) +
  geom_density(aes(x = d_pred), fill = "red", alpha = 0.3) +
  facet_wrap(~model_type) +
  theme_minimal()
