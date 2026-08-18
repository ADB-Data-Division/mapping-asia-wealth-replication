library(dplyr)
library(sf)
library(purrr)
library(tidyr)
library(readr)
library(yardstick)
library(knitr)
library(arrow)


rm(list = ls())


# ---- PARAMS

MIN_N = 5                              # min number of observations per ADM polygon
ADM1_DIR <- "~/Projects/mapping-asia-wealth/data-raw/boundaries/ADM1"
CENSUS_PATH <- "data-raw/census/PHL/census_wealth_panel_admin4_spatial.gpkg"
DHS_CLUSTERS_PATH <- "~/Projects/mapping-asia-wealth/data-clean/dhs/dhs_clusters_asia.csv"

DATASETS <- list(
  landsat_cnn = read_parquet("data-clean/outputs/predictions/landsat-test-set-resnet50-predictions.parquet"),
  spatial = read_csv("data-clean/outputs/predictions/xgboost-spatial-only-landsat-test-set-predictions.csv"),
  embeddings = read_csv("data-clean/outputs/predictions/xgboost-embeddings-only-landsat-test-set-predictions.csv"),
  all_features = read_csv("data-clean/outputs/predictions/xgboost-all-features-landsat-test-set-predictions.csv")
)


# ----------------------------------- DATA -----------------------------------

# --- CENSUS
census <- st_read(CENSUS_PATH) |>
  mutate(
    adm1_name = case_match(
      ADM1_EN,
      "Region I (Ilocos Region)"                                ~ "Ilocos Region",
      "Region II (Cagayan Valley)"                               ~ "Cagayan Valley",
      "Region III (Central Luzon)"                               ~ "Central Luzon",
      "Region IV-A (Calabarzon)"                                 ~ "Calabarzon",
      "Region V (Bicol Region)"                                  ~ "Bicol Region",
      "Region VI (Western Visayas)"                              ~ "Western Visayas",
      "Region VII (Central Visayas)"                             ~ "Central Visayas",
      "Region VIII (Eastern Visayas)"                            ~ "Eastern Visayas",
      "Region IX (Zamboanga Peninsula)"                          ~ "Zamboanga Peninsula",
      "Region X (Northern Mindanao)"                             ~ "Northern Mindanao",
      "Region XI (Davao Region)"                                 ~ "Davao Region",
      "Region XII (Soccsksargen)"                                ~ "Soccsksargen",
      "National Capital Region (NCR)"                            ~ "NCR",
      "Cordillera Administrative Region (CAR)"                   ~ "CAR",
      "Region XIII (Caraga)"                                     ~ "Caraga",
      "Mimaropa Region"                                          ~ "Mimaropa",
      "Bangsamoro Autonomous Region In Muslim Mindanao (BARMM)"  ~ "ARMM",
    )
  )


# --- ADMIN BOUNDARIES
adm1 <- list.files(ADM1_DIR, full.names = TRUE) |>
  map(st_read) |>
  bind_rows() |>
  select(adm1_name = shapeName, adm1_iso = shapeISO, country_code_iso3 = shapeGroup) |>
  st_make_valid()


# --- PREDICTIONS
for (model_name in names(DATASETS)) {
  DATASETS[[model_name]] <- DATASETS[[model_name]] |>
    mutate(model_type = model_name)
}

model_predictions <- bind_rows(DATASETS) |>
  st_as_sf(coords = c("lon","lat"), crs = 4326, remove = FALSE) |>
  rename(country_code_dhs = country_code) |>
  st_join(adm1, join = st_intersects) |>
  filter(!is.na(country_code_iso3))



# ----------------------------- ESTIMATE PERFORMANCE -------------------------------

# Aggregate to ADM1
preds_adm1 <- model_predictions |>
  st_drop_geometry() |>
  summarise(
    country_code_iso3 = first(country_code_iso3),
    adm1_name = first(adm1_name),
    n = n(),
    mean_true = mean(labels, na.rm = TRUE),
    mean_pred = mean(outputs, na.rm = TRUE),
    year = min(survey_year),                     # some surveys extend over two calendar years; take floor
    .by = c(adm1_iso, survey_id, model_type)
  )

# Calculate diffs
adm_diffs <- preds_adm1 |>
  inner_join(preds_adm1, by = c("adm1_iso", "model_type"), suffix = c("_to", "_from"), relationship = "many-to-many") |>
  filter(year_to > year_from) |>
  mutate(
    diff_true = mean_true_to - mean_true_from,
    diff_pred = mean_pred_to - mean_pred_from,
    year_from = year_from,
    year_to   = year_to,
    lowest_n_obs = pmin(n_to, n_from)
  ) |>
  filter(!is.na(diff_true), !is.na(diff_pred), lowest_n_obs >= MIN_N)

# Intertemporal Performance - Aggregate
adm_diffs |>
  summarise(
    n = n(),
    rsq = rsq_vec(truth = diff_true, estimate = diff_pred),
    .by = model_type
    ) |>
  kable()

# Intertemporal Performance - By Country
adm_diffs |>
  summarise(
    n = n(),
    rsq = rsq_vec(truth = diff_true, estimate = diff_pred),
    .by = c(model_type, country_code_iso3_to)
  ) |>
  kable()



# --------------------- COMPARISON WITH CENSUS ------------------------

# Get DHS AWI for ADM1 regions in Philippines
phl_preds <- preds_adm1 |>
  filter(country_code_iso3 == "PHL", year %in% c(2008, 2022), model_type == "spatial") |>
  mutate(year = case_match(year, 2008 ~ 2010, 2022 ~ 2020))

# Match census to DHS regions; compare AWI
census_dhs <- census |>
  st_drop_geometry() |>
  filter(year %in% c(2010, 2020)) |>
  summarise(
    wealth_index_year_specific = sum(wealth_index_year_specific * (households/sum(households))),
    wealth_index_common = sum(wealth_index_common * (households/sum(households))),
    .by = c(adm1_name, year)
  ) |>
  left_join(phl_preds, by = join_by(adm1_name == adm1_name, year == year))

