library(dplyr)
library(sf)
library(terra)
library(arrow)
library(sfarrow)


# TODO:
# - Fix issue of potential imagery overlap at boundaries of folds
# - Fix issue of imbalance across population and wealth

# --- PRELIMINARIES

rm(list = ls())
set.seed(123)

# Params
NUM_FOLDS <- 5                # num folds for CV
LABEL_YEAR <- 2020            # year to filter census results to
NUM_SAMPLES <- 10000000       # num points to sample
POP_RES_DEG <- 3/3600         # population raster resolution is 3 arcseconds



# --- BUILD FOLDS

# Load census data
phl_census <- st_read("data-raw/census/PHL/census_wealth_panel_admin4_spatial.gpkg") |>
  rename_with(tolower) |>
  filter(year == LABEL_YEAR) |>
  select(geo_id, ends_with("_en"), area_sqkm, households, year, wealth_index_common, wealth_index_year_specific) |>
  st_make_valid()

# Create folds by randomly assigning ADM2 regions to folds (while equalising numbers across folds)
phl_adm2 <- unique(phl_census$adm2_en)
folds <- data.frame(adm2_en = phl_adm2, fold = sample(rep(1:NUM_FOLDS, length.out = length(phl_adm2))))

# Create labels dataset for polygons with values (availability varies by year)
phl_census <- phl_census |>
  left_join(folds, by = "adm2_en")

# Create fold bounds dataset for polygons with values
phl_fold_bounds <- phl_census |>
  summarise(
    wealth_index_year = weighted.mean(x = wealth_index_year_specific, w = households, na.rm = TRUE),
    wealth_index_common = weighted.mean(wealth_index_common, households, na.rm = TRUE),
    households = sum(households),
    area_sqkm = sum(area_sqkm),
    geom = st_union(geom),
    .by = fold
  )

st_write(phl_census, paste0("data-raw/census/PHL/phl_census_labels_", LABEL_YEAR, ".gpkg"), delete_dsn = TRUE)
st_write(phl_fold_bounds, paste0("data-raw/census/PHL/phl_census_fold_bounds_", LABEL_YEAR, ".gpkg"), delete_dsn = TRUE)



# --- BUILD SAMPLE POINTS DATASET

# Load population raster from WorldPop
pop_raster <- rast("data-raw/census/PHL/phl_pop_2025_CN_100m_R2025A_v1.tif")

# Sample N points using the population raster as weights
sample_pts <- spatSample(pop_raster, size = NUM_SAMPLES, method = "weights", na.rm = TRUE, xy = TRUE, values = TRUE) |>
  mutate(
    pt_id = paste0("PHL", sprintf("%010d", 1:n())),
    x = x + runif(n(), -POP_RES_DEG/2, POP_RES_DEG/2),     #  uniform random jitter within cell for x,y
    y = y + runif(n(), -POP_RES_DEG/2, POP_RES_DEG/2)
  ) |>
  st_as_sf(coords = c("x", "y"), crs = "OGC:CRS84")

# Merge with census and save
sample_pts <- sample_pts |>
  st_join(phl_census, join = st_intersects, left = FALSE) |>
  st_write_parquet(paste0("data-raw/census/PHL/sample_pts_with_labels_", LABEL_YEAR, ".parquet"))
