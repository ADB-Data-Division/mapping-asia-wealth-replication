# Clean DHS Survey Questionnaires and Extract Asset-Related Questions.
#
# NOTES:
# - Questionnaire codes stable across surveys and rounds. Definitions available in DHS Recode Manual/Map.
# - See: https://dhsprogram.com/publications/publication-dhsg4-dhs-questionnaires-and-manuals.cfm
# - For RWI by Rural/Urban, use c("hv270a", "hv271a") for supported surveys.


library(haven)
library(dplyr)
library(readr)
library(sf)
library(stringr)
library(purrr)
library(ggplot2)
library(arrow)
library(dbscan)
library(tidyr)

# Env.
rm(list = ls())
set.seed(123)


# --------------------------------- PARAMS -----------------------------------

# File Path Parameters (**change these**)
HOME_DIR <- "/Users/jordanimahori/Projects/mapping-asia-wealth"   # project directory
DHS_DIR <- file.path(HOME_DIR, "data-raw", "dhs")                 # path to DHS survey zips
SAVE_DIR <- file.path(HOME_DIR, "data-clean", "dhs")              # path to store joined and cleaned labels file

# Cross-Validation Fold Params
NUM_FOLDS = 5                                     # number of folds to divide dataset into
MIN_CLUSTERS_KMEANS <- 10                         # min K for each country is 10
TARGET_CLUSTER_SIZE_KMEANS <- 250                 # choose K such that the average cluster should have around 250

DBSCAN_EPS_LANDSAT = 2*sqrt(2*3360^2)             # set distance to worst-case overlap
DBSCAN_EPS_SENTINEL = 2*sqrt(2*2500^2)            # set distance to worst-case overlap
DBSCAN_MAX_CLUSTER_PROP = 0.1                     # break clusters w. > 10% of obs and re-cluster using K-Means
DBSCAN_MIN_PTS = 2                                # must be connected to at least two images to continue cluster


GEO_GRID_SIZE <- 100000                           # size in meters of grid to partition obs into


# Variables to select: ID, basic cluster characteristics, asset-related questions
VARS <- c("hv000", "hv001", "hv002", "hv006", "hv007", "hv009", "hv201", "hv205", "hv206", "hv207", "hv208", "hv209", "hv210",
          "hv211", "hv212", "hv213", "hv214", "hv215", "hv221", "hv226", "hv225", "hv270", "hv271")

PCA_VARS <- c("hv205", "hv206", "hv207", "hv208", "hv209", "hv210", "hv211", "hv212", "hv213", "hv214", "hv215")



# --------------------- CLEAN INDIA'S DHS SURVEYS -----------------------

# India's DHS Surveys are too large to extract and load entirely into memory for most computers. Instead, you need to
# manually pre-process the DTA file in STATA by dropping irrelevant columns. I've done this and re-zipped the HR file
# with the same name (for compatibility with above), alongside the corresponding GPS file.

# To do this, in STATA:
# > use path/to/your/household-recode-file.dta
# > keep hv000 hv001 hv002 hv006 hv007 hv009 hv015 hv205 hv206 hv207 hv208 hv209 hv210 hv211 hv212 hv213 hv214 hv215 hv225 hv270 hv271
# > save path/to/your/new-household-recode-file-with-same-name.dta
# NOTE: If you change the VARS above (the variables to keep) you'll need to change them here too.
# NOTE ALSO: In addition to VARS, add hv015 since this is needed for validation in the clean_survey function.



# -------------------------- PRELIMINARIES -----------------------------

# Clean Household Recode file and merge with GPS coordinates for an individual survey
clean_survey <- function(hr_path, gps_path) {

  print(paste0("Processing Household Recode and GPS Files: ",
               str_sub(hr_path, start = -12L, end = -1L), " & ",
               str_sub(gps_path, start = -12L, end = -1L)))

  # Clean Cluster GPS Locations
  gps <- st_read(gps_path, quiet = TRUE)
  coords <- st_coordinates(gps)
  gps$lon <- coords[, 1]
  gps$lat <- coords[, 2]

  gps <- gps |>
    mutate(
      rural = case_when(
        trimws(tolower(URBAN_RURA)) == "r" ~ 1,
        trimws(tolower(URBAN_RURA)) == "u" ~ 0
      )
    ) |>
    rename(
      dhs_id = DHSID,
      dhs_country_code = DHSCC,
      cluster_id = DHSCLUST
    ) |>
    select(dhs_id, dhs_country_code, cluster_id, rural, lon, lat) |>
    filter(!(abs(lat) < 1e-5 & abs(lon) < 1e-5)) |>      # drop observations with missing GPS coordinates set to (0, 0)
    st_drop_geometry()

  # Clean Household Recode
  hr <- read_dta(hr_path)
  hr_clean <- hr |>
    filter(hv015 == 1) |>                                # filter out any unsuccessful surveys
    select(any_of(VARS)) |>                              # select any matching columns in VARS
    mutate(
      across(hv001:last_col(), as.integer),                       # convert responses to integer
      across(hv206:hv212, ~ if_else(.x <= 1, .x, NA_integer_))    # recode values > 1 as NA (i.e. missing values)
    ) |>
    mutate(
      hv000 = as.character(hv000),
      hv205 = case_when(                                 # toilet facility: -1 = low, 1 = high
        between(hv205, 10, 19) ~ 1,                           # flush toilet
        between(hv205, 20, 29) | between(hv205, 40, 49) ~ 0,  # pit latrine or approx. equivalent
        between(hv205, 30, 39) ~ -1,                          # no facility
        .default = NA_integer_
      ),
      hv213 = case_when(                                 # floor material: -1 = low, 1 = high
        between(hv213, 10, 19) ~ -1,                          # natural
        between(hv213, 20, 29) ~ 0,                           # rudimentary
        between(hv213, 30, 39) ~ 1,                           # finished
        .default = NA_integer_
      ),
      hv214 = case_when(                                 # wall material: -1 = low, 1 = high
        between(hv214, 10, 19) ~ -1,                          # natural
        between(hv214, 20, 29) ~ 0,                           # rudimentary
        between(hv214, 30, 39) ~ 1,                           # finished
        .default = NA_integer_
      ),
      hv215 = case_when(                                 # roof material: -1 = low, 1 = high
        between(hv215, 10, 19) ~ -1,                          # natural
        between(hv215, 20, 29) ~ 0,                           # rudimentary
        between(hv215, 30, 39) ~ 1,                           # finished
        .default = NA_integer_
      )
    ) |>
    inner_join(gps, by = join_by(hv001 == cluster_id))  # join with coordinates from GPS file

  missing_vars <- setdiff(VARS, names(hr_clean))
  if (length(missing_vars) > 0) {
    message("... Warning! Not all variables found. Missing: ", paste(missing_vars, collapse=", "))
  }
  print(paste0("... Success! ", nrow(hr_clean), " out of ", nrow(hr), " passed validation."))
  return(hr_clean)
}



# ------------------------ EXTRACT AND CLEAN HOUSEHOLD RECODE SURVEYS + GPS ----------------------------

# Extract DHS Survey Zips, locate the HR + GPS files, pass to clean_survey and merge all results
file_list <- list.files(DHS_DIR, full.names = TRUE, recursive = TRUE, pattern = "\\.(zip|ZIP)$")
all_surveys <- NULL

for (file in file_list) {

  # Unzip DHS survey
  temp_dir <- tempfile("dhs_zip_")
  dir.create(temp_dir)
  unzip(file, exdir = temp_dir)

  # Match Household Recode and GPS files by expected file name
  hr_path <- list.files(temp_dir, full.names = TRUE, recursive = TRUE,
                        pattern = "[A-Za-z]{2}HR[A-Za-z0-9]{2}FL\\.(dta|DTA)$")
  gps_path <- list.files(temp_dir, full.names = TRUE, recursive = TRUE,
                         pattern = "[A-Za-z]{2}GE[A-Za-z0-9]{2}FL\\.(shp|SHP)$")

  # Fail if multiple matches (print a warning and skip this file)
  if (length(hr_path) != 1) {
    print(paste0("Warning! Expected only 1 matching Household Recode file but found ",
                 length(hr_path), ": \n", paste(hr_path, collapse = ", ")))
    unlink(temp_dir, recursive = TRUE)
    next
  }
  if (length(gps_path) != 1) {
    print(paste0("Warning! Expected only 1 matching Cluster GPS file but found ",
                 length(gps_path), ": \n", paste(gps_path, collapse = ", ")))
    unlink(temp_dir, recursive = TRUE)
    next
  }

  # Try to clean survey
  cleaned_hr <- tryCatch({
    clean_survey(hr_path = hr_path, gps_path = gps_path)
  }, error = function(e) {
    print(paste0("Error processing survey ", hr_path, ": ", e$message))
    return(NULL)
  })

  # And if cleaning failed, skip to next zip file
  if (is.null(cleaned_hr)) {
    unlink(temp_dir, recursive = TRUE)
    next
  }

  if (is.null(all_surveys)) {     # otherwise merge with other surveys in-memory
    all_surveys <- cleaned_hr
  } else {
    all_surveys <- bind_rows(all_surveys, cleaned_hr)
  }

  unlink(temp_dir, recursive = TRUE)

}


# Post-Processing to Fix Identified Errors
all_surveys <- all_surveys |>
  mutate(
    # Fix issue with Nepal having incorrect visitation dates...
    hv007 = case_when(
      hv000 == "NP8" ~ 2022,
      hv000 == "NP7" ~ 2016,
      hv000 == "NP6" ~ 2011,
      hv000 == "NP5" ~ 2006,
      hv000 == "NP4" ~ 2001,
      .default = hv007
    ),
    # Fix issue where India country code incorrectly given as 'IN'
    dhs_country_code = if_else(dhs_country_code == "IN", "IA", dhs_country_code)
  )


# Save
write_parquet(all_surveys, file.path(SAVE_DIR, "dhs_survey_responses.parquet"))



# ---------------------------------- CALCULATE ASSET WEALTH INDEX -------------------------------------

# Drop observations with at least one missing asset-related question; define wealth_index as PC1 of PCA
all_surveys <- all_surveys |>
  drop_na(all_of(PCA_VARS))

all_surveys$wealth_index <- prcomp(select(all_surveys, all_of(PCA_VARS)), center = TRUE, scale = TRUE)$x[,1]


# Summarise by cluster
dhs_clusters <- all_surveys |>
  summarise(
    survey_id = unique(hv000),
    dhs_round = unique(str_sub(hv000, 3)),
    country_code = unique(dhs_country_code),
    survey_year = first(hv007),                         # some surveys spread across 2 years
    rural = unique(rural),
    lat = unique(lat),
    lon = unique(lon),
    wealth_index = mean(wealth_index, na.rm = TRUE),
    wealth_index_dhs = mean(hv271, na.rm = TRUE),
    .by = dhs_id
  ) |>
  mutate(wealth_index = (wealth_index - mean(wealth_index)) / sd(wealth_index)) |>   # normalise AWI across sample
  mutate(
    wealth_index_dhs = (wealth_index_dhs - mean(wealth_index_dhs)) / sd(wealth_index_dhs),   # normalise DHS index within survey
    .by = survey_id
  )


# --- Checks

if (!(nrow(dhs_clusters) == length(unique(dhs_clusters$dhs_id)))) {
  stop("Duplicated IDs detected. One or more vars in all_surveys is not unique within clusters")
}

if (cor(dhs_clusters$wealth_index, dhs_clusters$wealth_index_dhs) < 0.90) {
  stop("Lower than expected correlation between AWI and DHS RWI. Confirm PC1 wealth index is as expected.")
}

if (nrow(dhs_clusters) != 72298) {
  stop("More/fewer clusters than expected. Should be 72,298.")
}



# ------------------------ CREATE FOLDS FOR SPATIAL CROSS-VALIDATION ----------------------------

# --- HELPERS

utm_zone_from_lon <- function(lon) {
  floor((lon + 180) / 6) + 1
}


create_folds_greedy <- function(cluster_assignments, num_folds) {

  cl <- as.character(cluster_assignments)
  cluster_counts <- data.frame(cluster_id = cl) |>              # get cluster sizes
    count(cluster_id, name = "n") |>
    arrange(desc(n))

  fold_by_cluster <- setNames(rep(NA_integer_, nrow(cluster_counts)),
                              cluster_counts$cluster_id)

  fold_counts <- rep(0L, num_folds)
  for (i in seq_len(nrow(cluster_counts))) {
    cid <- cluster_counts$cluster_id[i]
    f <- which.min(fold_counts)
    fold_by_cluster[cid] <- f                                   # assign each to the fold with fewest obs
    fold_counts[f] <- fold_counts[f] + cluster_counts$n[i]
  }
  return(fold_by_cluster[cl])                                   # map each observation’s cluster label back to its fold
}


dbscan_split <- function(xy, eps, minPts, max_cluster_size) {

  db <- dbscan(xy, eps = eps, minPts = minPts)                      # cluster using dbscan
  clusters <- db$cluster
  clusters[clusters == 0] <- 1:sum(clusters == 0) + max(clusters)   # assign each noise point to a unique cluster

  next_id <- max(clusters)
  final_clusters <- clusters
  for (id in unique(clusters)) {

    idx <- which(clusters == id)
    if (length(idx) > max_cluster_size) {                           # if the cluster size > max_size, split it using k-means

      num_splits <- ceiling(length(idx) / max_cluster_size)
      kmeans_split <- kmeans(xy[idx, , drop = FALSE], centers = num_splits)

      for (j in seq_len(num_splits)) {
        next_id <- next_id + 1
        final_clusters[idx[kmeans_split$cluster == j]] <- next_id
      }
    }
  }
  return(final_clusters)
}


# --- CREATE SPATIAL CLUSTERS AND ASSIGN OBSERVATIONS INTO FOLDS

countries <- unique(dhs_clusters$country_code)
country_results <- list()
for (ctry in countries) {

  # Convert to proj CRS with locally approximately optimal UTM zone
  country_subset <- filter(dhs_clusters, country_code == ctry)
  utm_zone <- utm_zone_from_lon(mean(country_subset$lon))
  num_clusters <- max(ceiling((nrow(country_subset) / TARGET_CLUSTER_SIZE_KMEANS)), MIN_CLUSTERS_KMEANS)
  cluster_prefix <- paste0(ctry, "TCS", TARGET_CLUSTER_SIZE_KMEANS, "MC", MIN_CLUSTERS_KMEANS, "C")

  country_geo <- country_subset |>
    st_as_sf(coords = c("lon", "lat"), crs = "EPSG:4326", remove = FALSE) |>
    st_transform(paste0("EPSG:326", utm_zone))

  country_xy <- st_coordinates(country_geo)

  # --- Cluster using KMeans
  kmeans_result <- kmeans(country_xy, centers = num_clusters, nstart = 10, iter.max=100)
  country_geo$cluster_kmeans <- paste(cluster_prefix, sprintf("%04d", kmeans_result$cluster), sep = "-")

  # --- Cluster Using Geographic Blocking (i.e. a grid)
  grid <- country_geo |>
    st_make_grid(cellsize = GEO_GRID_SIZE) |>
    st_as_sf(crs = paste0("EPSG:326", utm_zone)) |>
    mutate(
      cluster_grid = paste(ctry, "GRID", sprintf("%06d", 1:n()), sep = "-")
    )

  country_geo <- st_join(country_geo, grid, join = st_intersects)
  stopifnot(nrow(country_geo) == nrow(country_subset))           # break if st_intersects yields duplicates e.g. boundary points

  # --- Cluster using DBSCAN
  dbscan_max_cluster_size = nrow(country_geo)*DBSCAN_MAX_CLUSTER_PROP
  dbscan_ls <- dbscan_split(xy = country_xy, eps = DBSCAN_EPS_LANDSAT, minPts = DBSCAN_MIN_PTS,
                            max_cluster_size = dbscan_max_cluster_size)
  dbscan_s2 <- dbscan_split(xy = country_xy, eps = DBSCAN_EPS_SENTINEL, minPts = DBSCAN_MIN_PTS,
                            max_cluster_size = dbscan_max_cluster_size)

  country_geo$cluster_dbscan_ls <- paste(ctry, "DBSCAN-LS", sprintf("%06d", dbscan_ls), sep = "-")
  country_geo$cluster_dbscan_s2 <- paste(ctry, "DBSCAN-S2", sprintf("%06d", dbscan_s2), sep = "-")

  # Map the cluster assignments back to each observation for fold assignment
  country_results[[ctry]] <- country_geo |>
    st_drop_geometry() |>
    mutate(
      fold_kmeans = create_folds_greedy(cluster_assignments = cluster_kmeans, num_folds = NUM_FOLDS),
      fold_grid = create_folds_greedy(cluster_assignments = cluster_grid, num_folds = NUM_FOLDS),
      fold_dbscan_ls = create_folds_greedy(cluster_assignments = cluster_dbscan_ls, num_folds = NUM_FOLDS),
      fold_dbscan_s2 = create_folds_greedy(cluster_assignments = cluster_dbscan_s2, num_folds = NUM_FOLDS),
    )

}


# Recombine the country subsets
dhs_clusters <- bind_rows(country_results) |>
  rename(patch_id = dhs_id)


# ---- SAVE

# Save as Geo
dhs_clusters |>
  st_as_sf(coords = c("lon", "lat"), remove = FALSE, crs = "EPSG:4326") |>
  st_write(dsn = file.path(SAVE_DIR, "dhs_clusters_asia.geojson"), delete_dsn = TRUE)

# Save as CSV
write_csv(dhs_clusters, file = file.path(SAVE_DIR, "dhs_clusters_asia.csv"))
