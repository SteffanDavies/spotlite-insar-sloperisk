#' @title Predict Movement and Assess Slope Risk
#'
#' @description
#' This workflow performs movement prediction using a pre-trained model and processes
#' the results to assess slope stability risk. It combines prediction outputs with
#' original spatial data to compute and visualize risk classifications and PCA-based
#' summaries.
#'
#' @details
#' The workflow has two main stages:
#' \enumerate{
#'   \item **Prediction Stage**: Uses `perform_prediction()` to generate predicted
#'   displacements for specified time intervals and save results to the working directory.
#'   \item **Risk Processing Stage**: Uses `risk_processing()` to compute risk classes
#'   from the predicted results, apply PCA for dimensionality reduction, and join
#'   spatial attributes from the original CSV to create \code{sf} (spatial feature)
#'   objects ready for mapping or GIS analysis.
#' }
#'
#' The outputs include:
#' \itemize{
#'   \item `com12D` — Combined 12-day prediction dataset.
#'   \item `area_risk` — Risk classification data.
#'   \item `com12D_sf` — Spatial version of `com12D` as an \code{sf} object.
#'   \item `area_risk_sf` — Spatial version of `area_risk` as an \code{sf} object.
#'   \item `summary_pca` — Summary statistics of the PCA analysis.
#'   \item `rotation_pca` — PCA component loadings.
#' }
#'
#' @param model_name Character string specifying the trained model file (e.g. `"Smovement_I22.model"`).
#' @param new_dataset_name Character string for the CSV file containing new data for prediction
#' (e.g. `"ver_A13.csv"`).
#' @param intervals Numeric vector specifying time intervals to process (e.g. `c(12)`).
#' @param pred_times Numeric vector specifying the prediction horizon (e.g. `c(12)`).
#' @param WDir Character string specifying the working directory path.
#' @param saveresults Character string indicating the folder name or prefix for saving results.
#' @param centers Numeric; number of clusters for the risk classification (default = 5).
#' @param alpha Numeric; weighting factor for primary component (default = 0.7).
#' @param beta Numeric; weighting factor for secondary component (default = 0.3).
#' @param original_csv Character string specifying the path to the original spatial CSV file
#' containing point coordinates and attributes.
#'
#' @return
#' A list with the following elements:
#' \item{com12D}{Predicted displacement dataset for the specified interval.}
#' \item{area_risk}{Risk classification dataset.}
#' \item{com12D_sf}{Spatial `sf` object of `com12D`.}
#' \item{area_risk_sf}{Spatial `sf` object of `area_risk`.}
#' \item{summary_pca}{Summary of PCA results.}
#' \item{rotation_pca}{Loadings of PCA components.}
#'
#' @examples
#' \dontrun{
#' # Step 1: Perform prediction
#' predict_movement <- perform_prediction(
#'   model_name = "Smovement_I22.model",
#'   new_dataset_name = "ver_A13.csv",
#'   intervals = c(12),
#'   pred_times = c(12),
#'   WDir = "C:/Users/domin/Documents/slope/risk",
#'   saveresults = "risk"
#' )
#'
#' # Step 2: Process risk and generate spatial outputs
#' risk_result <- risk_processing(
#'   pred_name = "risk.pred",
#'   centers = 5,
#'   alpha = 0.7,
#'   beta = 0.3,
#'   WDir = "C:/Users/domin/Documents/slope/risk",
#'   original_csv = "A13_vertical_2016_2023_clipped.csv",
#'   saveresults = "A13_risk_results"
#' )
#'
#' # Extract results
#' com12D_sf <- risk_result$com12D_sf
#' area_risk_sf <- risk_result$area_risk_sf
#'
#' # Optional: Export spatial results
#' # st_write(com12D_sf, "com12D_risk.gpkg", delete_dsn = TRUE)
#' # st_write(area_risk_sf, "area_risk.gpkg", delete_dsn = TRUE)
#' }
#'
#' @seealso
#' \code{\link{perform_prediction}}, \code{\link{risk_processing}}, \code{\link{st_write}}
#'
#' @author
#' Dominic Owusu-Ansah



# Call the perform_prediction function
predict_movement <- perform_prediction(model_name = "Smovement_I22.model", 
                                       new_dataset_name = "ver_A13.csv", ## direct long term 48 days dataset
                                       intervals = c(12),
                                       pred_times = c(12),
                                       WDir = "C:/Users/domin/Documents/slope/risk",
                                       saveresults = "risk")






#' @title Dynamic Risk Processing and Classification Function
#'
#' @description
#' Performs unsupervised slope stability risk classification using PCA and K-means.
#' The function dynamically adapts class labels (A–D, A–E, etc.) and scoring to
#' the number of clusters (`centers`) chosen.
#'
#' @param pred_name Character; name of the model prediction to load (without extension).
#' @param centers Integer; number of clusters for K-means (e.g., 4 or 5).
#' @param nstart Integer; number of random starts for K-means (default = 25).
#' @param alpha Numeric; weighting factor for hazard contribution in risk index.
#' @param beta Numeric; weighting factor for class contribution in risk index.
#' @param WDir Character; working directory containing the model and datasets.
#' @param original_csv Character; filename of the original dataset with coordinates.
#' @param saveresults Character or NULL; optional name for saving output as `.model`.
#'
#' @return A list with processed results including:
#' \item{com12D}{Detailed risk dataset with PCA, cluster, and risk metrics.}
#' \item{area_risk}{Aggregated area-level risk.}
#' \item{com12D_sf}{Spatial `sf` version of `com12D` (if original_csv provided).}
#' \item{area_risk_sf}{Spatial `sf` version of `area_risk`.}
#' \item{summary_pca}{PCA summary.}
#' \item{rotation_pca}{PCA loadings.}
#' @author
#' Dominic Owusu-Ansah

risk_result <- risk_processing(
  pred_name = "risk.pred",
  centers = 5,
  alpha = 0.7,
  beta = 0.3,
  WDir = "C:/Users/domin/Documents/slope/risk",
  original_csv = "A13_vertical_2016_2023_clipped.csv",
  saveresults = "A13_risk_results"
)


com12D <- risk_result$com12D
area_risk <- risk_result$area_risk
com12D_sf <- risk_result$com12D_sf
area_risk_sf <- risk_result$area_risk_sf


risk_result$summary_pca
risk_result$rotation_pca
# 
# # Save spatial datasets to GeoPackage
# st_write(com12D_sf, "com12D_risk.gpkg", delete_dsn = TRUE)
# st_write(area_risk_sf, "area_risk.gpkg", delete_dsn = TRUE)
# 
# cat("✅ GeoPackage files saved successfully:\n - com12D_risk.gpkg\n - area_risk.gpkg\n")
# 


table(com12D$slope_class)
      