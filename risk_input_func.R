

### Define the load_data function ----
load_data <- function(input_file) {
  
  # Load data
  # Check whether input is a file path (character) or a dataframe
  if (is.data.frame(input_file)) {
    cat("Loading data from provided dataframe\n")
    data <- input_file
  } else if (is.character(input_file)) {
    cat("Loading data from file: ", input_file, "\n")
    data <- read.csv(file = input_file, header = TRUE)
    cat("Data loaded successfully\n")
  } else {
    stop("Error: Input must be either a file path (character) or a dataframe.")
  }
  
  
  tryCatch({
    # Convert and edit data
    actual_workdata <- data.frame(t(data))
    colnames(actual_workdata) <- paste0("PID", seq_len(ncol(actual_workdata)))
    
    
    # Convert row names to Date format
    actual_workdata$Date <- sub("X", "", row.names(actual_workdata))
    
    # Remove the rownames
    rownames(actual_workdata) <- NULL
    
    # Move the 'Date' column to the first position using select()
    actual_workdata <- actual_workdata %>%
      select(Date, everything())
    
    
    actual_workdata <- na.omit(actual_workdata)
    
    
    # Convert the 'Date' column to the Date format again for consistency
    actual_workdata$Date <- as.Date(actual_workdata$Date, format = "%Y%m%d")
    
    # Calculate the differences between consecutive dates
    time_diff <- diff(actual_workdata$Date)
    
    # Find unique time differences
    unique_diff <- unique(time_diff)
    
    if (length(unique_diff) > 1) {
      cat("Irregular time intervals found:\n")
      print(unique_diff)
    } else {
      cat("All time intervals are consistent.\n")
    }
    
    # Define a function to handle missing dates for specific gap sizes
    fill_missing_dates <- function(time_diff, actual_workdata, gap_size, interval = 12) {
      # Identify the positions where the gap matches the specified size
      indices <- which(time_diff == gap_size) + 1  # Add 1 to get the start position of the gap
      
      # Extract the dates before and after the large gaps
      missing_dates_before <- actual_workdata$Date[indices - 1]
      missing_dates_after <- actual_workdata$Date[indices]
      
      # Generate the missing dates that should be between these gaps
      missing_dates <- sapply(seq_along(missing_dates_before), function(i) {
        seq(missing_dates_before[i] + interval, missing_dates_after[i] - interval, by = paste0(interval, " days"))
      })
      
      # Flatten the list of missing dates and remove duplicates
      unique(unlist(missing_dates))
    }
    
    # Handle missing dates for specific gaps (24 days, 72 days, etc.)
    missing_dates_24 <- fill_missing_dates(time_diff, actual_workdata, gap_size = 24, interval = 12)
    missing_dates_36 <- fill_missing_dates(time_diff, actual_workdata, gap_size = 36, interval = 12)
    missing_dates_72 <- fill_missing_dates(time_diff, actual_workdata, gap_size = 72, interval = 12)
    
    # Combine all missing dates
    missing_dates <- unique(c(missing_dates_24, missing_dates_36, missing_dates_72))
    missing_dates <- as.Date(missing_dates)
    
    # Combine original dates with missing dates
    all_dates <- sort(unique(c(actual_workdata$Date, missing_dates)))
    
    # Create a new dataframe with all dates
    expanded_data <- data.frame(Date = all_dates)
    
    # Merge the expanded data with the actual work data
    merged_data <- merge(expanded_data, actual_workdata, by = "Date", all.x = TRUE)
    
    # Perform spline interpolation for missing values
    interpolated_data <- merged_data %>%
      arrange(Date) %>%
      mutate(across(starts_with("PID"), ~ zoo::na.approx(., na.rm = FALSE)))
    
    # Update actual_workdata with the interpolated values
    actual_workdata <- interpolated_data
    
  }, error = function(e) {
    cat("Error in data conversion or formatting:\n")
    cat(e$message, "\n")
  })
  
  return(actual_workdata)
}


### Dynamically split data based on the number of intervals provided
extract_data <- function(data, num_splits) {
  cat("Extracting data subsets...\n")
  
  # Split the data into `num_splits` subsets as equally as possible
  split_data <- split(data, ceiling(seq_len(nrow(data)) / (nrow(data) / num_splits)))
  
  # Name the subsets
  names(split_data) <- paste0("W", seq_along(split_data))
  
  return(split_data)
}


# Define optimized sliding window function
sliding_window <- function(values, dates, point_id, window_size = 5) {
  # Ensure dates are in vector format
  dates <- as.vector(as.numeric(as.Date(dates)))
  
  # Create windows for measurements and dates, and reverse order for correct sequence
  windows <- t(apply(embed(values, window_size), 1, rev))
  date_windows <- t(apply(embed(dates, window_size), 1, rev))
  
  # Calculate time differences in days between consecutive dates in each window
  time_diffs <- t(apply(date_windows, 1, function(d) diff(d)))
  
  # Combine into data frame
  result <- data.frame(
    PointID = point_id,
    windows,
    date_windows,
    time_diffs
  )
  
  return(result)
}

### Function to process data ----
process_data <- function(Wx, interval, w) {
  cat("Processing ", interval, " data...\n")
  
  # Generate a sequence of dates at the specified interval
  new_dates <- seq.Date(from = min(Wx$Date), 
                        to = max(Wx$Date), 
                        by = paste(interval, "days"))
  
  # Subset the original dataset based on the new dates
  dataset <- Wx[Wx$Date %in% new_dates, ]
  
  if (nrow(dataset) == 0) {
    stop("No data for the given date range and interval.")
  }
  
  # Separate the Date column from the rest (measurements)
  dates <- dataset[["Date"]]  # Make sure to use the "Date" column
  measurements <- dataset[,-1]  # All other columns are the measurements
  
  # Apply to all measurements with vectorized time difference calculation
  result_list <- lapply(1:ncol(measurements), function(col_idx) {
    sliding_window(values=as.numeric(measurements[[col_idx]]), dates=dates, point_id=colnames(measurements)[col_idx], window_size = w)
  })
  
  # Combine all results into a single data frame
  new_dataset <- do.call(rbind, result_list)
  
  # Rename columns
  colnames(new_dataset) <- c("PointID", paste0("stage_", 1:w), paste0("D", 1:w), paste0("Time", 1:(w-1)))
  
  # Convert date columns (D1 to D5) back to "%Y%m%d" format
  new_dataset[, paste0("D", 1:w)] <- lapply(new_dataset[, paste0("D", 1:w)], function(x) format(as.Date(x, origin = "1970-01-01"), "%Y%m%d"))
  
  
  # Reorder columns
  column_order <- c("PointID", paste0("Time", 1:(w-1)), paste0("stage_", 1:w), paste0("D", 1:w))
  new_dataset <- new_dataset[, column_order]
  
  return(list(data = new_dataset, dates = new_dates))
}


### Function to process data in reverse ----
process_data_rev <- function(Wx, interval, w) {
  cat("Processing ", interval, " data...\n")
  
  # Generate a sequence of dates at the specified interval
  new_dates <- seq.Date(from = max(Wx$Date), 
                        to = min(Wx$Date), 
                        by = paste0("-", interval, " days"))
  
  # Subset the original dataset based on the new dates
  dataset <- Wx[Wx$Date %in% new_dates, ]
  
  if (nrow(dataset) == 0) {
    stop("No data for the given date range and interval.")
  }
  
  # Separate the Date column from the rest (measurements)
  dates <- dataset[["Date"]]  # Make sure to use the "Date" column
  measurements <- dataset[,-1]  # All other columns are the measurements
  
  # Apply to all measurements with vectorized time difference calculation
  result_list <- lapply(1:ncol(measurements), function(col_idx) {
    sliding_window(values=as.numeric(measurements[[col_idx]]), dates=dates, point_id=colnames(measurements)[col_idx], window_size = w)
  })
  
  # Combine all results into a single data frame
  new_dataset <- do.call(rbind, result_list)
  
  # Rename columns
  colnames(new_dataset) <- c("PointID", paste0("stage_", 1:w), paste0("D", 1:w), paste0("Time", 1:(w-1)))
  
  # Convert date columns (D1 to D5) back to "%Y%m%d" format
  new_dataset[, paste0("D", 1:w)] <- lapply(new_dataset[, paste0("D", 1:w)], function(x) format(as.Date(x, origin = "1970-01-01"), "%Y%m%d"))
  
  
  # Reorder columns
  column_order <- c("PointID", paste0("Time", 1:(w-1)), paste0("stage_", 1:w), paste0("D", 1:w))
  new_dataset <- new_dataset[, column_order]
  
  return(list(data = new_dataset, dates = new_dates))
}


### Function for processing last four dates ----
process_data_v1 <- function(Wx, interval) {
  cat("Processing data with interval of", interval, "days...\n")
  
  # Generate a sequence of dates at the specified interval from max to min date
  new_dates <- seq.Date(from = max(Wx$Date), 
                        to = min(Wx$Date), 
                        by = paste0("-", interval, " days"))
  
  # Select the first four dates using the head function
  first_dates <- head(new_dates, 4)
  
  # Subset the original dataset based on the selected first four dates
  dataset <- Wx[Wx$Date %in% first_dates, ]
  
  if (nrow(dataset) < 4) {
    stop("Not enough data points for the given date range and interval.")
  }
  
  # Data without the Date column
  dataset_without_date <- dataset %>% select(-Date)
  
  # Calculate the time difference between consecutive dates (absolute values)
  time_diff <- abs(as.numeric(diff(first_dates)))
  time_intervals <- rep(time_diff, each = ncol(dataset_without_date))
  
  # Transpose and format data into stages
  transposed_data <- as.data.frame(t(dataset_without_date))
  colnames(transposed_data) <- paste0("stage_", 1:4)
  
  # Create the FID column (row numbers)
  transposed_data$FID <- 1:nrow(transposed_data)
  
  # Create a time data table with absolute time intervals
  time_data <- as.data.frame(matrix(time_intervals, nrow = nrow(transposed_data), byrow = TRUE))
  colnames(time_data) <- paste0("Time", 1:3)
  
  # Combine the transposed data with time intervals
  combined_data <- cbind(transposed_data, time_data)
  
  # Reorder columns to have FID first, then stages, then time intervals
  combined_data <- combined_data[, c("FID", paste0("Time", 1:3), paste0("stage_", 1:4))]
  
  return(list(data = combined_data, dates = first_dates))
}


### Function to combine datasets into balanced and unbalanced datasets ----
com_datasets <- function(processed_data_list) {
  
  # Extract the 'data' component from each processed dataset
  datasets <- lapply(processed_data_list, function(dataset) dataset$data)
  
  # Combine the datasets (unbalanced dataset)
  unbalanced_data <- do.call(rbind, datasets)
  
  ######################################
  # Combine dataset for balanced
  
  # Determine the length of the shortest dataset
  min_length <- min(sapply(datasets, nrow))
  
  # Randomly sample 'min_length' rows from each dataset
  set.seed(123)  # Setting seed for reproducibility
  balanced_data <- do.call(rbind, lapply(datasets, function(data) {
    data[sample(nrow(data), min_length), ]
  }))
  
  # Return both unbalanced and balanced datasets
  return(list(unbalanced_data = unbalanced_data, balanced_data = balanced_data))
}


### Function to implement mining procedure ----
train_models <- function(Wdata, mlds) {
  
  # Run models
  MM <- lapply(mlds, function(m) {
    cat('\nRunning mining :: ', m, '\n')
    mining(x = stage_5 ~ .,
           data = Wdata,
           Runs = 5,
           method = c("kfold", 5),
           model = m,
           task = "reg",
           feature = "sens")
  })
  
  names(MM) <- mlds
  
  ## Metrics for each model
  metrics <- lapply(MM, function(x) {
    mmetric(x, metric = c('MAE', 'MSE', 'RMSE', 'R2'), aggregate = "mean")
  })
  
  ## Combine metrics into a data frame
  MET <- do.call(cbind, metrics)
  
  # Generate MM_Data for each model
  MM_Data <- lapply(MM, function(PP) {
    # Average predictions across runs
    MM_pred <- Reduce("+", PP$pred) / length(PP$pred)
    MM_pred <- data.frame(MM_pred)
    
    # Get the test observations from the first run
    MM_test <- PP$test[[1]]
    MM_test <- data.frame(MM_test)
    
    # Combine observed, predicted, and error values
    MM_set <- data.frame(Obs = MM_test,
                         Pred = MM_pred,
                         Error = abs(MM_pred - MM_test))
    
    colnames(MM_set) <- c("Obs", "Pred", "Error")
    
    return(MM_set)  
  })
  
  # Return the models, metrics, and MM_Data
  return(list(models = MM, MET = MET, metrics = metrics, MMData = MM_Data))
}


### Function to filter data by the specified time intervals
filter_by_time_intervals <- function(data, time_interval) {
  data %>% filter(Time1 == time_interval, Time2 == time_interval, 
                  Time3 == time_interval, Time4 == time_interval)
}

### Function to calculate standard deviation of errors
calculate_statistics <- function(data) {
  std_error <- sd(data$Error)
  return(std_error)
}

### Function to implement fit procedure ----
fit_models <- function(Wdata, mlds) {
  # Run models
  FM <- lapply(mlds, function(m) {
    cat('\nRunning fit :: ', m, '\n')
    fit(x = stage_5 ~ .,
        data = Wdata,
        Runs = 5,
        model = m,
        task = "reg")
  })
  
  names(FM) <- mlds
  
  # Return results
  return(fitModels = FM)
}

### Function to rename and clean columns
rename_and_clean_columns <- function(data, model) {
  if (model == "mr") {
    # Select only the columns for 'mr' model and other necessary columns
    data <- data %>% select(Time1, Time2, Time3, Time4, stage_1, stage_2, stage_3, stage_4, stage_5, starts_with("mr."))
    # Rename columns to generic 'Obs', 'Pred', 'Error'
    colnames(data)[colnames(data) == "mr.Obs"] <- "Obs"
    colnames(data)[colnames(data) == "mr.Pred"] <- "Pred"
    colnames(data)[colnames(data) == "mr.Error"] <- "Error"
  } else if (model == "mlpe") {
    # Select only the columns for 'mlpe' model and other necessary columns
    data <- data %>% select(Time1, Time2, Time3, Time4, stage_1, stage_2, stage_3, stage_4, stage_5, starts_with("mlpe."))
    # Rename columns to generic 'Obs', 'Pred', 'Error'
    colnames(data)[colnames(data) == "mlpe.Obs"] <- "Obs"
    colnames(data)[colnames(data) == "mlpe.Pred"] <- "Pred"
    colnames(data)[colnames(data) == "mlpe.Error"] <- "Error"
  }
  return(data)
}



### Function for predicting ----
perform_prediction <- function(model_name, new_dataset_name, intervals = c(12, 24, 72, 144), pred_times, WDir = getwd(),saveresults = NULL) {
  require(rminer)
  require(dplyr)
  require(tidyverse)
  require(rsample)
  require(zoo)
  require(slider)
  
  # Set working directory
  setwd(WDir)
  
  # Start timing for loading the model
  start_time <- Sys.time()
  cat("Loading the model...\n")
  
  # Load the model
  predictor <- loadmodel(model_name)
  w <- predictor$window_size
  models <- predictor$fit_results
  interval_stats = predictor$interval_stats
  filtered_datasets = predictor$filtered_datasets
  
  end_time <- Sys.time()
  cat("Time to load the model: ", end_time - start_time, " seconds\n\n")
  
  
  
  # Start timing for loading the new dataset
  start_time <- Sys.time()
  cat("Loading new dataset:\n")
  
  source_dataset <- load_data(new_dataset_name)
  
  end_time <- Sys.time()
  cat("Time to load new datasets: ", end_time - start_time, " seconds\n\n")
  
  
  # Load raw data for fallback method: check if new_dataset_name is a file path or already a dataframe
  if (is.data.frame(new_dataset_name)) {
    cat("Using provided dataframe as raw data.\n")
    raw_data <- new_dataset_name
  } else if (is.character(new_dataset_name)) {
    cat("Loading raw data from new_dataset.csv:\n")
    raw_data <- read.csv(new_dataset_name)
  } else {
    stop("Error: new_dataset_name must be either a file path (character) or a dataframe.")
  }
  
  raw_data_dates <- sub("X", "",colnames(raw_data))
  
  # Extract raw data dates and PID1 values
  raw_data_dates <- as.Date(raw_data_dates, format="%Y%m%d")
  raw_data_pid1 <- as.numeric(raw_data[1, ])
  
  
  all_predictions <- lapply(pred_times,function(t){
    
    cat("Processing for prediction time:", t, "\n")
    
    # Start timing for processing the dataset
    start_time <- Sys.time()
    
    
    if (all(c("Date", "PID1") %in% colnames(source_dataset)) && ncol(source_dataset) == 2) {
      cat("Applying fallback method for special dataset...\n")
      
      # Filter the source dataset to match the raw data dates
      source_dataset <- source_dataset %>%
        dplyr::filter(Date %in% raw_data_dates)
      
      # Ensure the source dataset contains only the raw data (Date and PID1)
      source_dataset <- source_dataset %>%
        dplyr::select(Date, PID1)
      
      # Calculate time differences from the Date column
      time_diffs <- diff(as.numeric(as.Date(source_dataset$Date)))
      
      # Create the required columns with a default PointID
      fallback_data <- data.frame(
        PointID = 1,  # Use the first value of PID1 as the PointID
        Time1 = time_diffs[1],
        Time2 = time_diffs[2],
        Time3 = time_diffs[3],
        Time4 = t,
        stage_1 = source_dataset$PID1[1],
        stage_2 = source_dataset$PID1[2],
        stage_3 = source_dataset$PID1[3],
        stage_4 = source_dataset$PID1[4]
      )
      new_data <- fallback_data
      
      PointID <- fallback_data$PointID
      
      new_data <- new_data %>% select(-PointID)
      print(new_data)
      dates_used <- NULL  # No dates available in fallback method
      
      # Create an empty tibble for date_data to avoid errors later
      date_data <- tibble()
      
    } else {
      # Proceed with the original processing logic
      DayAvailable <- as.numeric(difftime(max(source_dataset$Date), min(source_dataset$Date), units = "days"))
      
      
      # closest_day_index <- which.min(abs(DayAvailable - intervals))
      closest_day_index <- which.min(abs(DayAvailable - c(36, 72, 216, 432)))
      closest_interval_index <- which.min(abs(intervals - t))
      final_index <- min(closest_interval_index, closest_day_index)
      selected_interval <- intervals[final_index]
      
      # Handle the issue when new_data provided is not enough.
      tryCatch({ 
        processed_data <- process_data(source_dataset, intervals[final_index], w-1)
        comXD <- processed_data$data
        dates_used <- processed_data$dates
        PointID <- comXD$PointID
        date_data <- comXD %>% select(starts_with("D"))
        # 
        # # Store the FID column
        # fid_column <- comXD$FID
        
        # Update the Time4 column and reorder the columns
        comXD$Time4 <- t
        comXD <- as.data.frame(comXD) %>% select(1:4, Time4, everything())
        new_data <- as.data.frame(comXD) %>%  select(-PointID, -starts_with("D"))
      }, error = function(e) {
        cat("Error in processing data for one of the intervals. Using previously processed data if available.\n")
        cat(e$message, "\n")
      })
    }
    
    if (is.null(new_data)) {
      stop("Failed to generate new_data for prediction time: ", t)
    }
    
    
    # End timing for processing the dataset
    end_time <- Sys.time()
    cat("Time to process dataset for prediction time ", t, ": ", end_time - start_time, " seconds\n\n")
    
    
    # Start timing for model predictions
    start_time <- Sys.time()
    
    cat("Performing prediction for time:", t, "\n")
    predictions <- lapply(names(models), function(m) {
      fm <- models[[m]]
      predict(object = fm, newdata = new_data)
      
      # # Perform the prediction
      # pred <- predict(object = fm, newdata = new_data)
      # 
      # # Add the FID column back to the predictions
      # pred <- cbind(FID = fid_column, pred)
      
    })
    # Combine predictions into a single data frame
    prediction_matrix <- do.call(cbind, predictions)
    colnames(prediction_matrix) <- paste0(names(models), "_T", t, "D")  # Add time to column names
    
    # Attach PointID, obs, date_data, and predictions to new_data
    new_data <- new_data %>% 
      mutate(PointID = PointID) %>% 
      bind_cols(date_data) %>% 
      bind_cols(as.data.frame(prediction_matrix))
    
    # Move PointID to the first column
    new_data <- new_data[, c("PointID", setdiff(names(new_data), "PointID"))]
    
    # End timing for model predictions
    end_time <- Sys.time()
    cat("Time for model predictions: ", end_time - start_time, " seconds\n\n")
    
    names(predictions) <- names(models)
    
    return(list(Predictions = predictions, DatesUsed = dates_used, NewData = new_data))
    
  })
  names(all_predictions) <- paste0("T", pred_times, 'D')
  
  
  # Filtering process for all unique PointIDs
  filtered_results <- list()
  for (t in names(all_predictions)) {
    xx <- all_predictions[[t]]$NewData
    
    filtered_results[[t]] <- xx %>%
      group_by(PointID) %>%
      slice_tail(n = 1) %>%
      ungroup() %>%
      arrange(as.numeric(gsub("\\D", "", PointID)))  # Sort numerically
  }
  
  result <- list(
    model_name = model_name,
    pred_times = pred_times,
    all_predictions = all_predictions,
    last_four_dates = filtered_results,
    window_size = w,
    model_results = predictor$model_results,
    fit_results = predictor$fit_results,
    interval_stats = interval_stats,
    filtered_datasets = filtered_datasets,
    source_dataset = source_dataset
  )
  
  
  # Start timing for saving results
  start_time <- Sys.time()
  
  # Save the results if saveresults is provided
  if (!is.null(saveresults)) {
    savemodel(result,  paste0(saveresults, ".pred"))
    cat("Results saved as:", file.path(WDir, paste0(saveresults, ".pred")), "\n")
  }
  
  # End timing for saving results
  end_time <- Sys.time()
  cat("Time to save results: ", end_time - start_time, " seconds\n\n")
  
  return(result)
}



### Function for predicting ----
perform_prediction_rev <- function(model_name, new_dataset_name, intervals = c(12, 24, 72, 144), pred_times, WDir = getwd(),saveresults = NULL) {
  require(rminer)
  require(dplyr)
  require(tidyverse)
  require(rsample)
  require(zoo)
  require(slider)
  
  # Set working directory
  setwd(WDir)
  
  # Start timing for loading the model
  start_time <- Sys.time()
  cat("Loading the model...\n")
  
  # Load the model
  predictor <- loadmodel(model_name)
  w <- predictor$window_size
  models <- predictor$fit_results
  interval_stats = predictor$interval_stats
  filtered_datasets = predictor$filtered_datasets
  
  end_time <- Sys.time()
  cat("Time to load the model: ", end_time - start_time, " seconds\n\n")
  
  
  
  # Start timing for loading the new dataset
  start_time <- Sys.time()
  cat("Loading new dataset:\n")
  
  source_dataset <- load_data(new_dataset_name)
  
  end_time <- Sys.time()
  cat("Time to load new datasets: ", end_time - start_time, " seconds\n\n")
  
  
  
  # Load raw data for fallback method: check if new_dataset_name is a file path or already a dataframe
  if (is.data.frame(new_dataset_name)) {
    cat("Using provided dataframe as raw data.\n")
    raw_data <- new_dataset_name
  } else if (is.character(new_dataset_name)) {
    cat("Loading raw data from new_dataset.csv:\n")
    raw_data <- read.csv(new_dataset_name)
  } else {
    stop("Error: new_dataset_name must be either a file path (character) or a dataframe.")
  }
  
  
  raw_data_dates <- sub("X", "",colnames(raw_data))
  
  
  # Extract raw data dates and PID1 values
  raw_data_dates <- as.Date(raw_data_dates, format="%Y%m%d")
  raw_data_pid1 <- as.numeric(raw_data[1, ])
  
  
  all_predictions <- lapply(pred_times,function(t){
    
    cat("Processing for prediction time:", t, "\n")
    
    # Start timing for processing the dataset
    start_time <- Sys.time()
    
    
    if (all(c("Date", "PID1") %in% colnames(source_dataset)) && ncol(source_dataset) == 2) {
      cat("Applying fallback method for special dataset...\n")
      
      # Filter the source dataset to match the raw data dates
      source_dataset <- source_dataset %>%
        dplyr::filter(Date %in% raw_data_dates)
      
      # Ensure the source dataset contains only the raw data (Date and PID1)
      source_dataset <- source_dataset %>%
        dplyr::select(Date, PID1)
      
      # Calculate time differences from the Date column
      time_diffs <- diff(as.numeric(as.Date(source_dataset$Date)))
      
      # Create the required columns with a default PointID
      fallback_data <- data.frame(
        PointID = source_dataset$PID1[1],  # Use the first value of PID1 as the PointID
        Time1 = time_diffs[1],
        Time2 = time_diffs[2],
        Time3 = time_diffs[3],
        Time4 = t,
        stage_1 = source_dataset$PID1[1],
        stage_2 = source_dataset$PID1[2],
        stage_3 = source_dataset$PID1[3],
        stage_4 = source_dataset$PID1[4]
      )
      new_data <- fallback_data
      
      PointID <- fallback_data$PointID
      
      new_data <- new_data %>% select(-PointID)
      print(new_data)
      dates_used <- NULL  # No dates available in fallback method
      
      # Create an empty tibble for date_data to avoid errors later
      date_data <- tibble()
      
    } else {
      # Proceed with the original processing logic
      DayAvailable <- as.numeric(difftime(max(source_dataset$Date), min(source_dataset$Date), units = "days"))
      
      
      # closest_day_index <- which.min(abs(DayAvailable - intervals))
      closest_day_index <- which.min(abs(DayAvailable - c(36, 72, 216, 432)))
      closest_interval_index <- which.min(abs(intervals - t))
      final_index <- min(closest_interval_index, closest_day_index)
      selected_interval <- intervals[final_index]
      
      # Handle the issue when new_data provided is not enough.
      tryCatch({ 
        processed_data <- process_data_rev(source_dataset, intervals[final_index], w-1)
        comXD <- processed_data$data
        dates_used <- processed_data$dates
        PointID <- comXD$PointID
        date_data <- comXD %>% select(starts_with("D"))
        # 
        # # Store the FID column
        # fid_column <- comXD$FID
        
        # Update the Time4 column and reorder the columns
        comXD$Time4 <- t
        comXD <- as.data.frame(comXD) %>% select(1:4, Time4, everything())
        new_data <- as.data.frame(comXD) %>%  select(-PointID, -starts_with("D"))
      }, error = function(e) {
        cat("Error in processing data for one of the intervals. Using previously processed data if available.\n")
        cat(e$message, "\n")
      })
    }
    
    if (is.null(new_data)) {
      stop("Failed to generate new_data for prediction time: ", t)
    }
    
    
    # End timing for processing the dataset
    end_time <- Sys.time()
    cat("Time to process dataset for prediction time ", t, ": ", end_time - start_time, " seconds\n\n")
    
    
    # Start timing for model predictions
    start_time <- Sys.time()
    
    cat("Performing prediction for time:", t, "\n")
    predictions <- lapply(names(models), function(m) {
      fm <- models[[m]]
      predict(object = fm, newdata = new_data)
      
      # # Perform the prediction
      # pred <- predict(object = fm, newdata = new_data)
      # 
      # # Add the FID column back to the predictions
      # pred <- cbind(FID = fid_column, pred)
      
    })
    # Combine predictions into a single data frame
    prediction_matrix <- do.call(cbind, predictions)
    colnames(prediction_matrix) <- paste0(names(models), "_T", t, "D")  # Add time to column names
    
    # Attach PointID, obs, date_data, and predictions to new_data
    new_data <- new_data %>% 
      mutate(PointID = PointID) %>% 
      bind_cols(date_data) %>% 
      bind_cols(as.data.frame(prediction_matrix))
    
    # Move PointID to the first column
    new_data <- new_data[, c("PointID", setdiff(names(new_data), "PointID"))]
    
    # End timing for model predictions
    end_time <- Sys.time()
    cat("Time for model predictions: ", end_time - start_time, " seconds\n\n")
    
    names(predictions) <- names(models)
    
    return(list(Predictions = predictions, DatesUsed = dates_used, NewData = new_data))
    
  })
  names(all_predictions) <- paste0("T", pred_times, 'D')
  
  
  # Filtering process for all unique PointIDs
  filtered_results <- list()
  for (t in names(all_predictions)) {
    xx <- all_predictions[[t]]$NewData
    
    filtered_results[[t]] <- xx %>%
      group_by(PointID) %>%
      slice_tail(n = 1) %>%
      ungroup() %>%
      arrange(as.numeric(gsub("\\D", "", PointID)))  # Sort numerically
  }
  
  result <- list(
    model_name = model_name,
    pred_times = pred_times,
    all_predictions = all_predictions,
    last_four_dates = filtered_results,
    window_size = w,
    model_results = predictor$model_results,
    fit_results = predictor$fit_results,
    interval_stats = interval_stats,
    filtered_datasets = filtered_datasets,
    source_dataset = source_dataset
  )
  
  
  # Start timing for saving results
  start_time <- Sys.time()
  
  # Save the results if saveresults is provided
  if (!is.null(saveresults)) {
    savemodel(result,  paste0(saveresults, ".pred"))
    cat("Results saved as:", file.path(WDir, paste0(saveresults, ".pred")), "\n")
  }
  
  # End timing for saving results
  end_time <- Sys.time()
  cat("Time to save results: ", end_time - start_time, " seconds\n\n")
  
  return(result)
}



risk_processing <- function(pred_name, centers = 5, nstart = 25, alpha = 0.7, beta = 0.3, WDir = getwd(), 
                            original_csv = "A13_vertical_2016_2023_clipped.csv", saveresults = NULL) {
  
  require(rminer)
  require(dplyr)
  require(tidyverse)
  require(rsample)
  require(zoo)
  require(slider)
  require(cluster)
  require(NbClust)
  require(scales)
  require(sf)
  require(mapview)
  require(htmlwidgets)
  
  # ───────────────────────────────────────────────────────────────
  # 0. Set working directory and load model
  # ───────────────────────────────────────────────────────────────
  cat("Setting directory...\n")
  setwd(WDir)
  
  cat("Loading the model...\n")
  predictor <- loadmodel(pred_name) 
  
  com12D <- predictor$all_predictions$T12D$NewData
  
  # ───────────────────────────────────────────────────────────────
  # 1. Compute stage differences
  # ───────────────────────────────────────────────────────────────
  com12D <- com12D %>%
    mutate(
      pred          = mlpe_T12D,
      stagediff12   = abs(stage_2 - stage_1),
      stagediff23   = abs(stage_3 - stage_2),
      stagediff34   = abs(stage_4 - stage_3),
      avg_stagediff = (stagediff12 + stagediff23 + stagediff34) / 3,
      sd_stagediff  = apply(cbind(stagediff12, stagediff23, stagediff34), 1, sd)
    ) %>% 
    select(-mr_T12D, -mlpe_T12D)
  
  # ───────────────────────────────────────────────────────────────
  # 2. Prepare PCA
  # ───────────────────────────────────────────────────────────────
  features <- com12D %>%
    select(avg_stagediff, sd_stagediff, stagediff34, stage_4)
  
  features_scaled <- scale(features)
  
  correlation_features <- cor(features)
  
  pca_result <- prcomp(features_scaled, center = TRUE, scale. = TRUE)
  summary_pca <- summary(pca_result)
  rotation_pca <- pca_result$rotation
  pca_scores <- pca_result$x[, 1:2]
  
  # ───────────────────────────────────────────────────────────────
  # 3. K-means clustering on PCA
  # ───────────────────────────────────────────────────────────────
  set.seed(123)
  pca_kmeans_result <- kmeans(pca_scores, centers = centers, nstart = nstart)
  
  pca_df <- as.data.frame(pca_scores)
  colnames(pca_df) <- c("PC1", "PC2")
  
  com12D$pca_cluster <- pca_kmeans_result$cluster
  
  
  # ───────────────────────────────────────────────
  # 4. Dynamic slope class assignment (A–D, A–E, etc.)
  # ───────────────────────────────────────────────
  cluster_order <- com12D %>%
    group_by(pca_cluster) %>%
    summarise(mean_sd = mean(sd_stagediff, na.rm = TRUE)) %>%
    arrange(mean_sd)
  
  slope_labels <- LETTERS[1:centers]
  
  mapping <- cluster_order %>%
    mutate(slope_class = slope_labels[1:centers]) %>%
    select(pca_cluster, slope_class, mean_sd)
  
  com12D <- com12D %>%
    right_join(mapping, by = "pca_cluster")
  
  # ───────────────────────────────────────────────
  # 5. Compute Risk Index dynamically
  # ───────────────────────────────────────────────
  score_map <- setNames(seq_len(centers), slope_labels[1:centers])
  
  com12D <- com12D %>%
    mutate(
      ClassScore = score_map[slope_class],
      Hazard    = abs(stage_4),
      Hazard_n  = rescale(Hazard, to = c(0,1)),
      Class_n   = rescale(ClassScore, to = c(0,1)),
      RiskRaw   = alpha * Hazard_n + beta * Class_n,
      RiskIndex = rescale(RiskRaw, to = c(0,1))
    )
  
  # ───────────────────────────────────────────────
  # 6. Risk category (4 levels for map simplicity)
  # ───────────────────────────────────────────────
  breaks_q <- quantile(com12D$RiskIndex, probs = seq(0, 1, length.out = 5), na.rm = TRUE)
  
  com12D <- com12D %>%
    mutate(
      RiskCategory = cut(
        RiskIndex,
        breaks = unique(breaks_q),
        include.lowest = TRUE,
        labels = c("Low", "Moderate", "High", "Very High")
      )
    )
  
  # ───────────────────────────────────────────────
  # 7. Aggregate risk by PointID
  # ───────────────────────────────────────────────
  area_risk <- com12D %>%
    group_by(PointID) %>%
    summarise(MaxRiskIndex = max(RiskIndex, na.rm = TRUE))
  
  breaks_q_max <- quantile(area_risk$MaxRiskIndex, probs = seq(0,1,length.out=5), na.rm = TRUE)
  
  area_risk <- area_risk %>%
    mutate(
      MaxRiskCategory = cut(
        MaxRiskIndex,
        breaks = unique(breaks_q_max),
        include.lowest = TRUE,
        labels = c("Low", "Moderate", "High", "Very High")
      )
    )
  
  # ───────────────────────────────────────────────────────────────
  # 8. Spatial join with original coordinates (no save)
  # ───────────────────────────────────────────────────────────────
  if (file.exists(original_csv)) {
    cat("Loading original coordinate data...\n")
    
    original <- read.csv(original_csv)
    
    pid_coords <- original %>%
      mutate(PointID = paste0("PID", fid)) %>%
      select(PointID, longitude, latitude, acceleration, mean_velocity, max_temporal_coherence, slope_angle)
    
    pid_coords_sf <- pid_coords %>%
      st_as_sf(coords = c("longitude", "latitude"), crs = 4326)
    
    # Optional: convert to EPSG 3763
    # pid_coords_sf <- st_transform(pid_coords_sf, 3763)
    
    com12D_sf <- com12D %>%
      left_join(pid_coords_sf, by = "PointID")
    
    area_risk_sf <- area_risk %>%
      left_join(pid_coords_sf, by = "PointID")
    
  } else {
    warning("Original CSV not found — skipping spatial join.")
    com12D_sf <- NULL
    area_risk_sf <- NULL
  }
  
  # ───────────────────────────────────────────────────────────────
  # 9. Save and return results
  # ───────────────────────────────────────────────────────────────
  result <- list(
    pred_name = pred_name,
    com12D = com12D,
    features = features,
    features_scaled = features_scaled,
    correlation_features = correlation_features,
    summary_pca = summary_pca,
    rotation_pca = rotation_pca,
    pca_df = pca_df,
    pca_kmeans_result = pca_kmeans_result,
    area_risk = area_risk,
    com12D_sf = com12D_sf,
    area_risk_sf = area_risk_sf
  )
  
  if (!is.null(saveresults)) {
    savemodel(result, paste0(saveresults, ".model"))
    cat("Results saved as:", file.path(WDir, paste0(saveresults, ".model")), "\n")
  }
  
  return(result)
}
