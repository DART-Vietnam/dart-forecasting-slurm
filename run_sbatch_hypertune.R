# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
library(glue)

parallel_workers <- 30

root_dir <- "/home/tuyenhn/dart-forecasting-slurm/input_data"

LEARNERS_PATH <- glue("{root_dir}/spatiotempcv_13_lrners.qs2")
TRAIN_RSMP_PATH <- glue("{root_dir}/spatiotempcv_13_train_rsmp_list.qs2")
TASKS_PATH <- glue("{root_dir}/spatiotempcv_13_train_tsk_list.qs2")

OUTPUT_DIR <- glue("{root_dir}/../slurm_hypertune_output")

fs::dir_create(OUTPUT_DIR)

# ==============================================================================
# 2. WORKER FUNCTION
# ==============================================================================

# This function equals one job submitted to Slurm
worker_hpo <- function(task_idx) {
  # library imports and parallel configs
  library(mlr3)
  library(mlr3learners)
  library(mlr3forecast)
  library(mlr3mbo)
  library(qs2)
  library(stringr)
  library(magrittr)
  library(forcats)
  library(withr)

  data.table::setDTthreads(1)
  RhpcBLASctl::blas_set_num_threads(1)

  set.seed(764)

  # logging setup
  log_fpath <- file.path(OUTPUT_DIR, sprintf("hpo_fh%d.log", task_idx))
  con <- local_connection(file(log_fpath, open = "wt"))

  local_output_sink(con)
  local_message_sink(con)

  # set tune measures
  msr_bias <- msr("regr.bias")
  msr_bias$minimize <- TRUE
  tune_msrs <- list(msr("regr.rmse"), msr("regr.rqr"), msr_bias)

  # set tune terminators
  trm_run_time <- trm("run_time", secs = 60 * 60)
  trm_stag_batch <- trm("stagnation_hypervolume", threshold = 0.1)

  trm_combos <- trm(
    "combo",
    terminators = list(
      trm_run_time,
      trm_stag_batch
    )
  )

  # load data
  print(glue("Start hypertuning for horizon {task_idx}...", ))

  print(glue("Loading data (resampling sets, task, and learner)", ))
  all_rsmps <- qs_read(TRAIN_RSMP_PATH)
  all_tasks <- qs_read(TASKS_PATH)
  all_lrners <- qs_read(LEARNERS_PATH)

  .tsk <- all_tasks[[task_idx]]
  .lrn <- all_lrners[[task_idx]]
  .rsmp <- all_rsmps[[task_idx]]

  # make sure there's only 1 thread for internal tree parallelisation
  .lrnr_id <- .lrn$base_learner()$id %>%
    str_split_1("\\.") %>%
    `[[`(2)
  .lrnr_id <- if (.lrnr_id == "xgboost") "xgb" else .lrnr_id

  if (.lrnr_id == "ranger") {
    .lrn$param_set$values$regr.ranger.num.threads <- 1
  } else if (.lrnr_id == "xgb") {
    .lrn$param_set$values$regr.xgboost.nthread <- 1
  } else {
    NULL
  }

  print("Setting up TuningInstance and Tuner objects")
  # Setup Tuning Instance
  hpo_ti <- ti(
    task = .tsk,
    learner = .lrn,
    resampling = .rsmp,
    measure = tune_msrs,
    terminator = trm_combos,
    store_benchmark_result = FALSE # reduce ram usage by a lot if using graph learner
  )

  # Setup MBO tuner
  hpo_tnr <- tnr(
    "mbo",
    loop_function = bayesopt_emo, # Sobol initial design by default, size = 4*d (d: dimensions)
    # surrogate
    acq_function = acqf("ehvi")
    # acq_optimizer
  )

  # Run HPO
  print(glue(
    "Running HPO with future::plan(multicore) ({parallel_workers} workers)..."
  ))
  plan(multicore, workers = parallel_workers)
  hpo_tnr$optimize(hpo_ti)
  print("HPO completed successfully.")

  # 5. Save results
  result_path <- file.path(
    OUTPUT_DIR,
    sprintf("slurm_hpo_ti_fh%d.qs2", task_idx)
  )
  qs_save(hpo_ti, result_path)

  print(sprintf("Horizon %d finished. Saved to %s", task_idx, result_path))
}

# ==============================================================================
# 3. SUBMISSION
# ==============================================================================

# Libraries submission script
library(future)
library(future.batchtools)
library(qs2)


# Use the SLURM paths for the master submission as well
num_hori <- length(qs_read(TASKS_PATH))

# Set the plan
plan(
  batchtools_slurm,
  resources = list(
    time = "48:00:00",
    mem = "40G",
    asis = c(
      "--partition=big",
      glue("--cpus-per-task={parallel_workers}"),
      "--job-name=dart-hypertuning"
    ),
    nodes = 1,
    ntasks = 1,
    modules = c("R-base/4.5.3")
  )
)

# Submit the jobs
# invisible(lapply(seq_len(num_hori), function(i) future(worker_hpo(i))))
invisible(future(worker_hpo(1)))

print("All jobs submitted to SLURM")
