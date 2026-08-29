# ==============================================================================
# Expected Points (EP) Models for College Football using cfbfastR
# ==============================================================================
# Two separate, deliberately simple regression models — no multiclass /
# weighted-class-probability step:
#
#   Model A: predicts cfbfastR's own built-in ep_before (i.e., learns to
#            approximate cfbfastR's EP model from game-state features)
#   Model B: predicts next_score_points directly (the signed point value of
#            whatever scores next in the half — the actual ground-truth
#            target, same label nflscrapR/nflfastR-style EP models use,
#            just regressed on directly instead of via a 7-class softmax)
#
# Both are compared against (1) cfbfastR's ep_before and (2) the ground-truth
# next_score_points outcome, on a 2025 holdout untouched by training.
#
# Split: fit on 2014-2023, early-stop on 2024 (so test metrics aren't
# contaminated by using the test set for model selection), report final
# metrics only on 2025.
#
# Install once (run in a FRESH R session, before loading any libraries):
# install.packages(c("cfbfastR","dplyr","tidyr","stringr","xgboost","purrr","ggplot2"))
# Sys.setenv(CFBD_API_KEY = "your_key_here")  # get a free key at collegefootballdata.com
# ==============================================================================

library(cfbfastR)
library(dplyr)     # needs dplyr >= 1.1.0 for the glue-assignment syntax
                    # ("{name}" := value) used in add_epa() in Step 9 below
library(tidyr)
library(tibble)
library(stringr)
library(purrr)
library(xgboost)

# ------------------------------------------------------------------------------
# 1. PULL PLAY-BY-PLAY DATA
# ------------------------------------------------------------------------------
seasons <- 2014:2025
FIT_SEASONS  <- 2014:2023   # used to fit model weights
VAL_SEASON   <- 2024        # used only for early stopping / model selection
TEST_SEASON  <- 2025        # untouched holdout — all reported metrics come from here

pull_season <- function(yr) {
  path <- sprintf("pbp_%s.rds", yr)
  if (file.exists(path)) return(readRDS(path))
  pbp <- cfbfastR::load_cfb_pbp(seasons = yr)
  saveRDS(pbp, path)
  pbp
}

pbp_raw <- map_dfr(seasons, pull_season)

# ------------------------------------------------------------------------------
# 1b. SCHEMA CHECK
# ------------------------------------------------------------------------------
# Fail fast and clearly if cfbfastR's column names have drifted, rather than
# hitting a cryptic "object not found" several steps downstream.
required_cols <- c("game_id", "id_play", "half", "period", "down", "distance",
                    "yards_to_goal", "TimeSecsRem", "pos_team", "def_pos_team",
                    "offense_score", "defense_score", "offense_timeouts",
                    "defense_timeouts", "play_type", "season", "ep_before")
missing_cols <- setdiff(required_cols, names(pbp_raw))
if (length(missing_cols) > 0) {
  stop(
    "pbp_raw is missing expected columns: ", paste(missing_cols, collapse = ", "),
    "\nRun names(pbp_raw) to see what's actually available and update the ",
    "script to match (cfbfastR's schema has changed across versions)."
  )
}

# ------------------------------------------------------------------------------
# 2. BASIC FILTERING
# ------------------------------------------------------------------------------
pbp <- pbp_raw %>%
  filter(
    !is.na(down),
    down %in% 1:4,
    period <= 4,                     # drop OT — model separately if you want it
    !play_type %in% c(
      "Kickoff", "Timeout", "End Period", "End of Half",
      "End of Game", "Extra Point Good", "Extra Point Missed",
      "Two Point Rush", "Two Point Pass", "Penalty"
    ),
    !is.na(yards_to_goal),
    !is.na(TimeSecsRem)
  ) %>%
  arrange(game_id, id_play)

# ------------------------------------------------------------------------------
# 3. LABEL: next_score_points (SIGNED GROUND TRUTH TARGET FOR MODEL B)
# ------------------------------------------------------------------------------
# For every play, find the next scoring play within the same game+half and
# assign its point value, signed relative to whichever team had the ball on
# THIS play (positive = that team scores next, negative = the opponent does,
# 0 = no more scoring in the half). Touchdown play_type -> offense/defense
# credit based on actual counts observed in this pull:
#   pbp %>% filter(str_detect(play_type, "Touchdown")) %>% count(play_type)

TD_OFFENSE_CREDITED <- c(
  "Passing Touchdown", "Rushing Touchdown", "Penalty Touchdown",
  "Kickoff Team Fumble Recovery Touchdown", "Punt Team Fumble Recovery Touchdown"
)
TD_DEFENSE_CREDITED <- c(
  "Interception Return Touchdown", "Fumble Return Touchdown",
  "Fumble Recovery (Opponent) Touchdown", "Kickoff Return Touchdown",
  "Blocked Punt Touchdown", "Blocked Field Goal Touchdown",
  "Missed Field Goal Return Touchdown", "Punt Return Touchdown"
)
# "Uncategorized Touchdown" is dropped — too rare/ambiguous to classify safely.

touchdown_scores <- pbp %>%
  filter(play_type %in% c(TD_OFFENSE_CREDITED, TD_DEFENSE_CREDITED)) %>%
  transmute(
    game_id, half, score_id_play = id_play,
    scoring_team = if_else(play_type %in% TD_OFFENSE_CREDITED, pos_team, def_pos_team),
    points = 7
  )

fg_scores <- pbp %>%
  filter(play_type == "Field Goal Good") %>%
  transmute(game_id, half, score_id_play = id_play, scoring_team = pos_team, points = 3)

safety_scores <- pbp %>%
  filter(play_type == "Safety") %>%
  transmute(game_id, half, score_id_play = id_play, scoring_team = def_pos_team, points = 2)

scoring_plays <- bind_rows(touchdown_scores, fg_scores, safety_scores) %>%
  filter(!is.na(scoring_team))

next_score <- pbp %>%
  select(game_id, half, id_play, pos_team) %>%
  left_join(scoring_plays, by = c("game_id", "half")) %>%
  filter(score_id_play >= id_play) %>%
  group_by(game_id, half, id_play) %>%
  slice_min(score_id_play, n = 1) %>%
  ungroup()

pbp <- pbp %>%
  left_join(
    next_score %>% select(game_id, half, id_play, scoring_team, points),
    by = c("game_id", "half", "id_play")
  ) %>%
  mutate(
    next_score_points = case_when(
      is.na(points) ~ 0,
      scoring_team == pos_team ~ points,
      TRUE ~ -points
    )
  )

# ------------------------------------------------------------------------------
# 4. FEATURE ENGINEERING + FIT / VALIDATION / TEST SPLIT BY SEASON
# ------------------------------------------------------------------------------
required_step3_cols <- c("next_score_points")
missing_step3_cols <- setdiff(required_step3_cols, names(pbp))
if (length(missing_step3_cols) > 0) {
  stop(
    "pbp is missing columns created in Step 3: ", paste(missing_step3_cols, collapse = ", "),
    "\nRe-run the script from Step 1 through here in a single pass — this ",
    "usually means Step 3 didn't run against the current `pbp` object."
  )
}

model_data <- pbp %>%
  transmute(
    game_id, id_play,   # explicit join keys — do NOT rely on rownames()
    season,
    next_score_points,             # Model B's target (ground truth)
    cfb_ep_before   = ep_before,   # Model A's target, and comparison baseline
    yards_to_goal   = yards_to_goal,
    down            = factor(down),
    distance        = pmin(distance, 30),
    half_seconds    = pmin(TimeSecsRem, 1800),
    game_seconds    = TimeSecsRem + (period > 2) * 1800,
    score_diff      = pmin(pmax(offense_score - defense_score, -35), 35),
    off_timeouts    = offense_timeouts,
    def_timeouts    = defense_timeouts,
    under_2min      = as.integer(half_seconds <= 120),
    goal_to_go      = as.integer(distance >= yards_to_goal)
  ) %>%
  drop_na(next_score_points, cfb_ep_before, yards_to_goal, down, distance,
          half_seconds, score_diff, off_timeouts, def_timeouts)

fit_data  <- model_data %>% filter(season %in% FIT_SEASONS)
val_data  <- model_data %>% filter(season == VAL_SEASON)
test_data <- model_data %>% filter(season == TEST_SEASON)

cat(sprintf("Fit rows:   %s (seasons %s-%s)\n", nrow(fit_data), min(FIT_SEASONS), max(FIT_SEASONS)))
cat(sprintf("Val rows:   %s (season %s)\n", nrow(val_data), VAL_SEASON))
cat(sprintf("Test rows:  %s (season %s)\n", nrow(test_data), TEST_SEASON))

# ------------------------------------------------------------------------------
# 5. FIT THE TWO MODELS (regression, not classification)
# ------------------------------------------------------------------------------
FEATURE_FORMULA <- ~ yards_to_goal + down + distance + half_seconds +
                     score_diff + off_timeouts + def_timeouts +
                     under_2min + goal_to_go - 1

build_matrix <- function(df) model.matrix(FEATURE_FORMULA, data = df)

x_fit  <- build_matrix(fit_data)
x_val  <- build_matrix(val_data)
x_test <- build_matrix(test_data)

fit_one_model <- function(label_fit, label_val, model_name) {
  dfit <- xgb.DMatrix(data = x_fit, label = label_fit)
  dval <- xgb.DMatrix(data = x_val, label = label_val)
  cat(sprintf("\n--- Fitting %s ---\n", model_name))
  xgb.train(
    data = dfit,
    objective = "reg:squarederror",
    eval_metric = "rmse",
    max_depth = 5,
    eta = 0.05,
    nrounds = 500,
    subsample = 0.8,
    colsample_bytree = 0.8,
    watchlist = list(fit = dfit, val = dval),
    early_stopping_rounds = 30,
    print_every_n = 50,
    verbose = 1
  )
}

# Model A: predicts cfbfastR's own EP
model_ep_cfb <- fit_one_model(fit_data$cfb_ep_before, val_data$cfb_ep_before, "Model A (target = cfbfastR ep_before)")

# Model B: predicts the ground-truth next score directly
model_ep_nextscore <- fit_one_model(fit_data$next_score_points, val_data$next_score_points, "Model B (target = next_score_points)")

# ------------------------------------------------------------------------------
# 6. GENERATE PREDICTIONS
# ------------------------------------------------------------------------------
test_data$pred_ep_cfb_model        <- predict(model_ep_cfb, x_test)
test_data$pred_ep_nextscore_model  <- predict(model_ep_nextscore, x_test)

fit_data$pred_ep_cfb_model         <- predict(model_ep_cfb, x_fit)
fit_data$pred_ep_nextscore_model   <- predict(model_ep_nextscore, x_fit)
val_data$pred_ep_cfb_model         <- predict(model_ep_cfb, x_val)
val_data$pred_ep_nextscore_model   <- predict(model_ep_nextscore, x_val)

# ------------------------------------------------------------------------------
# 7. COMPARISON — ON THE 2025 HOLDOUT ONLY
# ------------------------------------------------------------------------------
# Two different comparisons matter here, for different reasons:
#
#  (a) vs. cfbfastR's ep_before: both are deterministic functions of game
#      state, so pointwise correlation/RMSE is meaningful and should be tight
#      for Model A (it's literally trying to reproduce ep_before) and looser
#      for Model B (it's targeting something else — actual outcomes — so
#      some divergence from cfbfastR's model is expected and not a bug).
#
#  (b) vs. next_score_points (ground truth): this target is a noisy REALIZED
#      outcome (mostly 0, occasionally +/-7, +/-3, +/-2), not an expectation.
#      No EP model, however good, will have low pointwise RMSE against raw
#      realizations — that's irreducible variance, not model error. The
#      metric that actually tells you if the model is well-calibrated is a
#      BINNED comparison: within each bucket of predicted EP, does the mean
#      *actual* next_score_points roughly match the mean *predicted* EP?
#      Both the pointwise numbers and the binned calibration table are
#      printed below — use the binned one to judge model quality.

comparison_metrics <- function(pred, actual, label) {
  tibble(
    comparison    = label,
    n             = length(pred),
    correlation   = cor(pred, actual, use = "complete.obs"),
    rmse          = sqrt(mean((pred - actual)^2)),
    mean_abs_diff = mean(abs(pred - actual)),
    mean_diff     = mean(pred - actual)
  )
}

pointwise_summary <- bind_rows(
  comparison_metrics(test_data$cfb_ep_before,          test_data$next_score_points, "cfbfastR EP vs. ground truth (baseline)"),
  comparison_metrics(test_data$pred_ep_cfb_model,       test_data$cfb_ep_before,     "Model A vs. cfbfastR EP"),
  comparison_metrics(test_data$pred_ep_cfb_model,       test_data$next_score_points, "Model A vs. ground truth"),
  comparison_metrics(test_data$pred_ep_nextscore_model, test_data$cfb_ep_before,     "Model B vs. cfbfastR EP"),
  comparison_metrics(test_data$pred_ep_nextscore_model, test_data$next_score_points, "Model B vs. ground truth")
)
cat("\n=== Pointwise comparison (2025 holdout) ===\n")
print(pointwise_summary)

# Binned calibration: the metric that actually matters for judging EP quality
calibration_table <- function(pred, actual, n_bins = 15) {
  tibble(pred = pred, actual = actual) %>%
    mutate(bin = ntile(pred, n_bins)) %>%
    group_by(bin) %>%
    summarise(mean_predicted = mean(pred), mean_actual_next_score = mean(actual), n = n()) %>%
    ungroup()
}

cat("\n=== Calibration vs. ground truth: cfbfastR EP (baseline) ===\n")
print(calibration_table(test_data$cfb_ep_before, test_data$next_score_points))

cat("\n=== Calibration vs. ground truth: Model A (target = cfbfastR EP) ===\n")
print(calibration_table(test_data$pred_ep_cfb_model, test_data$next_score_points))

cat("\n=== Calibration vs. ground truth: Model B (target = next_score_points) ===\n")
print(calibration_table(test_data$pred_ep_nextscore_model, test_data$next_score_points))

# ------------------------------------------------------------------------------
# 8. RENDERED PLOTS
# ------------------------------------------------------------------------------
library(ggplot2)

# Plot 1: calibration curves for all three predictors, overlaid
calibration_combined <- bind_rows(
  calibration_table(test_data$cfb_ep_before, test_data$next_score_points) %>%
    mutate(source = "cfbfastR EP (baseline)"),
  calibration_table(test_data$pred_ep_cfb_model, test_data$next_score_points) %>%
    mutate(source = "Model A (target = cfbfastR EP)"),
  calibration_table(test_data$pred_ep_nextscore_model, test_data$next_score_points) %>%
    mutate(source = "Model B (target = next score)")
)

calibration_plot <- calibration_combined %>%
  filter(n >= 20) %>%
  ggplot(aes(mean_predicted, mean_actual_next_score, color = source)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40", linewidth = 0.7) +
  geom_line(linewidth = 0.9, alpha = 0.7) +
  geom_point(aes(size = n), alpha = 0.85) +
  scale_size_continuous(range = c(1.5, 7), guide = "none") +
  labs(
    title = "EP calibration against ground truth (2025 holdout)",
    subtitle = "Within each bin of predicted EP, does the mean actual next-score outcome match?",
    x = "Mean predicted EP (binned)",
    y = "Mean actual next_score_points",
    color = NULL,
    caption = "Dashed line = perfect calibration."
  ) +
  theme_minimal(base_size = 12)

print(calibration_plot)
ggsave("ep_calibration_comparison.png", calibration_plot, width = 8.5, height = 6, dpi = 150)

# Plot 2: pointwise scatter, our two models vs. cfbfastR's EP
scatter_data <- bind_rows(
  test_data %>% transmute(cfb_ep_before, pred = pred_ep_cfb_model, model = "Model A (target = cfbfastR EP)"),
  test_data %>% transmute(cfb_ep_before, pred = pred_ep_nextscore_model, model = "Model B (target = next score)")
)

vs_cfb_plot <- scatter_data %>%
  ggplot(aes(cfb_ep_before, pred)) +
  geom_point(alpha = 0.03, size = 0.6, color = "#4C72B0") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#C44E52", linewidth = 0.8) +
  facet_wrap(~model) +
  coord_equal(xlim = c(-3, 7), ylim = c(-3, 7)) +
  labs(
    title = "Our models vs. cfbfastR's built-in EP model",
    subtitle = "2025 holdout",
    x = "cfbfastR ep_before",
    y = "Our model's prediction",
    caption = "Dashed red line = perfect agreement (y = x)."
  ) +
  theme_minimal(base_size = 12)

print(vs_cfb_plot)
ggsave("ep_models_vs_cfbfastr.png", vs_cfb_plot, width = 10, height = 5.5, dpi = 150)

# ------------------------------------------------------------------------------
# 9. EPA (Expected Points Added) FOR BOTH MODELS
# ------------------------------------------------------------------------------
# Generic helper: mirrors the standard sign-flip-on-possession-change logic,
# reused for both models rather than duplicating it. Uses dplyr's glue-string
# assignment ("{name}" := value), which requires dplyr >= 1.1.0 — if you're
# on an older dplyr, swap that line for: !!epa_col_name := ... (rlang syntax).
add_epa <- function(pbp_df, ep_col, epa_col_name) {
  pbp_df %>%
    arrange(game_id, id_play) %>%
    group_by(game_id) %>%
    mutate(
      .ep_next = lead(.data[[ep_col]]),
      .pos_team_next = lead(pos_team),
      .ep_next_adj = if_else(.pos_team_next == pos_team, .ep_next, -.ep_next),
      "{epa_col_name}" := .ep_next_adj - .data[[ep_col]]
    ) %>%
    ungroup() %>%
    select(-.ep_next, -.pos_team_next, -.ep_next_adj)
}

all_predictions <- bind_rows(fit_data, val_data, test_data) %>%
  select(game_id, id_play, pred_ep_cfb_model, pred_ep_nextscore_model)

pbp <- pbp %>%
  select(-any_of(c("pred_ep_cfb_model", "pred_ep_nextscore_model"))) %>%
  left_join(all_predictions, by = c("game_id", "id_play")) %>%
  add_epa("pred_ep_cfb_model", "epa_model_a") %>%
  add_epa("pred_ep_nextscore_model", "epa_model_b")

# ------------------------------------------------------------------------------
# 10. SAVE
# ------------------------------------------------------------------------------
saveRDS(model_ep_cfb, "cfb_ep_model_xgb.rds")
