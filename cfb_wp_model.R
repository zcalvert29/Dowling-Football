# ==============================================================================
# Win Probability (WP) Model for College Football using cfbfastR
# ==============================================================================
# Self-contained — pulls and processes data the same way as cfb_ep_model.R,
# using the same season split, but does NOT depend on that script's output.
# Uses cfbfastR's own built-in ep_before as the EP input feature directly,
# rather than our own trained EP model's predictions.
#
# Compared against (1) cfbfastR's own built-in wp_before and (2) ground truth
# (which team actually won the game), on a 2025 holdout untouched by training.
#
# Split: fit on 2014-2023, early-stop on 2024, report final metrics only on
# 2025 — same split as cfb_ep_model.R, so results are directly comparable.
#
# Install once (run in a FRESH R session, before loading any libraries):
# install.packages(c("cfbfastR","dplyr","tidyr","stringr","xgboost","purrr","ggplot2","tibble"))
# Sys.setenv(CFBD_API_KEY = "your_key_here")  # get a free key at collegefootballdata.com
# ==============================================================================

library(cfbfastR)
library(dplyr)     # needs dplyr >= 1.1.0 for the glue-assignment syntax used below
library(tidyr)
library(tibble)
library(stringr)
library(purrr)
library(xgboost)

# ------------------------------------------------------------------------------
# 1. PULL PLAY-BY-PLAY DATA
# ------------------------------------------------------------------------------
# Same pull + caching approach as cfb_ep_model.R. If you already ran that
# script in this working directory, the pbp_YYYY.rds cache files it created
# will be reused here automatically — no need to re-download.

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
# 1c. PULL GAME-LEVEL FINAL SCORES (separate from play-by-play)
# ------------------------------------------------------------------------------
# Play-by-play does NOT contain final scores — pos_team_score/offense_score
# are running, per-play scores, not the game's final result. Final scores
# live on the game-results endpoint (cfbd_game_info()), which returns
# home_points/away_points, not home_score/away_score. Cached per season, same
# pattern as the pbp pull above.

pull_game_results <- function(yr, api_key = "") {
  path <- sprintf("game_results_%s.rds", yr)
  if (file.exists(path)) return(readRDS(path))
  Sys.setenv(CFBD_API_KEY = api_key)   # cfbfastR reads auth from this env var,
                                        # not a function argument — fill in
                                        # api_key above (or set it once at the
                                        # top of your session instead)
  games <- cfbfastR::cfbd_game_info(year = yr)
  saveRDS(games, path)
  games
}

game_results_raw <- map_dfr(seasons, pull_game_results, api_key = "")

required_game_cols <- c("game_id", "home_points", "away_points")
missing_game_cols <- setdiff(required_game_cols, names(game_results_raw))
if (length(missing_game_cols) > 0) {
  stop(
    "game_results_raw is missing expected columns: ", paste(missing_game_cols, collapse = ", "),
    "\nRun names(game_results_raw) to see what's actually available — cfbd_game_info()'s ",
    "schema has changed across cfbfastR versions before (e.g. home_score vs home_points)."
  )
}

final_scores <- game_results_raw %>%
  filter(!is.na(home_points), !is.na(away_points)) %>%
  distinct(game_id, .keep_all = TRUE) %>%
  transmute(game_id, home_final = home_points, away_final = away_points)

# ------------------------------------------------------------------------------
# 1d. SCHEMA CHECK (play-by-play)
# ------------------------------------------------------------------------------
# Note: home_score/away_score deliberately NOT required here — they don't
# exist in pbp (see 1c above). "home"/"away" (team name columns) do exist in
# pbp and are what pos_team gets compared against for is_home_pos.
required_cols <- c("game_id", "id_play", "period", "down", "distance",
                    "yards_to_goal", "TimeSecsRem", "pos_team",
                    "offense_score", "defense_score", "offense_timeouts",
                    "defense_timeouts", "play_type", "season", "ep_before",
                    "wp_before", "home")
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
# Same filtering as cfb_ep_model.R, for consistency.
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
    !is.na(TimeSecsRem),
    !is.na(ep_before)
  ) %>%
  arrange(game_id, id_play)

# ------------------------------------------------------------------------------
# 3. LABEL: DID THE TEAM WITH THE BALL ON THIS PLAY WIN THE GAME?
# ------------------------------------------------------------------------------
# final_scores comes from the game-results pull in Step 1c, not from pbp
# itself (pbp has no home_score/away_score columns to derive this from).
pbp <- pbp %>%
  left_join(final_scores, by = "game_id") %>%
  mutate(
    home_won = as.integer(home_final > away_final),
    tie_game = home_final == away_final
  ) %>%
  filter(!tie_game) %>%
  mutate(label_win = if_else(pos_team == home, home_won, 1L - home_won))

# ------------------------------------------------------------------------------
# 4. FEATURE ENGINEERING + FIT / VALIDATION / TEST SPLIT BY SEASON
# ------------------------------------------------------------------------------
required_step3_cols <- c("label_win")
missing_step3_cols <- setdiff(required_step3_cols, names(pbp))
if (length(missing_step3_cols) > 0) {
  stop(
    "pbp is missing columns created in Step 3: ", paste(missing_step3_cols, collapse = ", "),
    "\nRe-run the script from Step 1 through here in a single pass."
  )
}

model_data <- pbp %>%
  transmute(
    game_id, id_play,   # explicit join keys — do NOT rely on rownames()
    season,
    label_win,                                    # ground truth
    cfb_wp_before   = wp_before,                   # cfbfastR's own WP, for comparison
    ep              = ep_before,                   # cfbfastR's built-in EP, used AS A FEATURE
    score_diff      = pmin(pmax(offense_score - defense_score, -50), 50),
    yards_to_goal,
    down            = factor(down),
    distance        = pmin(distance, 30),   # cfbfastR calls this column "distance", not "ydstogo" — note the self-referential
                                             # pmin(distance, 30) is valid dplyr: RHS reads the source
                                             # column before this line's assignment overwrites it
    half_seconds    = pmin(TimeSecsRem, 1800),
    game_seconds_remaining = pmin(TimeSecsRem + (period <= 2) * 1800, 3600),
    off_timeouts    = offense_timeouts,
    def_timeouts    = defense_timeouts,
    is_home_pos     = as.integer(pos_team == home),
    score_diff_time_ratio = (offense_score - defense_score) / (
      pmin(TimeSecsRem + (period <= 2) * 1800, 3600) / 60 + 1
    )
  ) %>%
  drop_na(label_win, ep, score_diff, yards_to_goal, down, distance,
          half_seconds, game_seconds_remaining, off_timeouts, def_timeouts,
          is_home_pos, score_diff_time_ratio)

fit_data  <- model_data %>% filter(season %in% FIT_SEASONS)
val_data  <- model_data %>% filter(season == VAL_SEASON)
test_data <- model_data %>% filter(season == TEST_SEASON)

cat(sprintf("Fit rows:   %s (seasons %s-%s)\n", nrow(fit_data), min(FIT_SEASONS), max(FIT_SEASONS)))
cat(sprintf("Val rows:   %s (season %s)\n", nrow(val_data), VAL_SEASON))
cat(sprintf("Test rows:  %s (season %s)\n", nrow(test_data), TEST_SEASON))

# ------------------------------------------------------------------------------
# 5. FIT TWO MODELS
# ------------------------------------------------------------------------------
# Model 1 (wp_model_truth): target = label_win, the actual ground-truth
#   outcome. This is "our" win probability model.
# Model 2 (wp_model_cfb):   target = cfb_wp_before, cfbfastR's own built-in
#   WP. Mirrors Model A from cfb_ep_model.R (predicting the built-in field
#   rather than ground truth) — useful as a sanity check on how well our
#   feature set can reproduce cfbfastR's model, separate from how well
#   either model tracks reality.
#
# FEATURE_FORMULA deliberately has no response variable on the left — build_
# matrix() only needs the RHS to build the design matrix, and leaving the
# response off means the SAME function can build a matrix for prediction-only
# data frames (like the sanity-check scenarios in Step 9) that don't have a
# label_win/cfb_wp_before column at all, with no risk of formula/prediction-
# data mismatches.
#
# cfb_wp_before is NOT a feature in either model — for wp_model_truth that
# would leak the "answer" cfbfastR already computed; for wp_model_cfb it
# would let the model trivially copy its own target instead of learning to
# reconstruct it from game state.
FEATURE_FORMULA <- ~ ep + score_diff + yards_to_goal + down + distance +
                    half_seconds + game_seconds_remaining + off_timeouts +
                    def_timeouts + is_home_pos + score_diff_time_ratio - 1

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
    objective = "binary:logistic",   # valid for continuous [0,1] targets too
                                      # (cross-entropy against a soft label),
                                      # not just 0/1 — used for both models here
    eval_metric = "logloss",
    max_depth = 4,
    eta = 0.03,
    nrounds = 800,
    subsample = 0.8,
    colsample_bytree = 0.7,
    watchlist = list(fit = dfit, val = dval),
    early_stopping_rounds = 30,
    print_every_n = 50
  )
}

wp_model_truth <- fit_one_model(fit_data$label_win, val_data$label_win,
                                 "WP Model (target = ground truth win/loss)")
wp_model_cfb   <- fit_one_model(fit_data$cfb_wp_before, val_data$cfb_wp_before,
                                 "WP Model (target = cfbfastR wp_before)")

# ------------------------------------------------------------------------------
# 6. GENERATE PREDICTIONS
# ------------------------------------------------------------------------------
test_data$pred_wp_truth <- predict(wp_model_truth, x_test)
test_data$pred_wp_cfb   <- predict(wp_model_cfb, x_test)

fit_data$pred_wp_truth  <- predict(wp_model_truth, x_fit)
fit_data$pred_wp_cfb    <- predict(wp_model_cfb, x_fit)
val_data$pred_wp_truth  <- predict(wp_model_truth, x_val)
val_data$pred_wp_cfb    <- predict(wp_model_cfb, x_val)

# ------------------------------------------------------------------------------
# 7. COMPARISON — ON THE 2025 HOLDOUT ONLY
# ------------------------------------------------------------------------------
# Two kinds of comparison, same reasoning as cfb_ep_model.R:
#  (a) vs. cfbfastR's wp_before — both sides are probabilities, so pointwise
#      correlation/RMSE is directly meaningful. wp_model_cfb should track
#      this tightly (it's literally the target); wp_model_truth is expected
#      to diverge somewhat since it's targeting something else.
#  (b) vs. label_win (ground truth, 0/1) — proper scoring rules (Brier score,
#      log loss), not raw RMSE. Binned calibration is the most useful
#      diagnostic of the three for judging real quality.

brier_score <- function(pred, actual) mean((pred - actual)^2)
log_loss <- function(pred, actual, eps = 1e-15) {
  pred <- pmin(pmax(pred, eps), 1 - eps)
  -mean(actual * log(pred) + (1 - actual) * log(1 - pred))
}

vs_cfb_metrics <- tibble(
  comparison    = c("wp_model_truth vs. cfbfastR WP", "wp_model_cfb vs. cfbfastR WP"),
  n             = nrow(test_data),
  correlation   = c(cor(test_data$pred_wp_truth, test_data$cfb_wp_before, use = "complete.obs"),
                     cor(test_data$pred_wp_cfb, test_data$cfb_wp_before, use = "complete.obs")),
  rmse          = c(sqrt(mean((test_data$pred_wp_truth - test_data$cfb_wp_before)^2)),
                     sqrt(mean((test_data$pred_wp_cfb - test_data$cfb_wp_before)^2))),
  mean_abs_diff = c(mean(abs(test_data$pred_wp_truth - test_data$cfb_wp_before)),
                     mean(abs(test_data$pred_wp_cfb - test_data$cfb_wp_before))),
  mean_diff     = c(mean(test_data$pred_wp_truth - test_data$cfb_wp_before),
                     mean(test_data$pred_wp_cfb - test_data$cfb_wp_before))
)
cat("\n=== Pointwise comparison vs. cfbfastR WP (2025 holdout) ===\n")
print(vs_cfb_metrics)

vs_truth_metrics <- tibble(
  comparison  = c("cfbfastR WP vs. ground truth (baseline)",
                   "wp_model_truth vs. ground truth",
                   "wp_model_cfb vs. ground truth"),
  n           = nrow(test_data),
  brier_score = c(brier_score(test_data$cfb_wp_before, test_data$label_win),
                   brier_score(test_data$pred_wp_truth, test_data$label_win),
                   brier_score(test_data$pred_wp_cfb, test_data$label_win)),
  log_loss    = c(log_loss(test_data$cfb_wp_before, test_data$label_win),
                   log_loss(test_data$pred_wp_truth, test_data$label_win),
                   log_loss(test_data$pred_wp_cfb, test_data$label_win)),
  mean_diff   = c(mean(test_data$cfb_wp_before - test_data$label_win),
                   mean(test_data$pred_wp_truth - test_data$label_win),
                   mean(test_data$pred_wp_cfb - test_data$label_win))
)
cat("\n=== Comparison vs. ground truth (2025 holdout) — lower is better ===\n")
print(vs_truth_metrics)

# Binned calibration — the metric that actually matters for judging WP quality
calibration_table <- function(pred, actual, n_bins = 20) {
  tibble(pred = pred, actual = actual) %>%
    mutate(bin = ntile(pred, n_bins)) %>%
    group_by(bin) %>%
    summarise(mean_predicted = mean(pred), actual_win_rate = mean(actual), n = n()) %>%
    ungroup()
}

cat("\n=== Calibration vs. ground truth: cfbfastR WP (baseline) ===\n")
print(calibration_table(test_data$cfb_wp_before, test_data$label_win))

cat("\n=== Calibration vs. ground truth: wp_model_truth ===\n")
print(calibration_table(test_data$pred_wp_truth, test_data$label_win))

cat("\n=== Calibration vs. ground truth: wp_model_cfb ===\n")
print(calibration_table(test_data$pred_wp_cfb, test_data$label_win))

# ------------------------------------------------------------------------------
# 8. RENDERED PLOTS
# ------------------------------------------------------------------------------
library(ggplot2)

# Plot 1: calibration curves, all three predictors overlaid
calibration_combined <- bind_rows(
  calibration_table(test_data$cfb_wp_before, test_data$label_win) %>% mutate(source = "cfbfastR WP (baseline)"),
  calibration_table(test_data$pred_wp_truth, test_data$label_win) %>% mutate(source = "wp_model_truth"),
  calibration_table(test_data$pred_wp_cfb, test_data$label_win) %>% mutate(source = "wp_model_cfb")
)

wp_calibration_plot <- calibration_combined %>%
  filter(n >= 20) %>%
  ggplot(aes(mean_predicted, actual_win_rate, color = source)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40", linewidth = 0.7) +
  geom_line(linewidth = 0.9, alpha = 0.7) +
  geom_point(aes(size = n), alpha = 0.85) +
  scale_size_continuous(range = c(1.5, 7), guide = "none") +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title = "Win Probability calibration against ground truth (2025 holdout)",
    x = "Mean predicted win probability (binned)",
    y = "Actual win rate",
    color = NULL,
    caption = "Dashed line = perfect calibration."
  ) +
  theme_minimal(base_size = 12)

print(wp_calibration_plot)
ggsave("wp_calibration_comparison.png", wp_calibration_plot, width = 8, height = 6.5, dpi = 150)

# Plot 2: pointwise scatter, both models vs. cfbfastR's WP, faceted
scatter_data <- bind_rows(
  test_data %>% transmute(cfb_wp_before, pred = pred_wp_truth, model = "wp_model_truth"),
  test_data %>% transmute(cfb_wp_before, pred = pred_wp_cfb, model = "wp_model_cfb")
)

wp_vs_cfb_plot <- scatter_data %>%
  ggplot(aes(cfb_wp_before, pred)) +
  geom_point(alpha = 0.03, size = 0.6, color = "#4C72B0") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#C44E52", linewidth = 0.8) +
  facet_wrap(~model) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title = "Our WP models vs. cfbfastR's built-in WP model",
    subtitle = "2025 holdout",
    x = "cfbfastR wp_before",
    y = "Our model's predicted WP",
    caption = "Dashed red line = perfect agreement (y = x)."
  ) +
  theme_minimal(base_size = 12)

print(wp_vs_cfb_plot)
ggsave("wp_models_vs_cfbfastr.png", wp_vs_cfb_plot, width = 10, height = 5.5, dpi = 150)

# ------------------------------------------------------------------------------
# 9. FEATURE IMPORTANCE + SANITY CHECKS
# ------------------------------------------------------------------------------
cat("\n=== Feature importance: wp_model_truth ===\n")
print(xgb.importance(model = wp_model_truth))
cat("\n=== Feature importance: wp_model_cfb ===\n")
print(xgb.importance(model = wp_model_cfb))

# Both sanity checks now go through build_matrix() — the same function used
# to build the training matrices — instead of a separately-written
# model.matrix(~ . - 1, ...) call. That separate call was a latent
# inconsistency risk (different formula object, easy for the two to drift
# apart silently); routing through build_matrix() removes that risk entirely.
# It also only works cleanly now that FEATURE_FORMULA has no response
# variable — previously it required a label_win column to even exist in
# these one-row scenario data frames.

# 1st-and-10 at midfield, tied game, start of 3rd quarter should be near 50%
sample_situation <- data.frame(
  ep = 0, score_diff = 0, yards_to_goal = 50,
  down = factor("1", levels = levels(model_data$down)),
  distance = 10, half_seconds = 900, game_seconds_remaining = 1800,
  off_timeouts = 3, def_timeouts = 3, is_home_pos = 1, score_diff_time_ratio = 0
)
sample_x <- build_matrix(sample_situation)
cat("\nSanity check — tied game, midfield, start of 3rd quarter (~0.5 expected):\n")
cat("wp_model_truth:", predict(wp_model_truth, sample_x), "\n")
cat("wp_model_cfb:  ", predict(wp_model_cfb, sample_x), "\n")

# 35-point lead with 2 minutes left "should" be ~99%+. If this comes out
# meaningfully lower (e.g. ~0.85-0.90) even after the build_matrix() fix
# above, that's most likely NOT a bug — it's a known, common property of
# regularized tree-based WP models: eta/subsample/colsample shrinkage and
# L2 regularization actively discourage the model from pushing predictions
# all the way to the extremes, even in situations that are "obviously" near-
# certain to a human. It tends to show up more with fewer boosting rounds or
# thinner training data at extreme feature combinations (blowout garbage-
# time situations are a small fraction of all plays). If you want the model
# to push closer to the extremes, options worth trying: lower eta with more
# nrounds, reduce reg_lambda/reg_alpha, or check how many fit-set examples
# actually resemble this exact situation (large score_diff, very little
# time left) — a genuinely thin region will always be less confident.
blowout_situation <- sample_situation %>%
  mutate(score_diff = 35, half_seconds = 120, game_seconds_remaining = 120,
         score_diff_time_ratio = 35 / (120 / 60 + 1))
blowout_x <- build_matrix(blowout_situation)
cat("\nSanity check — 35-pt lead, 2 min left (near-certain win, but see note above on why this may read < 0.99):\n")
cat("wp_model_truth:", predict(wp_model_truth, blowout_x), "\n")
cat("wp_model_cfb:  ", predict(wp_model_cfb, blowout_x), "\n")

# ------------------------------------------------------------------------------
# 10. ATTACH PREDICTIONS TO FULL PBP
# ------------------------------------------------------------------------------
all_predictions <- bind_rows(fit_data, val_data, test_data) %>%
  select(game_id, id_play, pred_wp_truth, pred_wp_cfb)

pbp <- pbp %>%
  select(-any_of(c("pred_wp_truth", "pred_wp_cfb"))) %>%
  left_join(all_predictions, by = c("game_id", "id_play"))

# ------------------------------------------------------------------------------
# 11. SAVE
# ------------------------------------------------------------------------------
saveRDS(wp_model_truth, "cfb_wp_model_truth_xgb.rds")
saveRDS(wp_model_cfb, "cfb_wp_model_cfb_xgb.rds")
saveRDS(pbp, "cfb_pbp_with_wp_predictions.rds")

# ==============================================================================
# Export trained xgboost models (saved via saveRDS) to portable JSON
# ==============================================================================
# saveRDS() wraps a model in R's own serialization format — Python's xgboost
# can't load that directly. xgb.save() writes the model in xgboost's own
# portable format instead, which IS loadable from Python via
# xgb.Booster().load_model(). Run this once (or whenever you retrain), then
# copy the resulting .json files into the Streamlit app's folder.
#
# Run this from the same working directory where the .rds files live.
# ==============================================================================
 
library(xgboost)
 
ep_model <- readRDS("cfb_ep_model_xgb.rds")
xgb.save(ep_model, "cfb_ep_model.json")
 
wp_model_truth <- readRDS("cfb_wp_model_truth_xgb.rds")
xgb.save(wp_model_truth, "cfb_wp_model_truth.json")
 
wp_model_cfb <- readRDS("cfb_wp_model_cfb_xgb.rds")
xgb.save(wp_model_cfb, "cfb_wp_model_cfb.json")
 
cat("Exported: cfb_ep_model.json, cfb_wp_model_truth.json, cfb_wp_model_cfb.json\n")
cat("Copy these three files into the Streamlit app's folder (same directory as app.py).\n")
