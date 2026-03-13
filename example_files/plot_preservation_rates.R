#!/usr/bin/env Rscript
# plot_preservation_rates.R
#
# Helper script to plot and summarize preservation (sampling) rate posteriors
# from PyRate MCMC output (mcmc.log files).
#
# Usage (from terminal):
#   Rscript plot_preservation_rates.R <mcmc_log_file> [burnin_fraction] [epochs_file]
#
# Arguments:
#   mcmc_log_file    : path to the PyRate mcmc.log output file
#   burnin_fraction  : fraction of samples to discard as burn-in (default: 0.2)
#   epochs_file      : (optional) path to epochs_q.txt used with -qShift; enables
#                      plotting of TPP preservation rates through time
#
# Examples:
#   # HPP or NHPP model (no time-varying rates)
#   Rscript plot_preservation_rates.R pyrate_mcmc_logs/Felidae_preservation_1_mcmc.log
#
#   # TPP model with epoch boundaries
#   Rscript plot_preservation_rates.R pyrate_mcmc_logs/Felidae_preservation_1_mcmc.log \
#           0.2 example_files/epochs_q.txt
#
# Output:
#   A PDF file named <input_basename>_preservation_plot.pdf saved next to the log file.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  cat("Usage: Rscript plot_preservation_rates.R <mcmc_log_file> [burnin_fraction] [epochs_file]\n")
  quit(status = 1)
}

mcmc_file      <- args[1]
burnin_frac    <- if (length(args) >= 2) as.numeric(args[2]) else 0.2
epochs_file    <- if (length(args) >= 3) args[3] else NULL

if (!file.exists(mcmc_file)) {
  cat("Error: file not found:", mcmc_file, "\n")
  quit(status = 1)
}

# ── Read MCMC log ──────────────────────────────────────────────────────────────
log_data <- read.table(mcmc_file, header = TRUE, sep = "\t", comment.char = "")
n_samples <- nrow(log_data)
burnin    <- ceiling(n_samples * burnin_frac)
post      <- log_data[(burnin + 1):n_samples, ]
cat(sprintf("Read %d samples; discarding %d as burn-in (%g%%).\n",
            n_samples, burnin, burnin_frac * 100))

# ── Output PDF ────────────────────────────────────────────────────────────────
out_pdf <- file.path(dirname(mcmc_file),
                     paste0(sub("_mcmc\\.log$", "", basename(mcmc_file)),
                            "_preservation_plot.pdf"))
pdf(out_pdf, width = 10, height = 8)

# ── Detect available preservation columns ────────────────────────────────────
has_q_rate <- "q_rate" %in% names(post)
has_alpha  <- "alpha"  %in% names(post)
tpp_cols   <- grep("^q_[0-9]+$", names(post), value = TRUE)
has_tpp    <- length(tpp_cols) > 0

par(mfrow = c(2, 2))

# ── 1. Preservation rate posterior (HPP / NHPP / mean TPP) ───────────────────
if (has_q_rate) {
  q_vec <- post[["q_rate"]]
  hist(q_vec, breaks = 50, col = "steelblue", border = "white",
       main = "Posterior: preservation rate (q)",
       xlab = "q (occurrences / lineage / Myr)",
       ylab = "Frequency")
  abline(v = mean(q_vec),   col = "red",    lwd = 2, lty = 1)
  abline(v = median(q_vec), col = "orange", lwd = 2, lty = 2)
  q_ci <- quantile(q_vec, c(0.025, 0.975))
  abline(v = q_ci, col = "darkgray", lwd = 1.5, lty = 3)
  legend("topright",
         legend = c(sprintf("Mean = %.3f", mean(q_vec)),
                    sprintf("Median = %.3f", median(q_vec)),
                    sprintf("95%% CI [%.3f, %.3f]", q_ci[1], q_ci[2])),
         col = c("red", "orange", "darkgray"),
         lwd = c(2, 2, 1.5), lty = c(1, 2, 3), bty = "n", cex = 0.85)
}

# ── 2. Gamma shape (alpha) posterior – heterogeneity among lineages ───────────
if (has_alpha) {
  a_vec <- post[["alpha"]]
  hist(a_vec, breaks = 50, col = "darkorange", border = "white",
       main = "Posterior: Gamma shape (alpha)\nheterogeneity among lineages",
       xlab = expression(alpha ~ "(Gamma shape)"),
       ylab = "Frequency")
  abline(v = mean(a_vec),   col = "red",    lwd = 2, lty = 1)
  abline(v = median(a_vec), col = "black",  lwd = 2, lty = 2)
  a_ci <- quantile(a_vec, c(0.025, 0.975))
  abline(v = a_ci, col = "darkgray", lwd = 1.5, lty = 3)
  legend("topright",
         legend = c(sprintf("Mean = %.3f", mean(a_vec)),
                    sprintf("Median = %.3f", median(a_vec)),
                    sprintf("95%% CI [%.3f, %.3f]", a_ci[1], a_ci[2])),
         col = c("red", "black", "darkgray"),
         lwd = c(2, 2, 1.5), lty = c(1, 2, 3), bty = "n", cex = 0.85)
  mtext(expression("Low " * alpha * " → high heterogeneity; high " * alpha * " → homogeneous"),
        side = 1, line = 4, cex = 0.75)
}

# ── 3. MCMC trace of q_rate (convergence check) ───────────────────────────────
if (has_q_rate) {
  plot(seq_along(post[["q_rate"]]), post[["q_rate"]], type = "l",
       col = "steelblue", lwd = 0.5,
       main = "MCMC trace: q_rate (post burn-in)",
       xlab = "Sample (post burn-in)", ylab = "q_rate")
  abline(h = mean(post[["q_rate"]]), col = "red", lwd = 1.5, lty = 2)
}

# ── 4. TPP preservation rates through time ────────────────────────────────────
if (has_tpp && !is.null(epochs_file) && file.exists(epochs_file)) {
  epoch_boundaries <- scan(epochs_file, quiet = TRUE)
  # epoch midpoints (younger boundary open at 0 if not present)
  boundaries <- sort(epoch_boundaries, decreasing = FALSE)
  # q_0 is oldest interval → reverse order to match time axis
  tpp_means <- sapply(rev(tpp_cols), function(col) mean(post[[col]]))
  tpp_lo    <- sapply(rev(tpp_cols), function(col) quantile(post[[col]], 0.025))
  tpp_hi    <- sapply(rev(tpp_cols), function(col) quantile(post[[col]], 0.975))

  n_intervals <- length(tpp_means)
  # Build step-function x coordinates from epoch boundaries
  if (length(boundaries) >= n_intervals) {
    x_lo <- boundaries[seq_len(n_intervals)]
    # Extend the youngest interval slightly so it is visible as a non-zero-width bar
    YOUNGEST_INTERVAL_EXTENSION <- 1.02
    x_hi <- c(boundaries[seq_len(n_intervals - 1) + 1], boundaries[n_intervals] * YOUNGEST_INTERVAL_EXTENSION)
  } else {
    x_lo <- seq(0, by = 1, length.out = n_intervals)
    x_hi <- x_lo + 1
  }
  x_mid <- (x_lo + x_hi) / 2

  plot(x_mid, tpp_means, type = "n",
       xlim = rev(range(c(x_lo, x_hi))),
       ylim = range(c(0, tpp_hi * 1.1)),
       main = "TPP preservation rates through time",
       xlab = "Time (Ma)", ylab = "q (occurrences / lineage / Myr)")
  # Shaded 95% CI
  for (i in seq_along(x_lo)) {
    rect(x_hi[i], tpp_lo[i], x_lo[i], tpp_hi[i],
         col = adjustcolor("steelblue", alpha.f = 0.25), border = NA)
    segments(x_hi[i], tpp_means[i], x_lo[i], tpp_means[i],
             col = "steelblue", lwd = 2)
  }
  legend("topright",
         legend = c("Posterior mean", "95% CI"),
         col    = c("steelblue", adjustcolor("steelblue", alpha.f = 0.25)),
         lwd    = c(2, 8), bty = "n", cex = 0.85)
} else if (has_tpp) {
  # No epochs file: show bar chart of mean TPP rates
  tpp_means <- sapply(tpp_cols, function(col) mean(post[[col]]))
  tpp_lo    <- sapply(tpp_cols, function(col) quantile(post[[col]], 0.025))
  tpp_hi    <- sapply(tpp_cols, function(col) quantile(post[[col]], 0.975))
  bp <- barplot(tpp_means, names.arg = tpp_cols, las = 2,
                col = "steelblue", border = "white",
                main = "TPP preservation rates by interval",
                ylab = "Mean q", cex.names = 0.7)
  arrows(bp, tpp_lo, bp, tpp_hi, angle = 90, code = 3, length = 0.05)
} else {
  plot.new()
  text(0.5, 0.5, "No TPP columns found in log.\nRun with -qShift to enable\ntime-variable preservation rates.",
       cex = 1.1, adj = 0.5)
}

dev.off()
cat("Plot saved to:", out_pdf, "\n")

# ── Print summary table ────────────────────────────────────────────────────────
cat("\n── Preservation rate summary (post burn-in) ──\n")
pres_cols <- intersect(c("q_rate", "alpha", tpp_cols), names(post))
if (length(pres_cols) > 0) {
  summary_mat <- t(sapply(pres_cols, function(col) {
    v <- post[[col]]
    c(Mean   = round(mean(v),   4),
      Median = round(median(v), 4),
      SD     = round(sd(v),     4),
      CI_2.5 = round(quantile(v, 0.025), 4),
      CI_97.5= round(quantile(v, 0.975), 4))
  }))
  print(summary_mat)
}
