# Crossed random-effects cumulative link mixed model: model fixed, random intercepts for conversation and rater.
# Sensitivity analysis for the model that treats rater as a fixed effect (revision_rating_analyses.py).
# install.packages("ordinal")
# Run from the repo root: Rscript scripts/revision_clmm_crossed.R
# Input:  rag_models/results/revision/ratings_long.csv
# Output: rag_models/results/revision/clmm_crossed.csv
library(ordinal)
dir <- "rag_models/results/revision"
d <- read.csv(file.path(dir, "ratings_long.csv"))
d <- subset(d, failed_generation == 0)
d$model <- relevel(factor(d$model), ref = "gpt-oss-120b")
d$conversation <- factor(d$conversation); d$rater <- factor(d$rater)
cat("Ratings:", nrow(d), " conversations:", nlevels(d$conversation), " raters:", nlevels(d$rater), "\n")
rows <- list()
for (dim in c("accuracy", "completeness", "conciseness", "clinical_utility", "overall_quality")) {
  d$y <- factor(d[[dim]], ordered = TRUE)
  f1 <- clmm(y ~ model + (1 | conversation) + (1 | rater), data = d, link = "logit", nAGQ = 1)
  f0 <- clmm(y ~ 1 + (1 | conversation) + (1 | rater), data = d, link = "logit", nAGQ = 1)
  a <- anova(f0, f1)
  cat("\n====", dim, "====\n"); print(a); print(summary(f1)$coefficients); print(VarCorr(f1))
  co <- summary(f1)$coefficients
  vc <- VarCorr(f1)
  row <- data.frame(dimension = dim, n = nrow(d),
                    lr_chisq = a$LR.stat[2], df = a$df[2], p = a$`Pr(>Chisq)`[2],
                    conversation_sd = sqrt(as.numeric(vc$conversation)), rater_sd = sqrt(as.numeric(vc$rater)),
                    converged = f1$optRes$convergence == 0,
                    max_abs_gradient = max(abs(f1$gradient)), cond_hessian = summary(f1)$condHess)
  for (m in c("gpt-oss-20b", "qwen3-32b", "medgemma-4b")) {
    b <- co[paste0("model", m), "Estimate"]; s <- co[paste0("model", m), "Std. Error"]
    row[[paste0("or_", m)]] <- exp(b)
    row[[paste0("or_", m, "_lo")]] <- exp(b - qnorm(.975) * s)
    row[[paste0("or_", m, "_hi")]] <- exp(b + qnorm(.975) * s)
  }
  rows[[dim]] <- row
}
out <- do.call(rbind, rows)
write.csv(out, file.path(dir, "clmm_crossed.csv"), row.names = FALSE)
cat("\n"); print(out, digits = 4, row.names = FALSE)
# Organization is left out: both gpt-oss models scored 5 on every rating (complete separation).
# With three raters the rater variance rests on three levels and is poorly estimated.
