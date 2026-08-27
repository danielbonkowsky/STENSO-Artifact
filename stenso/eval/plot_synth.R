library(ggplot2)
library(reshape2)
library(scales)
library(RColorBrewer)
library(dplyr)
library(tidyr)


get_benchmark_names <- function() {
  files <- list.files("../stenso/eval/benchmarks", pattern="*.py", full.names=FALSE)
  benchmarks <- gsub(".py", "", files)
  return(benchmarks)
}
benchmark_names <- get_benchmark_names()


# Figure 5: Synthesis time of STENSO with and without branch and bound
# #########################################
cOne <- "#3c76af"
cTwo <- "#83A0BD"
cThree <- "#cacaca"

rename_benchmarks <- function(df) {
  df_names <- read.csv("../stenso/eval/benchmark_names.csv")  # Replace benchmarks in "Benchmark" column according to df_names. Lookup in "from" column and replace with "to" column.
  # If not found, keep original name.
  df <- merge(df, df_names, by.x="benchmark", by.y="from", all.x=TRUE)
  df$benchmark <- ifelse(is.na(df$to), df$benchmark, df$to)
  df <- df %>% select(-to)
  return(df)
}

df <- read.csv("stats.csv")
df$runtime <- ifelse(is.na(df$runtime), df$runtime, df$runtime)

# Select columns "benchmark", "runtime", "search_config"
df <- df %>%
  select(benchmark, runtime, search_config)

# Add missing rows. Ground truth is the cartesian product of all benchmarks in benchmark_names and search_config
all_search_configs <- unique(df$search_config)
all_combinations <- expand.grid(benchmark = benchmark_names, search_config = all_search_configs, stringsAsFactors = FALSE)
df <- merge(all_combinations, df, by=c("benchmark", "search_config"), all.x=TRUE)
# Replace NA in runtime with 360
df$runtime[is.na(df$runtime)] <- 360


df$Synthesizer <- recode(df$search_config, "stenso_branch_and_bound" = "Simplification + Branch and Bound", "stenso_exhaustive" = "Simplification", "bottom_up" = "Bottom Up")

df <- rename_benchmarks(df)

# Reorder
df$Synthesizer <- factor(df$Synthesizer, levels=c("Bottom Up", "Simplification", "Simplification + Branch and Bound"))

reorder_benchmarks_by_runtime <- function(df, synthesizer) {
  simplification_times <- df %>%
    filter(Synthesizer == synthesizer) %>%
    select(benchmark, runtime)
  
  ordered_benchmarks <- simplification_times %>%
    arrange(runtime) %>%
    pull(benchmark)
  
  df$benchmark <- factor(df$benchmark, levels = ordered_benchmarks)
  
  return(df)
}
df <- reorder_benchmarks_by_runtime(df, "Simplification")

ggplot(df, aes(x=benchmark, y=runtime, fill=Synthesizer)) +
  geom_col(position="dodge") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle=45, hjust=1),
    legend.position = "top",
  ) +
  labs(x="Benchmark", y="Synthesis Time") +
  geom_hline(yintercept=360) +
  scale_fill_manual(values=c(cThree, cTwo, cOne)) +
  annotate("text", x = 1, y = 360, label = "timeout", hjust = 0, vjust = 1.5)
ggsave("fig_5.pdf", width=8, height=3.5, units="in")