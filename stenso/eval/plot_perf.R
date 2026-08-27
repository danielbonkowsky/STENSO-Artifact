library(ggplot2)
library(reshape2)
library(scales)
library(RColorBrewer)
library(dplyr)
library(tidyr)

cNumpy <- "#999999"
cJax <- "#E69F00"
cTorch <- "#009E73"
cTorchCompiled <- "#56B4E9"

cSTENSO <- "#B21C0E"

cSynthetic <- "#cacaca"
cReal <- "#3c76af"

gm_mean <- function(x, na.rm=TRUE){
  exp(sum(log(x[x > 0]), na.rm=na.rm) / length(x))
}

rename_benchmarks <- function(df) {
  df_names <- read.csv("../stenso/eval/benchmark_names.csv")
  # Replace benchmarks in "Benchmark" column according to df_names. Lookup in "from" column and replace with "to" column.
  # If not found, keep original name.
  df <- merge(df, df_names, by.x="Benchmark", by.y="from", all.x=TRUE)
  df$Benchmark <- ifelse(is.na(df$to), df$Benchmark, df$to)
  df <- df %>% select(-to)
  return(df)
}

get_speedup_df <- function() {
  # Parse all csv files from directory "performance_results". And add a column "Platform" with the filename (without extension).
  files <- list.files("../stenso/eval/performance_results", pattern="*.csv", full.names=TRUE)
  df_list <- lapply(files, function(file) {
    df <- read.csv(file)
    df$Platform <- gsub(".csv", "", basename(file))  # Extract filename without extension
    return(df)
  })
  # Combine all dataframes into one
  df <- do.call(rbind, df_list)
  
  # Replace all "UnknownVendor_" in Platform with ""
  df$Platform <- gsub("UnknownVendor_", "", df$Platform)
  
  # Calculate speedup
  df$Speedup <- df$OriginalTime / df$OptimizedTime
  # Remove columns "OriginalTime" and "OptimizedTime"
  df <- df %>%
      select(-OriginalTime, -OptimizedTime)
  
  df <- rename_benchmarks(df)
  
  return(df)
}

get_classes_df <- function() {
  df_classes <- read.csv("../stenso/eval/benchmark_classes.csv")
  df_classes <- rename_benchmarks(df_classes)
  return(df_classes) 
}


add_geomean_to_df <- function(df) {
  # Calculate geomean per platform
  # - Filter out inf and NaN values
  df2 <- df %>%
      filter(!is.infinite(Speedup) & !is.nan(Speedup))
  # - Calculate geometric mean for each Compiler and Platform
  #df_gm <- aggregate(Speedup ~ Compiler, df2, gm_mean)
  df_gm <- aggregate(Speedup ~ Compiler + Platform, df2, gm_mean)
  df_gm$Benchmark <- "Geometric Mean"
  # - Combine with original data.
  df <- rbind(df, df_gm)
  return(df)
}

reorder_and_rename <- function(df) {
  # Reorder variants to NumPy, JAX, PyTorch
  df$Compiler <- factor(df$Compiler, levels=c("JAX", "PyTorch", "NumPy"))

  # Reorder benchmarks so that the "Geometric Mean" is last
  order <- c(setdiff(unique(df$Benchmark), "Geometric Mean"), "Geometric Mean")
  df$Benchmark <- factor(df$Benchmark, levels=order)
  
  # Rename "JAX" to "JAX (Compiled)" and "PyTorch" to "PyTorch (Compiled)"
  df$Compiler <- recode(df$Compiler, "JAX" = "JAX (Compiled)", "PyTorch" = "PyTorch (Compiled)")
  return(df)
}

# Figure 4: Summary
# ##############################
df <- get_speedup_df()
df <- add_geomean_to_df(df)
df <- reorder_and_rename(df)

df_plot <- df %>% filter(Benchmark== "Geometric Mean")
df_plot$Code<- "STENSO"

# Add another code "Original" with all Speedups being 1.
df_plot_original <- df_plot %>%
    select(Benchmark, Compiler, Platform, Speedup) %>%
    mutate(Speedup = 1, Code= "Original")
df_plot <- rbind(df_plot, df_plot_original)

# Rename Compilers
df_plot$Compiler <- recode(df_plot$Compiler, "JAX (Compiled)" = "JAX\n(Compiled)", "PyTorch (Compiled)" = "PyTorch\n(Compiled)")



# Create plot. Y axis speedup. X axis backend. Facets Platform.
ggplot(df_plot, aes(x=Compiler, y=Speedup, fill=Code)) +
    facet_wrap(~Platform, nrow=1) +
    geom_col(position="dodge") +
    geom_hline(yintercept=1, linetype="dashed") +
    scale_y_continuous(
      trans = pseudo_log_trans(base = 10),
      breaks = c(1, 2^(0:8)),
      minor_breaks = rep(1:9, 21)*(2^rep(-10:10, each=9)),
      labels = label_number(accuracy = 1),
      expand = expansion(mult = c(0, .25))
    ) +
    scale_fill_manual(values=c(cNumpy, cSTENSO, cTorch, cTorchCompiled)) +
    geom_text(aes(label=sprintf("%.1fx", Speedup)), vjust=-0.3, size=3, position=position_dodge(width=0.9)) +
    theme_minimal() +
    theme(
        axis.text.x = element_text(angle=45, hjust=1),
        legend.position = "right",
    ) + 
    labs(fill="Implementation", x="Framework", y="Geometric Mean Speedup")
ggsave("fig_4.pdf", width=8, height=3, units="in")

# Figure 8
# ##############################
df <- get_speedup_df()
df <- reorder_and_rename(df)
df_classes <- get_classes_df()
df <- merge(df, df_classes, by="Benchmark")

ggplot(df, aes(x=Benchmark, y=Speedup, fill=Compiler)) +
  facet_grid(Platform ~ Class, scales = "free_x", space = "free_x") +
  geom_col(position="dodge") +
  geom_hline(yintercept=1, linetype="dashed") +
  scale_y_continuous(
    trans = pseudo_log_trans(base = 10),
    breaks = c(1, 2^(0:8)),
    minor_breaks = rep(1:9, 21)*(2^rep(-10:10, each=9)),
    labels = label_number(accuracy = 1),
    expand = expansion(mult = c(0, .25))
  ) +
  scale_fill_manual(values=c(cJax, cTorch, cNumpy, cTorchCompiled)) +
  geom_text(aes(label=sprintf("%.1fx", Speedup)), 
            hjust=-0.2, 
            size=2.5, 
            position=position_dodge(width=0.9), 
            angle=90) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle=45, hjust=1),
    legend.position = "top",
    panel.spacing.x = unit(1, "lines") 
  ) +
  labs(fill="Framework")
ggsave("fig_8.pdf", width=8, height=6, units="in")

# Figure 7
# ##############################
df <- get_speedup_df()
df <- reorder_and_rename(df)
df_classes <- get_classes_df()
df <- merge(df, df_classes, by="Benchmark")
df_gm <- df %>%
  filter(!is.infinite(Speedup) & !is.nan(Speedup)) %>%
  group_by(Class, Compiler, Platform) %>%
  summarize(Speedup = gm_mean(Speedup)) %>%
  ungroup()

df_gm$Class <- gsub(" ", "\n", df_gm$Class)

ggplot(df_gm, aes(x=Class, y=Speedup, fill=Compiler)) +
  facet_grid(Platform ~ .) +
  geom_col(position="dodge") +
  geom_hline(yintercept=1, linetype="dashed") +
  scale_y_continuous(
    trans = pseudo_log_trans(base = 10),
    breaks = c(1, 2^(0:8)),
    minor_breaks = rep(1:9, 21)*(2^rep(-10:10, each=9)),
    labels = label_number(accuracy = 1),
    expand = expansion(mult = c(0, .25))
  ) +
  scale_fill_manual(values=c(cJax, cTorch, cNumpy, cTorchCompiled)) +
  geom_text(aes(label=sprintf("%.1fx", Speedup)), 
            hjust=0.5,
            vjust=-0.3,
            size=3, 
            position=position_dodge(width=0.9)) +
  theme_minimal() +
  theme(
    legend.position = "right",
    panel.spacing.x = unit(1, "lines") 
  ) +
  labs(fill="Framework", x="Transformation", y="Geometric Mean Speedup")
ggsave("fig_7.pdf", width=8, height=6, units="in")

# Figure 6
# #############################
df_classes <- read.csv("../stenso/eval/benchmark_classes.csv")
df_count <- df_classes %>%
  group_by(Class) %>%
  summarize(Count = n()) %>%
  ungroup()

df_count$Class <- gsub(" ", "\n", df_count$Class)

ggplot(df_count, aes(x=Class, y=Count)) +
  geom_col(width=0.4) +
  theme_minimal() +
  theme(
    panel.spacing.x = unit(1, "lines"),
  ) +
  labs(x="Transformation", y="Number of Benchmarks") +
  scale_fill_manual(values=c(cSynthetic, cReal)) +
  scale_y_continuous(breaks = scales::pretty_breaks(n = max(df_count$Count)))
ggsave("fig_6.pdf", width=4.5, height=2.5, units="in")

