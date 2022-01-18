suppressPackageStartupMessages({
  library(tidyverse)
  library(magrittr)
  library(stickylabeller)
  library(lemon)
  library(patchwork)
  library(jsonlite)
  library(broom)
})

knitr::opts_chunk$set(
    warning=FALSE, message=FALSE, fig.width=6, fig.height=4, fig.align="center"
)

options(
  "summ-model.info"=FALSE, 
  "summ-model.fit"=FALSE, 
  "summ-re.table"=FALSE, 
  "summ-groups.table"=FALSE,
  "jtools-digits"=3,
  "pillar.subtle" = FALSE

)

RED =  "#E41A1C" 
BLUE =  "#377EB8" 
GREEN =  "#4DAF4A" 
PURPLE =  "#984EA3" 
ORANGE =  "#FF7F00" 
YELLOW =  "#FFDD47" 
GRAY = "#ADADAD"
BLACK = "#111111"

theme_set(theme_classic(base_size = 14) + theme(
  strip.background = element_blank(),
  strip.text.x = element_text(size=14),
  plot.tag.position = c(0, 1)
))
update_geom_defaults("line", list(size = 1.2))

feature_colors = scale_colour_manual(values=c(
  "2"="darkgray", "5"="#E65E68"
), aesthetics=c("fill", "colour"), name="Features")

nudge_colors = scale_colour_manual(values=c(
    "darkgray",
    "dodgerblue"
), aesthetics=c("fill", "colour"), name="Nudge")

kable = knitr::kable
glue = glue::glue

system('mkdir -p figs')
system('mkdir -p .fighist')
fig = function(name="tmp", w=4, h=4, dpi=320, ...) {
    ggsave("/tmp/fig.png", width=w, height=h, dpi=dpi, ...)
    stamp = format(Sys.time(), "%m-%d-%H-%M-%S")
    p = glue('".fighist/{gsub("/", "-", name)}-{stamp}.png"')
    system(glue('mkdir -p `dirname {name}`'))
    system(glue('mv /tmp/fig.png {p}'))
    system(glue('cp {p} figs/"{name}".png'))
    if(name != "tmp") {
      ggsave(glue("figs/{name}.pdf"), width=w, height=h, dpi=dpi, ...)
    }
    # invisible(dev.off())
    # knitr::include_graphics(p)
}


only = function(xs) {
  u = unique(xs)
  stopifnot(length(u) == 1)
  u[1]
}

inject = rlang::inject
tidylm = function(data, xvar, yvar) {
    y = ensym(yvar)
    x = ensym(xvar)
    inject(lm(!!y ~ !!x, data=data))
}

# %% ==================== Saving results ====================

sprintf_transformer <- function(text, envir) {
  m <- regexpr(":.+$", text)
  if (m != -1) {
    format <- substring(regmatches(text, m), 2)
    regmatches(text, m) <- ""
    res <- eval(parse(text = text, keep.source = FALSE), envir)
    do.call(sprintf, list(glue("%{format}f"), res))
  } else {
    eval(parse(text = text, keep.source = FALSE), envir)
  }
}

fmt <- function(..., .envir = parent.frame()) {
  glue(..., .transformer = sprintf_transformer, .envir = .envir)
}

pval = function(p) {
  # if (p < .001) "p < .001" else glue("p = {str_sub(format(round(p, 3)), 2)}")
  if (p < .001) "p < .001" else glue("p = {str_sub(format(round(p, 3), nsmall=3), 2)}")
}

tex_writer = function(path) {
  dir.create(path, recursive=TRUE, showWarnings=FALSE)
  function(name, tex) {
    name = glue(name, .envir=parent.frame()) %>% str_replace("[:*]", "-")
    tex = fmt(tex, .envir=parent.frame())
    file = glue("{path}/{name}.tex")
    print(paste0(file, ": ", tex))
    writeLines(paste0(tex, "\\unskip"), file)
  }
}

write_tex = function(file, tex) {
  if (!endsWith(file, ".tex")) {
    file = paste0(file, ".tex")
  }
  file = glue(file, .envir=parent.frame())
  file = str_replace(file, "[:*]", "-")
  dir.create(dirname(file), recursive=TRUE, showWarnings=FALSE)
  tex = fmt(tex, .envir=parent.frame())
  print(paste0(file, ": ", tex))
  writeLines(paste0(tex, "\\unskip"), file)
}

ONE_TAILED = TRUE  # as pre-registered

write_model = function(model, path) UseMethod("write_model")
write_model.glm = function(model, path) {
  path = glue(path, .envir=parent.frame())
  tidy(model) %>% 
      filter(term != "(Intercept)") %>% 
      mutate(p.value = if (ONE_TAILED) p.value / 2 else p.value) %>% 
      rowwise() %>% group_walk(~ with(.x, 
          write_tex("{path}/{term}.tex",
                    "$z={statistic:.2},\\ {pval(p.value)}$")
      ))
}

write_model.lm = function(model, path) {
    path = glue(path, .envir=parent.frame())
    tidy(model) %>% 
        filter(term != "(Intercept)") %>% 
        mutate(p.value = if (ONE_TAILED) p.value / 2 else p.value) %>% 
        rowwise() %>% group_walk(~ with(.x, 
            write_tex("{path}/{term}.tex",
                      "$t({model$df})={statistic:.2},\\ {pval(p.value)}$")
        ))
}

# %% ==================== Exclusions ====================

args = commandArgs(trailingOnly=TRUE)
EXCLUDE = !(length(args) > 0 & args[1] == "--full") 

apply_exclusion = function(data, is_control, rate=0.5) {
  if (!EXCLUDE) return(data)
  keep = data %>%
      filter({{is_control}}) %>% 
      group_by(participant_id) %>% 
      summarise(no_click_rate = mean(num_values_revealed == 0)) %>%
      filter(no_click_rate <= rate) %>% 
      with(participant_id)
  n_total = length(unique(data$participant_id))
  n_exclude = n_total - length(keep)
  print(glue("Excluding {n_exclude}/{n_total} participants who didn't click on {100*rate}% of control trials"))
  filter(data, participant_id %in% keep)
}

report_exclusion = function(path, human_raw, human) {
  n_original = length(unique(human_raw$participant_id))
  n_final = length(unique(human$participant_id))
  write_tex("{path}/n_final", n_final)
  write_tex("{path}/n_exclude", n_original - n_final)
  write_tex("{path}/percent_exclude", round(100*(n_original-n_final) / (n_original)))
  write_tex("{path}/n_trial", nrow(human))
}

# %% ==================== Plot utils ====================

savefig = function(name, w, h) {
  fn = paste0(name, if(EXCLUDE) "" else "-full")
  fig(fn, w, h)
}

point_and_error = list(
    stat_summary(fun.data=mean_cl_boot, geom="errorbar", width=.1, size=.3),
    stat_summary(fun=mean, geom="point", size=1)
)

point_and_error_fast = list(
    stat_summary(fun.data=mean_cl_normal, geom="errorbar", width=.1, size=.3),
    stat_summary(fun=mean, geom="point", size=1)
)

option_feature_grid = facet_grid(n_option ~ n_feature, 
    labeller = label_glue(
        rows = "{n_option} Options",
        cols = "{n_feature} Features"
    )
)

option_feature_grid_rep = facet_rep_grid(n_option ~ n_feature, 
    labeller = label_glue(
        rows = "{n_option} Options",
        cols = "{n_feature} Features"
    )
)

chance_line = geom_hline(aes(yintercept = 1/n_option), lty="dotted")

random_payoff = 150
maximum_payoff = 183.63861
payoff_line_lims = list(
  geom_hline(yintercept=c(maximum_payoff), linetype="dashed"),
  coord_cartesian(xlim=c(NULL), ylim=c(random_payoff, maximum_payoff))
)

