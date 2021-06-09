library(tidyverse)
library(magrittr)
library(stickylabeller)
library(lemon)

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

theme_set(theme_classic(base_size = 14) + theme(strip.background = element_blank()))
update_geom_defaults("line", list(size = 1.2))

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
    # invisible(dev.off())
    # knitr::include_graphics(p)
}


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
  if (p < .001) "p < .001" else glue("p = {str_sub(format(round(p, 3)), 2)}")
}

write_tex = function(tex, file) {
  print(fmt("{file}: {tex}"))
  writeLines("{tex}\\unskip", file)
}

only = function(xs) {
  u = unique(xs)
  stopifnot(length(u) == 1)
  u[1]
}

# %% ==================== Plot utils ====================

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


