#' A \code{data.frame} to illustrate \code{\link{bubble_for_rank}}.
"df_rank"

#' Make bubble plot for ranks
#' @param data A \code{data.frame} in long format.
#' @param x_name Variable name in \code{data} to be plotted on x-axis (e.g.,
#'   method names).
#' @param y_name Variable name in \code{data} to be plotted on y-axis (e.g.,
#'   variable names).
#' @param rank_name Variable name in \code{data} to indicate rank of \code{x}.
#' @param highlight_name Variable name in \code{data} to indicate which bubbles
#'   to highlight.
#' @param arrange_rows_by A level in \code{x_name}. Rows will be arranged by
#'   rank for this method. Default is \code{NULL}, i.e., plot as-is.
#' @param x_order Ordering of columns. Default is \code{NULL}, i.e., based on
#'   levels in \code{x}.
#' @param x_labels Labels to print on x-axis. Default is \code{NULL}, i.e., use
#'   levels in \code{x}.
#' @param vline_at x-position(s) to draw vertical lines at, e.g., 1.5 to draw
#'   between 1st and 2nd columns. Default is \code{NULL}, i.e., no line.
#' @return Returns a \code{ggplot2} object to allow further modification.
#' @export
#' @import ggplot2
#' @import ggpubr
#' @import dplyr
#' @import tidyr
#' @example tests/test_bubbles.R
bubble_for_rank <- function(data, x_name, y_name, rank_name, highlight_name,
                            arrange_rows_by = NULL, x_order = NULL,
                            x_labels = NULL, vline_at = NULL) {
  df <- data.frame(x = data[, x_name], y = data[, y_name],
                   rank = data[, rank_name],
                   highlight = data[, highlight_name],
                   stringsAsFactors = TRUE) %>%
    arrange(x) %>%
    group_by(x) %>% mutate(rank_int = rank(rank)) %>% ungroup()
  df_mat <- df %>% select(-highlight, -rank) %>%
    pivot_wider(id_cols = y, names_from = x, values_from = rank_int) %>%
    as.data.frame(check.names = FALSE)
  if (!is.null(arrange_rows_by)) {
    df_mat <- df_mat[sort.list(df_mat[, arrange_rows_by]), ]
  }
  rownames(df_mat) <- df_mat$y
  df_mat <- df_mat %>% select(-y)
  if (!is.null(x_order)) {
    df_mat <- df_mat[, x_order]
    if (is.null(x_labels)) x_labels <- x_order
  } else {
    if (is.null(x_labels)) x_labels <- levels(df$x)
  }

  df_font <- df %>%
    mutate(y = as.numeric(factor(y, levels = rev(rownames(df_mat)))),
           x = as.numeric(factor(x, levels = colnames(df_mat)))) %>%
    select(x, y, highlight, rank_int)

  p <- df_mat %>%
    ggpubr::ggballoonplot(fill = "value", color = "lightgray",
                          size = 10, show.label = FALSE) +
    ggpubr::gradient_fill(c("darkorange", "white", "steelblue")) +
    geom_text(data = df_font, aes(x = x, y = y, label = rank_int),
              size = ifelse(df_font$highlight, 5.5, 5),
              fontface = ifelse(df_font$highlight, "bold.italic", "plain")) +
    scale_x_discrete(labels = x_labels) +
    theme(text = element_text(size = 16),
          axis.text.x = element_text(angle = 0, hjust = 0.5))
  if (!is.null(vline_at)) p <- p + geom_vline(xintercept = vline_at, color = "grey90")
  p
}
