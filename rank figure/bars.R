#' Draw horizontal bar plot, possibly with error bars.
#' @param val A numeric vector of model reliance.
#' @param val_sd A numeric vector of standard deviations of model reliance (with
#'   the same length as \code{val}), if available. Alternatively, provide
#'   \code{val_lower} and \code{val_upper} instead of \code{val_sd} to reflect
#'   uncertainty in \code{val}.
#' @param val_lower A numeric vector of the lower bound of 95\% confidence
#'   interval of model reliance (with the same length as \code{val}), if
#'   available.
#' @param val_upper A numeric vector of the upper bound of 95\% confidence
#'   interval of model reliance (with the same length as \code{val}), if
#'   available.
#' @param var_names A factor or string vector of variable names (with the same
#'   length as \code{val}). If a factor is provided, variables will be plotted
#'   as ordered in \code{levels(var_names)}, otherwise variables will be ordered
#'   by \code{val}. If unspecified, variables will be named 'X1', 'X2', etc.
#' @param x_lab Title of x-axis (optional).
#' @param color Color for bars with positive and negative values. Default is
#'   steel blue for positive and grey for negative.
#' @param title Title of the bar plot (optional).
#' @param subtitle Subtitle of the bar plot (optional).
#' @return Returns a ggplot object for the bar plot.
#' @export
#' @import ggplot2
#' @examples tests/test_bars.R
draw_bars <- function(val, val_sd = NULL, val_lower = NULL, val_upper = NULL,
                      var_names = NULL, x_lab = "",
                      color = list(pos = "steelblue", neg = "grey"),
                      title = NULL, subtitle = NULL) {
  if (is.null(var_names)) {
    var_names <- paste0("X", seq_along(val))
  }
  if ("factor" %in% class(var_names)) {
    var_names_ordered <- var_names
  } else {
    var_names_ordered <- factor(var_names, levels = var_names[order(val)])
  }
  dt <- data.frame(val = val, variables = var_names_ordered)
  if (is.null(val_lower) | is.null(val_upper)) {
    if (is.null(val_sd)) {
      draw_errors <- FALSE
      dt$val_lower <- val
      dt$val_upper <- val
    } else {
      draw_errors <- TRUE
      dt$val_lower <- val - 1.96 * val_sd
      dt$val_upper <- val + 1.96 * val_sd
    }
  } else {
    draw_errors <- TRUE
    dt$val_lower <- val_lower
    dt$val_upper <- val_upper
  }
  if (draw_errors) {
    # dt$val_lower should have been supplied or computed
    # Putting 1 (i.e., +ve sign) as reference level, so that steelblue will be
    # used when all are +ve
    dt$sign_vec <- factor(as.numeric(dt$val_lower > 0), levels = c(1, 0))
  } else {
    # dt$sign_vec <- factor(rep(1, nrow(dt)), levels = c(1, 0))
    dt$sign_vec <- factor(as.numeric(dt$val_lower > 0), levels = c(1, 0))
  }
  common_theme <- theme(panel.grid.major.y = element_line(colour = "grey95"),
                        panel.grid.minor = element_blank(),
                        panel.background = element_blank(),
                        axis.line.x = element_line(colour = "black"),
                        axis.ticks.y = element_blank())
  # x_lab <- "Average model reliance (>0 suggests importance)\nwith 95% prediction interval"
  p <- ggplot(data = dt,
              mapping = aes_string(x = "variables", y = "val", fill = "sign_vec")) +
    geom_hline(yintercept = 0, color = "grey") +
    geom_bar(stat = "identity") +
    common_theme +
    coord_flip() +
    labs(x = "", y = x_lab, title = title, subtitle = subtitle) +
    theme(legend.position = "none") +
    scale_fill_manual(values = c(color$pos, color$neg))
  # if (!is.null(labels)) {
  #   dt$labels = labels
  #   p <- p +
  #     geom_text(data = dt,
  #               aes_string(x = "variables", y = 0, label = "labels"),
  #               nudge_x = 0.2, hjust = 0)
  # }
  if (draw_errors) {
    p <- p +
      geom_errorbar(aes_string(ymin = "val_lower", ymax = "val_upper"),
                    width = 0.2)
  }
  p
}
