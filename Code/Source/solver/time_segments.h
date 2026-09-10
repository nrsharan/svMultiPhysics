// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// Time segments: piecewise-constant time step sizes
// (GeneralSimulationParameters <Add_time_step_segment>) and on/off time
// intervals (<Time_segments> in <CCBActiveCMMGandR>). The rules follow
// FEDDLib's "Timestepping Intervalls".

#ifndef TIME_SEGMENTS_H
#define TIME_SEGMENTS_H

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace time_segments {

/// @brief Whether two times are equal up to the tolerance used for segment
/// boundaries.
inline bool approx_equal(const double a, const double b)
{
  return std::fabs(a - b) <= 1.0e-8 * std::max(1.0, std::fabs(b));
}

/// @brief Time step size for the time step that starts at time t_n.
///
/// The active segment is the last one whose start time is not after t_n.
/// Its time step size is used, shortened if needed so that the step ends
/// exactly at the start of the next segment, or at final_time for the last
/// segment.
inline double next_time_step(const std::vector<std::array<double,2>>& segments,
                             const double final_time, const double t_n)
{
  int active = 0;
  for (int i = 0; i < static_cast<int>(segments.size()); i++) {
    if (t_n > segments[i][0] || approx_equal(t_n, segments[i][0])) {
      active = i;
    }
  }

  const bool last = active + 1 == static_cast<int>(segments.size());
  const double end = last ? final_time : segments[active + 1][0];
  double dt = segments[active][1];

  if (t_n + dt > end && !approx_equal(t_n + dt, end)) {
    dt = end - t_n;
  }

  return dt;
}

/// @brief Number of time steps from t_start to final_time.
inline int number_of_time_steps(const std::vector<std::array<double,2>>& segments,
                                const double final_time, const double t_start = 0.0)
{
  double t = t_start;
  int n = 0;

  while (t < final_time && !approx_equal(t, final_time)) {
    t += next_time_step(segments, final_time, t);
    n += 1;

    if (n > 100000000) {
      throw std::runtime_error("[time_segments] More than 1e8 time steps; check the time step segments.");
    }
  }

  return n;
}

/// @brief Whether time t lies in one of the [start, end) intervals.
inline bool in_intervals(const std::vector<std::array<double,2>>& intervals, const double t)
{
  for (const auto& interval : intervals) {
    const bool after_start = t > interval[0] || approx_equal(t, interval[0]);
    const bool before_end = t < interval[1] && !approx_equal(t, interval[1]);

    if (after_start && before_end) {
      return true;
    }
  }

  return false;
}

}

#endif
