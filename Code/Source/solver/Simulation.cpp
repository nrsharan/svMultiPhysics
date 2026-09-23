// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "Simulation.h"
#include "time_segments.h"
#include "Integrator.h"

#include "all_fun.h"
#include "load_msh.h"

#include "mpi.h"

#include <iostream>
#include <stdexcept>

Simulation::Simulation() 
{
  roInf = 0.2;
  com_mod.cm.new_cm(MPI_COMM_WORLD);

  history_file_name = "histor.dat";
}

Simulation::~Simulation() 
{
}

const mshType& Simulation::get_msh(const std::string& name)
{
  for (auto& mesh : com_mod.msh) { 
    if (mesh.name == name) {
      return mesh;
    }
  }
}

/// @brief Read solver parameters.
//
void Simulation::read_parameters(const std::string& file_name)
{
  parameters.read_xml(file_name);
}

/// @brief Set the simulation and module member data.
///
/// Replicates the README subroutine lines to set COMMOD module varliables
///
///   lPtr => list%get(nTs,"Number of time steps",1,ll=1)
//
void Simulation::set_module_parameters()
{
  // Set ComMod module varliables.
  //
  auto& general = parameters.general_simulation_parameters;

  com_mod.iniFilePath = general.simulation_initialization_file_path.value();
  com_mod.nsd = general.number_of_spatial_dimensions.value();
  com_mod.nsymd = 3*(com_mod.nsd-1);

  com_mod.nITs = general.number_of_initialization_time_steps.value();
  com_mod.startTS = general.starting_time_step.value();

  // Time stepping: a fixed time step size and number of time steps, or time
  // step segments that run up to a final time.
  const auto& segments = general.time_step_segments;

  if (segments.empty()) {
    if (!general.number_of_time_steps.defined() || !general.time_step_size.defined()) {
      throw std::runtime_error("[Simulation] <GeneralSimulationParameters> requires <Number_of_time_steps> "
          "and <Time_step_size>, or <Add_time_step_segment> elements with a <Final_time>.");
    }
    if (general.final_time.defined()) {
      throw std::runtime_error("[Simulation] <Final_time> is only used with <Add_time_step_segment> elements.");
    }
    com_mod.nTS = general.number_of_time_steps.value();
    com_mod.dt = general.time_step_size.value();

  } else {
    if (general.number_of_time_steps.defined() || general.time_step_size.defined()) {
      throw std::runtime_error("[Simulation] With <Add_time_step_segment> elements, the time step size and "
          "number of time steps follow from the segments and <Final_time>; remove <Number_of_time_steps> "
          "and <Time_step_size>.");
    }
    if (!general.final_time.defined()) {
      throw std::runtime_error("[Simulation] <Add_time_step_segment> elements require a <Final_time>.");
    }
    if (com_mod.nITs > 0) {
      throw std::runtime_error("[Simulation] <Number_of_initialization_time_steps> cannot be combined with "
          "<Add_time_step_segment> elements.");
    }
    if (!time_segments::approx_equal(segments[0][0], 0.0)) {
      throw std::runtime_error("[Simulation] The first <Add_time_step_segment> must start at time 0.");
    }
    for (int i = 0; i < static_cast<int>(segments.size()); i++) {
      if (segments[i][1] <= 0.0) {
        throw std::runtime_error("[Simulation] The <Time_step_size> of every <Add_time_step_segment> must be positive.");
      }
      if (i > 0 && segments[i][0] <= segments[i-1][0]) {
        throw std::runtime_error("[Simulation] The <Start_time> values of <Add_time_step_segment> elements must increase.");
      }
    }
    if (general.final_time.value() <= segments.back()[0]) {
      throw std::runtime_error("[Simulation] <Final_time> must be later than the start of the last <Add_time_step_segment>.");
    }

    com_mod.dtSegments = segments;
    com_mod.finalTime = general.final_time.value();
    com_mod.dt = time_segments::next_time_step(com_mod.dtSegments, com_mod.finalTime, 0.0);
    com_mod.nTS = time_segments::number_of_time_steps(com_mod.dtSegments, com_mod.finalTime);
  }

  // Adaptive time stepping: the time step size given for a segment is the
  // largest step that segment may take, and a time step that fails -- an
  // element that cannot compute its state, a linear solver that breaks down,
  // or a Newton iteration that reaches <Max_iterations> without converging --
  // is repeated from the same state with a smaller step (see main.cpp's
  // iterate_solution()). The run then ends
  // at <Final_time> rather than after nTS time steps, and nTS is only an
  // estimate: every repeated and every shortened step adds to it.
  com_mod.adaptiveDt = general.adaptive_time_stepping.value();

  if (com_mod.adaptiveDt) {
    if (segments.empty()) {
      throw std::runtime_error("[Simulation] <Adaptive_time_stepping> requires <Add_time_step_segment> "
          "elements: the <Time_step_size> given for a segment is the largest step it may take.");
    }

    com_mod.adaptiveDtCutFactor = general.time_step_reduction_factor.value();
    com_mod.adaptiveDtGrowFactor = general.time_step_increase_factor.value();
    com_mod.adaptiveDtGrowAfter = general.converged_time_steps_before_increase.value();
    com_mod.adaptiveDtDivergenceFactor = general.newton_divergence_factor.value();

    if (com_mod.adaptiveDtDivergenceFactor < 0.0 ||
        (com_mod.adaptiveDtDivergenceFactor > 0.0 && com_mod.adaptiveDtDivergenceFactor <= 1.0)) {
      throw std::runtime_error("[Simulation] <Newton_divergence_factor> must be larger than 1, or 0 to "
          "leave a diverging time step to <Max_iterations>.");
    }

    if (com_mod.adaptiveDtCutFactor <= 0.0 || com_mod.adaptiveDtCutFactor >= 1.0) {
      throw std::runtime_error("[Simulation] <Time_step_reduction_factor> must be larger than 0 and smaller than 1.");
    }
    if (com_mod.adaptiveDtGrowFactor < 1.0) {
      throw std::runtime_error("[Simulation] <Time_step_increase_factor> must be at least 1.");
    }

    double smallest = segments[0][1];
    for (const auto& segment : segments) {
      if (segment[1] < smallest) {
        smallest = segment[1];
      }
    }

    com_mod.adaptiveDtMin = general.minimum_time_step_size.defined()
        ? general.minimum_time_step_size.value() : 1.0e-3 * smallest;

    if (com_mod.adaptiveDtMin <= 0.0) {
      throw std::runtime_error("[Simulation] <Minimum_time_step_size> must be positive.");
    }
    if (com_mod.adaptiveDtMin > smallest) {
      throw std::runtime_error("[Simulation] <Minimum_time_step_size> is larger than the smallest "
          "<Time_step_size> of the <Add_time_step_segment> elements.");
    }

    // The first time step takes the first segment's size.
    com_mod.adaptiveDtLimit = segments[0][1];
  }

  com_mod.stopTrigName = general.searched_file_name_to_trigger_stop.value();
  com_mod.ichckIEN = general.check_ien_order.value();
  com_mod.saveVTK = general.save_results_to_vtk_format.value();
  com_mod.saveXDMF = general.save_results_in_xdmf_format.value();
  com_mod.saveName = general.name_prefix_of_saved_vtk_files.value();
  com_mod.saveName = chnl_mod.appPath + com_mod.saveName;
  com_mod.saveIncr = general.increment_in_saving_vtk_files.value();

  // Writing the results every so much simulated time rather than every so
  // many time steps. The first time step is written, and from then on the
  // first step that ends at or after each multiple of the interval.
  com_mod.saveTimeIncr = general.save_results_every_time.value();
  com_mod.nextSaveTime = 0.0;

  if (com_mod.saveTimeIncr < 0.0) {
    throw std::runtime_error("[Simulation] <Save_results_every_time> cannot be negative; 0 writes the "
        "results every <Increment_in_saving_VTK_files> time steps instead.");
  }
  com_mod.saveATS = general.start_saving_after_time_step.value();
  com_mod.saveAve = general.save_averaged_results.value();
  com_mod.alwaysSaveDomainID = general.save_domain_id_in_every_file.value();
  com_mod.zeroAve = general.start_averaging_from_zero.value();
  com_mod.stFileRepl = general.overwrite_restart_file.value();
  com_mod.stFileName = chnl_mod.appPath + general.restart_file_name.value();
  com_mod.stFileIncr = general.increment_in_saving_restart_files.value();
  com_mod.rmsh.isReqd = general.simulation_requires_remeshing.value();

  auto& precomp_sol = parameters.precomputed_solution_parameters;
  com_mod.usePrecomp = precomp_sol.use_precomputed_solution.value();
  com_mod.precompFileName = precomp_sol.file_path.value();
  com_mod.precompFieldName = precomp_sol.field_name.value();
  com_mod.precompDt = precomp_sol.time_step.value();

  if ((com_mod.precompDt == 0.0) && (com_mod.usePrecomp)) {
    std::cout << "Precomputed time step size is zero. Setting to simulation time step size." << std::endl;
    com_mod.precompDt = com_mod.dt;
  }
  // Set simulation parameters.
  nTs = general.number_of_time_steps.value();
  fTmp = general.simulation_initialization_file_path.value();
  roInf = general.spectral_radius_of_infinite_time_step.value();
}

/// @brief Initialize the Integrator object after simulation setup is complete
///
/// This should be called at the end of initialize() after solution states have been
/// fully initialized. The Integrator takes ownership of these solution states.
///
/// @param solutions Solution states containing old acceleration, displacement, and velocity
void Simulation::initialize_integrator(SolutionStates&& solutions)
{
  integrator_ = std::make_unique<Integrator>(this, std::move(solutions));
}

/// @brief Get reference to the Integrator object
///
/// @return Reference to the Integrator
Integrator& Simulation::get_integrator()
{
  if (!integrator_) {
    throw std::runtime_error("Integrator not initialized. Call initialize_integrator() first.");
  }
  return *integrator_;
}
