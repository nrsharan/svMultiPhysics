// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XDMF_WRITER_H
#define XDMF_WRITER_H

#include <string>
#include <vector>

class vtkUnstructuredGrid;

/// @brief Writes the results of the saved time steps to one XDMF file with
/// its data in one HDF5 file, instead of one VTU file per saved time step
/// (GeneralSimulationParameters 'Save_results_in_XDMF_format').
///
/// The HDF5 file '<prefix>.h5' holds
///  - /Mesh/<m>/Geometry (number of points x 3, double) and
///    /Mesh/<m>/Topology: a mesh, written again only when its points or cells
///    change (e.g. after remeshing);
///  - /Step/<n>/PointData/<array> and /Step/<n>/CellData/<array>: the results
///    of time step n, with its time ('Time') and mesh ('Mesh') as attributes
///    of /Step/<n>.
///
/// The XDMF file '<prefix>.xdmf', the one to open in ParaView, lists the
/// steps of the HDF5 file as a temporal collection. It is rewritten from the
/// HDF5 file after every step, so the two always agree, also after a
/// simulation is continued. Cell data that are only written for the first
/// saved step (Domain_ID, Proc_ID) are listed for all later steps on the same
/// mesh.
///
/// The data are those of the VTU files: the same point and cell arrays with
/// the same values, the cells with the VTK node order, and the time.
class XdmfWriter {
  public:
    /// \param prefix          the file names without extension
    /// \param continued_from  for a continued simulation, the time step it
    ///   continues from: the steps after it are removed from an existing HDF5
    ///   file and the others kept; -1 starts new files.
    XdmfWriter(const std::string& prefix, int continued_from);

    /// @brief Add the results of time step 'step' at time 'time' to the files
    /// (master process only).
    ///
    /// Steps from 'step' on that are already in the file (from a simulation
    /// continued from an earlier step, or before remeshing) are replaced.
    void write(vtkUnstructuredGrid& results, int step, double time);

    /// A mesh as it is written to the HDF5 file.
    struct Mesh {
      std::vector<double> points;       ///< 3 coordinates per point
      std::vector<long long> topology;  ///< the XDMF topology array
      std::string type;                 ///< XDMF TopologyType
      long long num_cells = 0;
      int nodes_per_cell = 0;           ///< 0 for 'Mixed'

      bool operator==(const Mesh& other) const;
    };

  private:
    std::string h5_path_;
    std::string xdmf_path_;
    std::string h5_name_;  ///< the HDF5 file name as the XDMF file refers to it
    int continued_from_;
    bool started_ = false;

    /// The mesh last written, as /Mesh/<mesh_id_>; -1 if none.
    int mesh_id_ = -1;
    Mesh mesh_;
};

#endif
