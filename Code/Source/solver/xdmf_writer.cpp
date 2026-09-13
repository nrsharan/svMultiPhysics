// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "xdmf_writer.h"

#ifdef SV_HDF5_FROM_VTK
#include <vtk_hdf5.h>
#else
#include <hdf5.h>
#endif

#include <vtkCellData.h>
#include <vtkCellType.h>
#include <vtkDataArray.h>
#include <vtkDataSetAttributes.h>
#include <vtkIdList.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>

namespace {

[[noreturn]] void hdf5_error(const std::string& what)
{
  throw std::runtime_error("Error writing the XDMF/HDF5 results: " + what + ".");
}

void check(herr_t status, const std::string& what)
{
  if (status < 0) {
    hdf5_error(what);
  }
}

/// An HDF5 identifier, closed when it goes out of scope.
class Handle {
  public:
    Handle(hid_t id, herr_t (*close)(hid_t), const std::string& what) : id_(id), close_(close)
    {
      if (id_ < 0) {
        hdf5_error(what);
      }
    }
    ~Handle() { close_(id_); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    operator hid_t() const { return id_; }

  private:
    hid_t id_;
    herr_t (*close_)(hid_t);
};

Handle open_group(hid_t location, const std::string& name)
{
  return Handle(H5Gopen2(location, name.c_str(), H5P_DEFAULT), H5Gclose, "opening the group " + name);
}

Handle create_group(hid_t location, const std::string& name)
{
  return Handle(H5Gcreate2(location, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose,
      "creating the group " + name);
}

bool exists(hid_t location, const std::string& name)
{
  const htri_t status = H5Lexists(location, name.c_str(), H5P_DEFAULT);
  check(status, "looking for " + name);
  return status > 0;
}

/// The names of the links in a group, in name order.
std::vector<std::string> children(hid_t group)
{
  H5G_info_t info;
  check(H5Gget_info(group, &info), "reading a group");
  std::vector<std::string> names;

  for (hsize_t i = 0; i < info.nlinks; i++) {
    const ssize_t size = H5Lget_name_by_idx(group, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
    check(size < 0 ? -1 : 0, "reading a group");
    std::string name(size + 1, '\0');
    H5Lget_name_by_idx(group, ".", H5_INDEX_NAME, H5_ITER_INC, i, &name[0], size + 1, H5P_DEFAULT);
    name.resize(size);
    names.push_back(name);
  }

  return names;
}

/// The time steps in the /Step group, in increasing order.
std::vector<int> steps(hid_t file)
{
  std::vector<int> numbers;
  for (const auto& name : children(open_group(file, "/Step"))) {
    numbers.push_back(std::stoi(name));
  }
  std::sort(numbers.begin(), numbers.end());
  return numbers;
}

void remove_steps(hid_t file, const std::function<bool(int)>& remove)
{
  for (int step : steps(file)) {
    if (remove(step)) {
      const std::string name = "/Step/" + std::to_string(step);
      check(H5Ldelete(file, name.c_str(), H5P_DEFAULT), "removing " + name);
    }
  }
}

void write_attribute(hid_t object, const std::string& name, hid_t type, const void* value)
{
  Handle space(H5Screate(H5S_SCALAR), H5Sclose, "creating the attribute " + name);
  Handle attribute(H5Acreate2(object, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose,
      "creating the attribute " + name);
  check(H5Awrite(attribute, type, value), "writing the attribute " + name);
}

void write_attribute(hid_t object, const std::string& name, const std::string& value)
{
  Handle type(H5Tcopy(H5T_C_S1), H5Tclose, "creating the attribute " + name);
  check(H5Tset_size(type, value.size() + 1), "creating the attribute " + name);
  write_attribute(object, name, type, value.c_str());
}

void read_attribute(hid_t object, const std::string& name, hid_t type, void* value)
{
  Handle attribute(H5Aopen(object, name.c_str(), H5P_DEFAULT), H5Aclose, "opening the attribute " + name);
  check(H5Aread(attribute, type, value), "reading the attribute " + name);
}

std::string read_string_attribute(hid_t object, const std::string& name)
{
  Handle attribute(H5Aopen(object, name.c_str(), H5P_DEFAULT), H5Aclose, "opening the attribute " + name);
  Handle file_type(H5Aget_type(attribute), H5Tclose, "reading the attribute " + name);
  const size_t size = H5Tget_size(file_type);
  Handle type(H5Tcopy(H5T_C_S1), H5Tclose, "reading the attribute " + name);
  check(H5Tset_size(type, size), "reading the attribute " + name);
  std::string value(size, '\0');
  check(H5Aread(attribute, type, &value[0]), "reading the attribute " + name);
  return value.c_str();
}

/// A dataset of dimensions 'dims' (row-major), with the attribute 'Name' if
/// 'original_name' is not empty.
void write_dataset(hid_t group, const std::string& name, hid_t memory_type, hid_t file_type,
    const std::vector<hsize_t>& dims, const void* data, const std::string& original_name = "")
{
  Handle space(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr), H5Sclose,
      "creating the dataset " + name);
  Handle dataset(H5Dcreate2(group, name.c_str(), file_type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
      H5Dclose, "creating the dataset " + name);
  check(H5Dwrite(dataset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), "writing the dataset " + name);

  if (!original_name.empty()) {
    write_attribute(dataset, "Name", original_name);
  }
}

template <class T>
std::vector<T> read_dataset(hid_t group, const std::string& name, hid_t memory_type)
{
  Handle dataset(H5Dopen2(group, name.c_str(), H5P_DEFAULT), H5Dclose, "opening the dataset " + name);
  Handle space(H5Dget_space(dataset), H5Sclose, "reading the dataset " + name);
  std::vector<T> values(H5Sget_simple_extent_npoints(space));
  check(H5Dread(dataset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()),
      "reading the dataset " + name);
  return values;
}

/// The XDMF topology of a VTK cell type. XDMF uses the VTK node order for
/// all of them.
struct CellType {
  int vtk;
  const char* xdmf;
  int code;   // in a Mixed topology
  int nodes;
};

const CellType& cell_type(int vtk_type)
{
  static const CellType types[] = {
    {VTK_LINE, "Polyline", 2, 2},
    {VTK_TRIANGLE, "Triangle", 4, 3},
    {VTK_QUAD, "Quadrilateral", 5, 4},
    {VTK_TETRA, "Tetrahedron", 6, 4},
    {VTK_PYRAMID, "Pyramid", 7, 5},
    {VTK_WEDGE, "Wedge", 8, 6},
    {VTK_HEXAHEDRON, "Hexahedron", 9, 8},
    {VTK_QUADRATIC_EDGE, "Edge_3", 34, 3},
    {VTK_QUADRATIC_TRIANGLE, "Triangle_6", 36, 6},
    {VTK_QUADRATIC_QUAD, "Quadrilateral_8", 37, 8},
    {VTK_BIQUADRATIC_QUAD, "Quadrilateral_9", 35, 9},
    {VTK_QUADRATIC_TETRA, "Tetrahedron_10", 38, 10},
    {VTK_QUADRATIC_WEDGE, "Wedge_15", 40, 15},
    {VTK_QUADRATIC_HEXAHEDRON, "Hexahedron_20", 48, 20},
    {VTK_TRIQUADRATIC_HEXAHEDRON, "Hexahedron_27", 50, 27},
  };

  for (const auto& type : types) {
    if (type.vtk == vtk_type) {
      return type;
    }
  }

  hdf5_error("the VTK cell type " + std::to_string(vtk_type) + " has no XDMF topology");
}

XdmfWriter::Mesh extract_mesh(vtkUnstructuredGrid& grid)
{
  XdmfWriter::Mesh mesh;
  const vtkIdType num_points = grid.GetNumberOfPoints();
  mesh.points.resize(3 * num_points);

  for (vtkIdType i = 0; i < num_points; i++) {
    grid.GetPoint(i, &mesh.points[3 * i]);
  }

  const vtkIdType num_cells = grid.GetNumberOfCells();
  if (num_cells == 0) {
    hdf5_error("the results have no cells");
  }
  mesh.num_cells = num_cells;

  bool uniform = true;
  for (vtkIdType c = 1; c < num_cells && uniform; c++) {
    uniform = grid.GetCellType(c) == grid.GetCellType(0);
  }

  vtkNew<vtkIdList> nodes;

  if (uniform) {
    const CellType& type = cell_type(grid.GetCellType(0));
    mesh.type = type.xdmf;
    mesh.nodes_per_cell = type.nodes;
    mesh.topology.reserve(num_cells * type.nodes);
  } else {
    mesh.type = "Mixed";
  }

  for (vtkIdType c = 0; c < num_cells; c++) {
    const CellType& type = cell_type(grid.GetCellType(c));
    grid.GetCellPoints(c, nodes);

    if (nodes->GetNumberOfIds() != type.nodes) {
      hdf5_error("a cell of type " + std::string(type.xdmf) + " has " + std::to_string(nodes->GetNumberOfIds()) +
          " nodes");
    }

    if (!uniform) {
      mesh.topology.push_back(type.code);
      if (type.code == 2) {
        mesh.topology.push_back(type.nodes);
      }
    }

    for (vtkIdType i = 0; i < nodes->GetNumberOfIds(); i++) {
      mesh.topology.push_back(nodes->GetId(i));
    }
  }

  return mesh;
}

void write_mesh(hid_t file, int id, const XdmfWriter::Mesh& mesh)
{
  Handle group = create_group(file, "/Mesh/" + std::to_string(id));
  write_attribute(group, "TopologyType", mesh.type);
  write_attribute(group, "NumberOfElements", H5T_NATIVE_LLONG, &mesh.num_cells);
  write_attribute(group, "NodesPerElement", H5T_NATIVE_INT, &mesh.nodes_per_cell);

  write_dataset(group, "Geometry", H5T_NATIVE_DOUBLE, H5T_IEEE_F64LE, {mesh.points.size() / 3, 3},
      mesh.points.data());

  std::vector<hsize_t> dims{mesh.topology.size()};
  if (mesh.nodes_per_cell > 0) {
    dims = {static_cast<hsize_t>(mesh.num_cells), static_cast<hsize_t>(mesh.nodes_per_cell)};
  }
  write_dataset(group, "Topology", H5T_NATIVE_LLONG, H5T_STD_I64LE, dims, mesh.topology.data());
}

XdmfWriter::Mesh read_mesh(hid_t file, int id)
{
  Handle group = open_group(file, "/Mesh/" + std::to_string(id));
  XdmfWriter::Mesh mesh;
  mesh.type = read_string_attribute(group, "TopologyType");
  read_attribute(group, "NumberOfElements", H5T_NATIVE_LLONG, &mesh.num_cells);
  read_attribute(group, "NodesPerElement", H5T_NATIVE_INT, &mesh.nodes_per_cell);
  mesh.points = read_dataset<double>(group, "Geometry", H5T_NATIVE_DOUBLE);
  mesh.topology = read_dataset<long long>(group, "Topology", H5T_NATIVE_LLONG);
  return mesh;
}

bool is_integer(int vtk_data_type)
{
  switch (vtk_data_type) {
    case VTK_CHAR: case VTK_SIGNED_CHAR: case VTK_UNSIGNED_CHAR: case VTK_SHORT: case VTK_UNSIGNED_SHORT:
    case VTK_INT: case VTK_UNSIGNED_INT: case VTK_LONG: case VTK_UNSIGNED_LONG: case VTK_LONG_LONG:
    case VTK_UNSIGNED_LONG_LONG: case VTK_ID_TYPE:
      return true;
  }
  return false;
}

/// The arrays of 'data' as the datasets of the new group 'name', one per
/// array: integer arrays as 32-bit (VTK_INT and smaller) or 64-bit integers,
/// all others as doubles.
void write_arrays(hid_t step, const std::string& name, vtkDataSetAttributes& data)
{
  Handle group = create_group(step, name);

  for (int k = 0; k < data.GetNumberOfArrays(); k++) {
    vtkDataArray* array = data.GetArray(k);
    if (array == nullptr || array->GetName() == nullptr) {
      continue;
    }

    const std::string array_name = array->GetName();
    std::string link = array_name.empty() ? "unnamed" : array_name;
    std::replace(link.begin(), link.end(), '/', '_');
    if (link == ".") {
      link = "_";
    }

    const hsize_t tuples = array->GetNumberOfTuples();
    const int components = array->GetNumberOfComponents();
    std::vector<hsize_t> dims{tuples};
    if (components > 1) {
      dims.push_back(components);
    }

    const int type = array->GetDataType();

    if (is_integer(type)) {
      const bool small = type == VTK_INT || type == VTK_SHORT || type == VTK_UNSIGNED_SHORT || type == VTK_CHAR ||
                         type == VTK_SIGNED_CHAR || type == VTK_UNSIGNED_CHAR;
      std::vector<long long> values(tuples * components);
      for (hsize_t i = 0; i < tuples; i++) {
        for (int j = 0; j < components; j++) {
          values[i * components + j] = std::llround(array->GetComponent(i, j));
        }
      }
      write_dataset(group, link, H5T_NATIVE_LLONG, small ? H5T_STD_I32LE : H5T_STD_I64LE, dims, values.data(),
          array_name);
    } else {
      std::vector<double> values(tuples * components);
      for (hsize_t i = 0; i < tuples; i++) {
        for (int j = 0; j < components; j++) {
          values[i * components + j] = array->GetComponent(i, j);
        }
      }
      write_dataset(group, link, H5T_NATIVE_DOUBLE, H5T_IEEE_F64LE, dims, values.data(), array_name);
    }
  }
}

std::string xml_escape(const std::string& text)
{
  std::string escaped;
  for (char c : text) {
    switch (c) {
      case '&': escaped += "&amp;"; break;
      case '<': escaped += "&lt;"; break;
      case '>': escaped += "&gt;"; break;
      case '"': escaped += "&quot;"; break;
      default: escaped += c;
    }
  }
  return escaped;
}

/// A dataset as an XDMF DataItem.
struct DataItem {
  std::string path;
  std::vector<hsize_t> dims;
  bool integer = false;
  size_t precision = 8;
};

DataItem data_item(hid_t file, const std::string& path)
{
  Handle dataset(H5Dopen2(file, path.c_str(), H5P_DEFAULT), H5Dclose, "opening the dataset " + path);
  Handle space(H5Dget_space(dataset), H5Sclose, "reading the dataset " + path);
  Handle type(H5Dget_type(dataset), H5Tclose, "reading the dataset " + path);

  DataItem item;
  item.path = path;
  item.dims.resize(H5Sget_simple_extent_ndims(space));
  H5Sget_simple_extent_dims(space, item.dims.data(), nullptr);
  item.integer = H5Tget_class(type) == H5T_INTEGER;
  item.precision = H5Tget_size(type);
  return item;
}

void write_data_item(std::ostream& xml, const std::string& h5_name, const DataItem& item, const std::string& indent)
{
  xml << indent << "<DataItem Dimensions=\"";
  for (size_t i = 0; i < item.dims.size(); i++) {
    xml << (i > 0 ? " " : "") << item.dims[i];
  }
  xml << "\" NumberType=\"" << (item.integer ? "Int" : "Float") << "\" Precision=\"" << item.precision
      << "\" Format=\"HDF\">" << xml_escape(h5_name) << ":" << xml_escape(item.path) << "</DataItem>\n";
}

/// An array of a step, as an XDMF Attribute.
struct Attribute {
  std::string name;
  std::string center;
  DataItem item;
};

std::vector<Attribute> attributes(hid_t file, const std::string& group_path, const std::string& center)
{
  std::vector<Attribute> list;
  if (!exists(file, group_path)) {
    return list;
  }
  for (const auto& link : children(open_group(file, group_path))) {
    const std::string path = group_path + "/" + link;
    Handle dataset(H5Dopen2(file, path.c_str(), H5P_DEFAULT), H5Dclose, "opening the dataset " + path);
    list.push_back({read_string_attribute(dataset, "Name"), center, data_item(file, path)});
  }
  return list;
}

void write_attribute_xml(std::ostream& xml, const std::string& h5_name, const Attribute& attribute)
{
  // Arrays other than scalars and 3-vectors (e.g. the 6 stress components)
  // are Matrix attributes, so that their components are read as they are
  // stored rather than reordered as an XDMF tensor.
  const auto& dims = attribute.item.dims;
  const char* type = dims.size() == 1 ? "Scalar" : dims[1] == 3 ? "Vector" : "Matrix";
  xml << "        <Attribute Name=\"" << xml_escape(attribute.name) << "\" AttributeType=\"" << type
      << "\" Center=\"" << attribute.center << "\">\n";
  write_data_item(xml, h5_name, attribute.item, "          ");
  xml << "        </Attribute>\n";
}

/// Rewrite the XDMF file from the contents of the HDF5 file.
void write_xdmf(hid_t file, const std::string& xdmf_path, const std::string& h5_name)
{
  std::ostringstream xml;
  xml << std::setprecision(17);
  xml << "<?xml version=\"1.0\" ?>\n"
      << "<Xdmf Version=\"3.0\">\n"
      << "  <Domain>\n"
      << "    <Grid Name=\"Results\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";

  // Cell arrays of earlier steps on the same mesh (Domain_ID, Proc_ID).
  std::map<std::string, Attribute> earlier_cell_arrays;
  int earlier_mesh = -1;

  for (int step : steps(file)) {
    const std::string step_path = "/Step/" + std::to_string(step);
    double time = 0.0;
    int mesh = 0;
    {
      Handle group = open_group(file, step_path);
      read_attribute(group, "Time", H5T_NATIVE_DOUBLE, &time);
      read_attribute(group, "Mesh", H5T_NATIVE_INT, &mesh);
    }

    const std::string mesh_path = "/Mesh/" + std::to_string(mesh);
    std::string topology_type;
    long long num_cells = 0;
    int nodes_per_cell = 0;
    {
      Handle group = open_group(file, mesh_path);
      topology_type = read_string_attribute(group, "TopologyType");
      read_attribute(group, "NumberOfElements", H5T_NATIVE_LLONG, &num_cells);
      read_attribute(group, "NodesPerElement", H5T_NATIVE_INT, &nodes_per_cell);
    }

    xml << "      <Grid Name=\"step_" << step << "\" GridType=\"Uniform\">\n"
        << "        <Time Value=\"" << time << "\" />\n"
        << "        <Topology TopologyType=\"" << topology_type << "\" NumberOfElements=\"" << num_cells << "\"";
    if (nodes_per_cell > 0) {
      xml << " NodesPerElement=\"" << nodes_per_cell << "\"";
    }
    xml << ">\n";
    write_data_item(xml, h5_name, data_item(file, mesh_path + "/Topology"), "          ");
    xml << "        </Topology>\n"
        << "        <Geometry GeometryType=\"XYZ\">\n";
    write_data_item(xml, h5_name, data_item(file, mesh_path + "/Geometry"), "          ");
    xml << "        </Geometry>\n";

    for (const auto& attribute : attributes(file, step_path + "/PointData", "Node")) {
      write_attribute_xml(xml, h5_name, attribute);
    }

    if (mesh != earlier_mesh) {
      earlier_cell_arrays.clear();
      earlier_mesh = mesh;
    }
    auto cell_arrays = attributes(file, step_path + "/CellData", "Cell");
    for (const auto& attribute : cell_arrays) {
      write_attribute_xml(xml, h5_name, attribute);
    }
    for (const auto& [name, attribute] : earlier_cell_arrays) {
      if (std::none_of(cell_arrays.begin(), cell_arrays.end(), [&](const Attribute& a) { return a.name == name; })) {
        write_attribute_xml(xml, h5_name, attribute);
      }
    }
    for (const auto& attribute : cell_arrays) {
      earlier_cell_arrays[attribute.name] = attribute;
    }

    xml << "      </Grid>\n";
  }

  xml << "    </Grid>\n"
      << "  </Domain>\n"
      << "</Xdmf>\n";

  // Replace the file only once the new one is complete.
  const std::string temporary = xdmf_path + ".tmp";
  {
    std::ofstream out(temporary);
    out << xml.str();
    if (!out) {
      hdf5_error("writing " + temporary);
    }
  }
  if (std::rename(temporary.c_str(), xdmf_path.c_str()) != 0) {
    hdf5_error("renaming " + temporary + " to " + xdmf_path);
  }
}

} // namespace

bool XdmfWriter::Mesh::operator==(const Mesh& other) const
{
  return type == other.type && num_cells == other.num_cells && nodes_per_cell == other.nodes_per_cell &&
         points == other.points && topology == other.topology;
}

XdmfWriter::XdmfWriter(const std::string& prefix, int continued_from)
    : h5_path_(prefix + ".h5"), xdmf_path_(prefix + ".xdmf"), continued_from_(continued_from)
{
  h5_name_ = h5_path_.substr(h5_path_.find_last_of('/') + 1);
}

void XdmfWriter::write(vtkUnstructuredGrid& results, int step, double time)
{
  hid_t file_id = -1;

  if (!started_) {
    // A continued simulation adds to the file of the one it continues;
    // otherwise the file starts anew, as the VTU files are overwritten.
    if (continued_from_ >= 0 && std::ifstream(h5_path_).good()) {
      file_id = H5Fopen(h5_path_.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    } else {
      file_id = H5Fcreate(h5_path_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      continued_from_ = -1;
    }
  } else {
    file_id = H5Fopen(h5_path_.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  }

  // The file is closed after every step, so that it is complete if the
  // simulation stops.
  Handle file(file_id, H5Fclose, "opening " + h5_path_);

  if (!started_) {
    for (const char* group : {"/Mesh", "/Step"}) {
      if (!exists(file, group)) {
        create_group(file, group);
      }
    }

    if (continued_from_ >= 0) {
      remove_steps(file, [&](int n) { return n > continued_from_; });

      // Reuse the mesh of the last step kept if it is still the same.
      const auto kept = steps(file);
      if (!kept.empty()) {
        Handle group = open_group(file, "/Step/" + std::to_string(kept.back()));
        read_attribute(group, "Mesh", H5T_NATIVE_INT, &mesh_id_);
        mesh_ = read_mesh(file, mesh_id_);
      }
    }

    started_ = true;
  }

  remove_steps(file, [&](int n) { return n >= step; });

  Mesh mesh = extract_mesh(results);

  if (mesh_id_ < 0 || !(mesh == mesh_)) {
    Handle meshes = open_group(file, "/Mesh");
    int id = 0;
    while (exists(meshes, std::to_string(id))) {
      id++;
    }
    write_mesh(file, id, mesh);
    mesh_id_ = id;
    mesh_ = std::move(mesh);
  }

  {
    Handle group = create_group(file, "/Step/" + std::to_string(step));
    write_attribute(group, "Time", H5T_NATIVE_DOUBLE, &time);
    write_attribute(group, "Mesh", H5T_NATIVE_INT, &mesh_id_);
    write_arrays(group, "PointData", *results.GetPointData());
    write_arrays(group, "CellData", *results.GetCellData());
  }

  check(H5Fflush(file, H5F_SCOPE_GLOBAL), "flushing " + h5_path_);
  write_xdmf(file, xdmf_path_, h5_name_);
}
