// R083 Phase-A standalone surface natural-neighbor weight builder.
//
// This source intentionally contains only the offline CGAL builder and its
// strict binary contract.  It has no trainer, checkpoint, or visualization
// integration.

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/enum.h>
#include <CGAL/number_utils.h>
#include <CGAL/surface_neighbor_coordinates_3.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Delaunay = CGAL::Delaunay_triangulation_3<Kernel>;
using Point = Kernel::Point_3;
using Vector = Kernel::Vector_3;
using Coordinate = std::pair<Point, Kernel::FT>;

constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint64_t kInputHeaderBytes = 32;
constexpr std::uint64_t kOutputHeaderBytes = 104;
constexpr std::uint64_t kOutputMethodBytes = 64;
// Model normals may be normalized in float32 and then converted to float64;
// retain a fixed 1e-5 contract tolerance without normalizing them here.
constexpr double kNormalNormTolerance = 1.0e-5;
constexpr double kRowSumTolerance = 1.0e-10;
constexpr std::uint64_t kDefaultProgressInterval = 10000;

constexpr char kInputMagic[] = "R083SNNI";
constexpr char kOutputMagic[] = "R083SNNO";
constexpr char kMethodIdentity[] =
    "CGAL.surface_neighbor_coordinates_3.Delaunay.v1";

static_assert(sizeof(kInputMagic) - 1 == 8, "input magic must be eight bytes");
static_assert(sizeof(kOutputMagic) - 1 == 8,
              "output magic must be eight bytes");
static_assert(sizeof(kMethodIdentity) - 1 <= kOutputMethodBytes,
              "method identity does not fit the output header");
static_assert(sizeof(double) == 8, "R083 requires eight-byte doubles");

class ContractError : public std::runtime_error {
 public:
  explicit ContractError(const std::string& message)
      : std::runtime_error(message) {}
};

std::uint64_t checked_add(
    std::uint64_t first,
    std::uint64_t second,
    const char* label) {
  if (first > std::numeric_limits<std::uint64_t>::max() - second) {
    throw ContractError(std::string(label) + " count arithmetic overflow");
  }
  return first + second;
}

std::uint64_t checked_mul(
    std::uint64_t first,
    std::uint64_t second,
    const char* label) {
  if (first != 0 && second > std::numeric_limits<std::uint64_t>::max() / first) {
    throw ContractError(std::string(label) + " count arithmetic overflow");
  }
  return first * second;
}

std::size_t checked_size(std::uint64_t value, const char* label) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw ContractError(std::string(label) + " is too large for this platform");
  }
  return static_cast<std::size_t>(value);
}

void require_little_endian_platform() {
  if (!std::numeric_limits<double>::is_iec559) {
    throw ContractError("R083 requires an IEEE-754 binary64 double platform");
  }
  const std::uint16_t marker = 1;
  unsigned char first_byte = 0;
  std::memcpy(&first_byte, &marker, sizeof(first_byte));
  if (first_byte != 1) {
    throw ContractError("R083 binary contract requires a little-endian platform");
  }
}

void require_absent_no_follow(
    const std::filesystem::path& path,
    const std::string& description) {
  std::error_code status_error;
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(path, status_error);
  if (status_error &&
      status_error !=
          std::make_error_code(std::errc::no_such_file_or_directory)) {
    throw ContractError("cannot inspect " + description + ": " +
                        status_error.message());
  }
  if (!status_error &&
      status.type() != std::filesystem::file_type::not_found) {
    throw ContractError("refusing existing " + description + ": " +
                        path.string());
  }
}

void require_output_destination_absent(
    const std::filesystem::path& destination,
    const char* phase) {
  require_absent_no_follow(
      destination,
      std::string("output destination before ") + phase +
          " (Phase A has no overwrite option)");
}

void write_bytes(std::ostream& output, const unsigned char* bytes, std::size_t count) {
  if (count == 0) {
    return;
  }
  output.write(reinterpret_cast<const char*>(bytes),
               static_cast<std::streamsize>(count));
  if (!output) {
    throw ContractError("failed while writing the temporary output file");
  }
}

void write_u32_le(std::ostream& output, std::uint32_t value) {
  std::array<unsigned char, 4> bytes{
      static_cast<unsigned char>(value & 0xffU),
      static_cast<unsigned char>((value >> 8U) & 0xffU),
      static_cast<unsigned char>((value >> 16U) & 0xffU),
      static_cast<unsigned char>((value >> 24U) & 0xffU),
  };
  write_bytes(output, bytes.data(), bytes.size());
}

void write_u64_le(std::ostream& output, std::uint64_t value) {
  std::array<unsigned char, 8> bytes{};
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    bytes[index] = static_cast<unsigned char>(value >> (8U * index));
  }
  write_bytes(output, bytes.data(), bytes.size());
}

void write_f64_le(std::ostream& output, double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  write_u64_le(output, bits);
}

class ByteReader {
 public:
  explicit ByteReader(const std::vector<unsigned char>& bytes) : bytes_(bytes) {}

  std::uint32_t read_u32_le() {
    require(4);
    const std::uint32_t value =
        static_cast<std::uint32_t>(bytes_[offset_]) |
        (static_cast<std::uint32_t>(bytes_[offset_ + 1]) << 8U) |
        (static_cast<std::uint32_t>(bytes_[offset_ + 2]) << 16U) |
        (static_cast<std::uint32_t>(bytes_[offset_ + 3]) << 24U);
    offset_ += 4;
    return value;
  }

  std::uint64_t read_u64_le() {
    require(8);
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < 8; ++index) {
      value |= static_cast<std::uint64_t>(bytes_[offset_ + index])
               << (8U * index);
    }
    offset_ += 8;
    return value;
  }

  double read_f64_le() {
    const std::uint64_t bits = read_u64_le();
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  void read_bytes(unsigned char* destination, std::size_t count) {
    require(count);
    if (count != 0) {
      std::memcpy(destination, bytes_.data() + offset_, count);
    }
    offset_ += count;
  }

  std::size_t offset() const { return offset_; }

 private:
  void require(std::size_t count) const {
    if (offset_ > bytes_.size() || count > bytes_.size() - offset_) {
      throw ContractError("truncated R083 input payload");
    }
  }

  const std::vector<unsigned char>& bytes_;
  std::size_t offset_ = 0;
};

std::vector<unsigned char> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw ContractError("cannot open input file: " + path.string());
  }
  const std::streamoff end = input.tellg();
  if (end < 0) {
    throw ContractError("cannot determine input file length: " + path.string());
  }
  const auto file_size = static_cast<std::uintmax_t>(end);
  if (file_size > std::numeric_limits<std::uint64_t>::max() ||
      file_size > std::numeric_limits<std::size_t>::max() ||
      file_size > std::numeric_limits<std::streamsize>::max()) {
    throw ContractError("input file is too large for this platform");
  }
  const std::size_t size = static_cast<std::size_t>(file_size);
  std::vector<unsigned char> bytes(size);
  input.seekg(0, std::ios::beg);
  if (size != 0 &&
      !input.read(reinterpret_cast<char*>(bytes.data()),
                  static_cast<std::streamsize>(size))) {
    throw ContractError("failed while reading input file: " + path.string());
  }
  if (static_cast<std::size_t>(input.gcount()) != size && size != 0) {
    throw ContractError("truncated input file: " + path.string());
  }
  return bytes;
}

struct PointLexicographicLess {
  using Compare = Kernel::Compare_xyz_3;

  PointLexicographicLess() : compare(Kernel().compare_xyz_3_object()) {}

  bool operator()(const Point& first, const Point& second) const {
    return compare(first, second) == CGAL::SMALLER;
  }

  Compare compare;
};

using GuidePointMap = std::map<Point, std::uint32_t, PointLexicographicLess>;

struct InputData {
  std::vector<Point> guide_points;
  std::vector<Point> query_points;
  std::vector<Vector> query_normals;
  GuidePointMap guide_ids;
};

double checked_coordinate(ByteReader& reader, const char* name, std::size_t index) {
  const double value = reader.read_f64_le();
  if (!std::isfinite(value)) {
    throw ContractError(std::string(name) + " contains a non-finite value at index " +
                        std::to_string(index));
  }
  return value;
}

InputData read_input(const std::filesystem::path& path) {
  const std::vector<unsigned char> bytes = read_file(path);
  if (bytes.size() < kInputHeaderBytes) {
    throw ContractError("truncated R083 input header");
  }

  ByteReader reader(bytes);
  std::array<unsigned char, 8> magic{};
  reader.read_bytes(magic.data(), magic.size());
  if (std::memcmp(magic.data(), kInputMagic, magic.size()) != 0) {
    throw ContractError("bad R083 input magic");
  }
  const std::uint32_t version = reader.read_u32_le();
  if (version != kFormatVersion) {
    throw ContractError("unsupported R083 input version: " +
                        std::to_string(version));
  }
  const std::uint32_t header_bytes = reader.read_u32_le();
  if (header_bytes != kInputHeaderBytes) {
    throw ContractError("bad R083 input header size: " +
                        std::to_string(header_bytes));
  }
  const std::uint64_t guide_count = reader.read_u64_le();
  const std::uint64_t query_count = reader.read_u64_le();
  if (guide_count == 0) {
    throw ContractError("guide count must be positive");
  }
  if (guide_count > std::numeric_limits<std::uint32_t>::max()) {
    throw ContractError("guide count exceeds the uint32 guide-ID contract");
  }

  const std::uint64_t guide_bytes = checked_mul(guide_count, 3 * 8, "input guide");
  const std::uint64_t query_bytes = checked_mul(query_count, 3 * 8, "input query");
  const std::uint64_t normal_bytes = checked_mul(query_count, 3 * 8, "input normal");
  std::uint64_t expected_bytes = kInputHeaderBytes;
  expected_bytes = checked_add(expected_bytes, guide_bytes, "input file");
  expected_bytes = checked_add(expected_bytes, query_bytes, "input file");
  expected_bytes = checked_add(expected_bytes, normal_bytes, "input file");
  if (expected_bytes != bytes.size()) {
    if (expected_bytes > bytes.size()) {
      throw ContractError("truncated R083 input: expected " +
                          std::to_string(expected_bytes) + " bytes, got " +
                          std::to_string(bytes.size()));
    }
    throw ContractError("trailing bytes in R083 input: expected " +
                        std::to_string(expected_bytes) + " bytes, got " +
                        std::to_string(bytes.size()));
  }

  const std::size_t guide_size = checked_size(guide_count, "guide count");
  const std::size_t query_size = checked_size(query_count, "query count");
  InputData input;
  if (guide_size > input.guide_points.max_size() ||
      query_size > input.query_points.max_size()) {
    throw ContractError("R083 input counts exceed vector capacity");
  }
  input.guide_points.reserve(guide_size);
  input.query_points.reserve(query_size);
  input.query_normals.reserve(query_size);

  for (std::size_t guide_index = 0; guide_index < guide_size; ++guide_index) {
    const double x = checked_coordinate(reader, "guide points", guide_index * 3);
    const double y = checked_coordinate(reader, "guide points", guide_index * 3 + 1);
    const double z = checked_coordinate(reader, "guide points", guide_index * 3 + 2);
    const Point point(x, y, z);
    const auto inserted = input.guide_ids.emplace(
        point, static_cast<std::uint32_t>(guide_index));
    if (!inserted.second) {
      throw ContractError("duplicate guide coordinates in input");
    }
    input.guide_points.push_back(point);
  }

  for (std::size_t query_index = 0; query_index < query_size; ++query_index) {
    const double x = checked_coordinate(reader, "query points", query_index * 3);
    const double y = checked_coordinate(reader, "query points", query_index * 3 + 1);
    const double z = checked_coordinate(reader, "query points", query_index * 3 + 2);
    input.query_points.emplace_back(x, y, z);
  }

  for (std::size_t query_index = 0; query_index < query_size; ++query_index) {
    const double x = checked_coordinate(reader, "query normals", query_index * 3);
    const double y = checked_coordinate(reader, "query normals", query_index * 3 + 1);
    const double z = checked_coordinate(reader, "query normals", query_index * 3 + 2);
    const double norm = std::hypot(std::hypot(x, y), z);
    if (!std::isfinite(norm) || std::abs(norm - 1.0) > kNormalNormTolerance) {
      throw ContractError(
          "query normal at index " + std::to_string(query_index) +
          " is not unit length within tolerance " +
          std::to_string(kNormalNormTolerance) +
          "; normals are not normalized silently");
    }
    input.query_normals.emplace_back(x, y, z);
  }
  if (reader.offset() != bytes.size()) {
    throw ContractError("R083 input parser did not consume the complete file");
  }
  return input;
}

struct Neighbor {
  std::uint32_t guide_id = 0;
  double weight = 0.0;
  Point point = Point(0.0, 0.0, 0.0);
};

struct OutputData {
  std::vector<std::uint64_t> row_offsets;
  std::vector<std::uint32_t> guide_ids;
  std::vector<double> weights;
  std::vector<std::uint8_t> success;
  std::vector<double> barycentric_errors;
};

std::uint64_t output_size(
    std::uint64_t query_count,
    std::uint64_t nnz) {
  const std::uint64_t offset_count = checked_add(query_count, 1, "output row offset");
  std::uint64_t total = kOutputHeaderBytes;
  total = checked_add(
      total,
      checked_mul(offset_count, 8, "output row offset"),
      "output file");
  total = checked_add(total, checked_mul(nnz, 4, "output guide ID"), "output file");
  total = checked_add(total, checked_mul(nnz, 8, "output weight"), "output file");
  total = checked_add(total, query_count, "output success flag");
  total = checked_add(
      total,
      checked_mul(query_count, 8, "output barycentric error"),
      "output file");
  return total;
}

OutputData build_weights(
    const InputData& input,
    std::uint64_t progress_interval) {
  const std::size_t query_count = input.query_points.size();
  if (query_count == 0) {
    // An empty query batch is a valid CSR artifact.  Positive, unique guide
    // coordinates were already checked while parsing; no 3D triangulation is
    // needed when there is no query to evaluate.
    OutputData empty;
    empty.row_offsets.push_back(0);
    output_size(0, 0);
    return empty;
  }

  Delaunay delaunay(input.guide_points.begin(), input.guide_points.end());
  if (delaunay.number_of_vertices() != input.guide_points.size()) {
    throw ContractError("Delaunay vertex count does not equal guide count");
  }
  if (delaunay.dimension() != 3) {
    throw ContractError("guide sites are undersampled or not full-dimensional");
  }
  if (!delaunay.is_valid()) {
    throw ContractError("constructed Delaunay triangulation is invalid");
  }

  OutputData output;
  output.row_offsets.reserve(query_count + 1);
  output.success.reserve(query_count);
  output.barycentric_errors.reserve(query_count);
  output.row_offsets.push_back(0);

  const std::size_t guide_count = input.guide_points.size();
  std::vector<std::uint64_t> seen_generation(guide_count, 0);
  std::uint64_t nnz = 0;

  for (std::size_t query_index = 0; query_index < query_count; ++query_index) {
    const std::uint64_t query_number = static_cast<std::uint64_t>(query_index) + 1;
    std::vector<Coordinate> coordinates;
    // This is the sole coordinate routine used for each query.  In
    // particular, there is no local candidate filter or fallback path.
    const auto result = CGAL::surface_neighbor_coordinates_3(
        delaunay,
        input.query_points[query_index],
        input.query_normals[query_index],
        std::back_inserter(coordinates));
    if (!result.third) {
      throw ContractError("surface-neighbor coordinate query " +
                          std::to_string(query_index) + " returned success=false");
    }

    const double normalization = CGAL::to_double(result.second);
    if (!std::isfinite(normalization) || normalization <= 0.0) {
      throw ContractError("query " + std::to_string(query_index) +
                          " returned a non-finite or non-positive normalization");
    }

    std::vector<Neighbor> neighbors;
    neighbors.reserve(coordinates.size());
    const std::uint64_t generation = query_number;
    for (const Coordinate& coordinate : coordinates) {
      const auto found = input.guide_ids.find(coordinate.first);
      if (found == input.guide_ids.end() || !(found->first == coordinate.first)) {
        throw ContractError("surface-neighbor query " + std::to_string(query_index) +
                            " returned a point that does not exactly match one guide");
      }
      const std::uint32_t guide_id = found->second;
      if (seen_generation[guide_id] == generation) {
        throw ContractError("surface-neighbor query " + std::to_string(query_index) +
                            " returned a duplicate guide ID");
      }
      seen_generation[guide_id] = generation;

      const double raw_weight = CGAL::to_double(coordinate.second);
      if (!std::isfinite(raw_weight) || raw_weight < 0.0) {
        throw ContractError("query " + std::to_string(query_index) +
                            " returned a non-finite or negative raw weight");
      }
      const double normalized_weight = raw_weight / normalization;
      if (!std::isfinite(normalized_weight) || normalized_weight < 0.0) {
        throw ContractError("query " + std::to_string(query_index) +
                            " returned a non-finite or negative normalized weight");
      }
      neighbors.push_back(Neighbor{guide_id, normalized_weight, coordinate.first});
    }

    std::sort(
        neighbors.begin(),
        neighbors.end(),
        [](const Neighbor& first, const Neighbor& second) {
          return first.guide_id < second.guide_id;
        });

    double row_sum = 0.0;
    long double reconstruction_x = 0.0L;
    long double reconstruction_y = 0.0L;
    long double reconstruction_z = 0.0L;
    for (const Neighbor& neighbor : neighbors) {
      row_sum += neighbor.weight;
      reconstruction_x += static_cast<long double>(neighbor.weight) *
                          static_cast<long double>(CGAL::to_double(neighbor.point.x()));
      reconstruction_y += static_cast<long double>(neighbor.weight) *
                          static_cast<long double>(CGAL::to_double(neighbor.point.y()));
      reconstruction_z += static_cast<long double>(neighbor.weight) *
                          static_cast<long double>(CGAL::to_double(neighbor.point.z()));
    }
    if (!std::isfinite(row_sum) ||
        std::abs(row_sum - 1.0) > kRowSumTolerance) {
      throw ContractError("bad normalized row sum for query " +
                          std::to_string(query_index));
    }

    const long double query_x =
        static_cast<long double>(CGAL::to_double(input.query_points[query_index].x()));
    const long double query_y =
        static_cast<long double>(CGAL::to_double(input.query_points[query_index].y()));
    const long double query_z =
        static_cast<long double>(CGAL::to_double(input.query_points[query_index].z()));
    const long double dx = reconstruction_x - query_x;
    const long double dy = reconstruction_y - query_y;
    const long double dz = reconstruction_z - query_z;
    const long double squared_error = dx * dx + dy * dy + dz * dz;
    if (!std::isfinite(squared_error) || squared_error < 0.0L) {
      throw ContractError("non-finite barycentric reconstruction error for query " +
                          std::to_string(query_index));
    }
    const long double error_long_double = std::sqrt(squared_error);
    const double barycentric_error = static_cast<double>(error_long_double);
    if (!std::isfinite(barycentric_error) || barycentric_error < 0.0) {
      throw ContractError("non-finite barycentric reconstruction error for query " +
                          std::to_string(query_index));
    }

    const std::uint64_t row_nnz = static_cast<std::uint64_t>(neighbors.size());
    nnz = checked_add(nnz, row_nnz, "output nnz");
    for (const Neighbor& neighbor : neighbors) {
      output.guide_ids.push_back(neighbor.guide_id);
      output.weights.push_back(neighbor.weight);
    }
    output.row_offsets.push_back(nnz);
    output.success.push_back(1);
    output.barycentric_errors.push_back(barycentric_error);

    if (progress_interval != 0 &&
        (query_number % progress_interval == 0 ||
         query_number == static_cast<std::uint64_t>(query_count))) {
      std::cout << "R083_PROGRESS query=" << query_number
                << " queries=" << query_count << " nnz=" << nnz << "\n"
                << std::flush;
    }
  }
  if (output.row_offsets.size() != query_count + 1 ||
      output.success.size() != query_count ||
      output.barycentric_errors.size() != query_count ||
      output.guide_ids.size() != output.weights.size()) {
    throw ContractError("internal CSR size mismatch");
  }
  output_size(static_cast<std::uint64_t>(query_count), nnz);
  return output;
}

std::filesystem::path reserve_temporary_sibling_directory(
    const std::filesystem::path& destination) {
  for (std::uint64_t attempt = 0; attempt < 10000; ++attempt) {
    std::filesystem::path candidate = destination;
    candidate += ".r083.tmp." + std::to_string(attempt);
    std::error_code create_error;
    if (std::filesystem::create_directory(candidate, create_error)) {
      return candidate;
    }
    // create_directory is the exclusive reservation: it cannot turn an
    // existing file or symlink into our temporary directory.  Some standard
    // library implementations report an existing path without an error;
    // both forms are treated as a collision and retried.
    if (!create_error ||
        create_error == std::make_error_code(std::errc::file_exists)) {
      continue;
    }
    throw ContractError("cannot reserve temporary output directory: " +
                        create_error.message());
  }
  throw ContractError("could not reserve a temporary sibling output directory");
}

void cleanup_temporary_directory(const std::filesystem::path& directory) {
  std::error_code cleanup_error;
  std::filesystem::remove_all(directory, cleanup_error);
  if (cleanup_error) {
    throw ContractError("temporary output cleanup failed for " +
                        directory.string() + ": " + cleanup_error.message());
  }
}

void write_output_atomic(
    const std::filesystem::path& destination,
    std::uint64_t guide_count,
    std::uint64_t query_count,
    const OutputData& output) {
  if (output.row_offsets.size() != query_count + 1 ||
      output.success.size() != query_count ||
      output.barycentric_errors.size() != query_count ||
      output.guide_ids.size() != output.weights.size()) {
    throw ContractError("cannot serialize inconsistent CSR output");
  }
  const std::uint64_t nnz = static_cast<std::uint64_t>(output.guide_ids.size());
  const std::uint64_t expected_bytes = output_size(query_count, nnz);
  require_output_destination_absent(destination, "output construction");
  const std::filesystem::path temporary_directory =
      reserve_temporary_sibling_directory(destination);
  try {
    const std::filesystem::path payload = temporary_directory / "payload.bin";
    require_absent_no_follow(payload, "temporary output payload");
    {
      std::ofstream file(payload, std::ios::binary | std::ios::trunc);
      if (!file) {
        throw ContractError("cannot open temporary output file: " +
                            payload.string());
      }
      write_bytes(
          file,
          reinterpret_cast<const unsigned char*>(kOutputMagic),
          8);
      write_u32_le(file, kFormatVersion);
      write_u32_le(file, static_cast<std::uint32_t>(kOutputHeaderBytes));
      write_u64_le(file, guide_count);
      write_u64_le(file, query_count);
      write_u64_le(file, nnz);
      std::array<unsigned char, kOutputMethodBytes> method{};
      std::memcpy(method.data(), kMethodIdentity, sizeof(kMethodIdentity) - 1);
      write_bytes(file, method.data(), method.size());

      for (const std::uint64_t offset : output.row_offsets) {
        write_u64_le(file, offset);
      }
      for (const std::uint32_t guide_id : output.guide_ids) {
        write_u32_le(file, guide_id);
      }
      for (const double weight : output.weights) {
        write_f64_le(file, weight);
      }
      for (const std::uint8_t success : output.success) {
        const unsigned char byte = success;
        write_bytes(file, &byte, 1);
      }
      for (const double error : output.barycentric_errors) {
        write_f64_le(file, error);
      }
      file.flush();
      if (!file) {
        throw ContractError("failed while finalizing temporary output file");
      }
    }

    std::error_code payload_status_error;
    const std::filesystem::file_status payload_status =
        std::filesystem::symlink_status(payload, payload_status_error);
    if (payload_status_error ||
        payload_status.type() != std::filesystem::file_type::regular) {
      throw ContractError("temporary output payload is not a regular non-symlink file");
    }
    std::error_code payload_size_error;
    const std::uintmax_t actual_bytes =
        std::filesystem::file_size(payload, payload_size_error);
    if (payload_size_error) {
      throw ContractError("cannot determine temporary output byte count: " +
                          payload_size_error.message());
    }
    if (actual_bytes != expected_bytes) {
      throw ContractError("temporary output byte count mismatch: expected " +
                          std::to_string(expected_bytes) + ", got " +
                          std::to_string(actual_bytes));
    }

    require_output_destination_absent(destination, "publication");
    std::error_code rename_error;
    std::filesystem::rename(payload, destination, rename_error);
    if (rename_error) {
      throw ContractError("atomic output rename failed: " + rename_error.message());
    }

    cleanup_temporary_directory(temporary_directory);
  } catch (...) {
    const std::exception_ptr failure = std::current_exception();
    try {
      cleanup_temporary_directory(temporary_directory);
    } catch (const std::exception& cleanup_error) {
      throw ContractError(
          std::string("R083 output operation failed; temporary cleanup failed: ") +
          cleanup_error.what());
    }
    std::rethrow_exception(failure);
  }
}

struct Options {
  std::filesystem::path input;
  std::filesystem::path output;
  std::uint64_t progress_interval = kDefaultProgressInterval;
};

std::uint64_t parse_uint64(const std::string& text, const char* option) {
  if (text.empty()) {
    throw ContractError(std::string(option) + " requires a positive integer");
  }
  std::uint64_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size() ||
      value == 0) {
    throw ContractError(std::string(option) + " requires a positive uint64 integer");
  }
  return value;
}

void print_usage(std::ostream& stream) {
  stream << "usage: surface_natural_neighbor_weights --input <bin> --output <bin> "
            "[--progress-interval N]\n"
            "Phase A refuses an existing output destination; no overwrite option is available.\n";
}

Options parse_options(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    print_usage(std::cout);
    std::exit(0);
  }
  Options options;
  bool have_input = false;
  bool have_output = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--input" || argument == "--output" ||
        argument == "--progress-interval") {
      if (index + 1 >= argc) {
        throw ContractError(argument + " requires a value");
      }
      const std::string value = argv[++index];
      if (argument == "--input") {
        if (have_input) {
          throw ContractError("--input was specified more than once");
        }
        if (value.empty()) {
          throw ContractError("--input requires a non-empty path");
        }
        options.input = value;
        have_input = true;
      } else if (argument == "--output") {
        if (have_output) {
          throw ContractError("--output was specified more than once");
        }
        if (value.empty()) {
          throw ContractError("--output requires a non-empty path");
        }
        options.output = value;
        have_output = true;
      } else {
        options.progress_interval = parse_uint64(value, "--progress-interval");
      }
    } else {
      throw ContractError("unknown argument: " + argument);
    }
  }
  if (!have_input || !have_output) {
    print_usage(std::cerr);
    throw ContractError("both --input and --output are required");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require_little_endian_platform();
    if (argc < 2) {
      print_usage(std::cerr);
      throw ContractError("arguments are required");
    }
    const Options options = parse_options(argc, argv);
    // Refuse an existing destination before reading or triangulating any
    // input.  Phase A intentionally has no overwrite option.
    require_output_destination_absent(options.output, "computation");
    const InputData input = read_input(options.input);
    std::cout << std::setprecision(17)
              << "R083_STATUS=ready guides=" << input.guide_points.size()
              << " queries=" << input.query_points.size()
              << " progress_interval=" << options.progress_interval
              << " normal_norm_tolerance=" << kNormalNormTolerance
              << " row_sum_tolerance=" << kRowSumTolerance
              << " method=" << kMethodIdentity << "\n"
              << std::flush;

    OutputData output = build_weights(input, options.progress_interval);
    write_output_atomic(
        options.output,
        static_cast<std::uint64_t>(input.guide_points.size()),
        static_cast<std::uint64_t>(input.query_points.size()),
        output);
    std::cout << "R083_STATUS=complete guides=" << input.guide_points.size()
              << " queries=" << input.query_points.size()
              << " nnz=" << output.guide_ids.size()
              << " method=" << kMethodIdentity << "\n"
              << std::flush;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "R083_ERROR=" << error.what() << "\n";
    return 1;
  }
}
